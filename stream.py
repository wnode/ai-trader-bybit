"""
X API v2 Filtered Stream — escuta tweets sobre Bitcoin em tempo real.
Roda em thread daemon e comunica com o main loop via fila thread-safe.
"""
import json
import time
import queue
import logging
import threading
from datetime import datetime, timezone

import requests

import config as cfg

logger = logging.getLogger(__name__)

STREAM_URL = "https://api.x.com/2/tweets/search/stream"
RULES_URL = "https://api.x.com/2/tweets/search/stream/rules"

URGENT_KEYWORDS = {
    "crash", "hack", "hacked", "exploit", "ban", "banned", "sec",
    "etf", "approved", "rejected", "liquidat", "whale", "dump",
    "pump", "flash crash", "black swan", "regulation", "arrest",
    "fraud", "insolvent", "bankrupt", "emergency", "breaking",
}

HIGH_KEYWORDS = {
    "fed", "fomc", "interest rate", "inflation", "cpi",
    "halving", "fork", "upgrade", "lawsuit", "subpoena",
    "treasury", "executive order", "tariff",
}


def _classify_urgency(text: str) -> int:
    """Classifica urgencia de um tweet (1-10) baseado em keywords."""
    lower = text.lower()
    for kw in URGENT_KEYWORDS:
        if kw in lower:
            return 9
    for kw in HIGH_KEYWORDS:
        if kw in lower:
            return 7
    return 3


class XStreamListener:
    """Escuta X API v2 filtered stream em background thread."""

    def __init__(self, alert_queue: queue.Queue):
        self._queue = alert_queue
        self._bearer = cfg.X_BEARER_TOKEN
        self._running = False
        self._thread: threading.Thread | None = None
        self._keywords = [k.strip() for k in cfg.X_STREAM_KEYWORDS.split(",") if k.strip()]
        self._urgency_threshold = cfg.X_STREAM_URGENCY_THRESHOLD

    def start(self):
        """Inicia thread daemon do stream."""
        if not self._bearer:
            logger.warning("[STREAM] X_BEARER_TOKEN nao configurado — stream desativado")
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="x-stream")
        self._thread.start()
        logger.info(f"[STREAM] Thread iniciada — keywords: {self._keywords}")

    def stop(self):
        """Para o stream."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            logger.info("[STREAM] Thread encerrada")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._bearer}"}

    def _setup_rules(self):
        """Configura regras de filtro no stream."""
        resp = requests.get(RULES_URL, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        existing = resp.json().get("data", [])
        if existing:
            ids = [r["id"] for r in existing]
            requests.post(
                RULES_URL,
                headers=self._headers(),
                json={"delete": {"ids": ids}},
                timeout=30,
            )
            logger.info(f"[STREAM] {len(ids)} regras antigas removidas")

        rule_value = " OR ".join(self._keywords)
        if len(rule_value) > 512:
            rule_value = rule_value[:512]

        resp = requests.post(
            RULES_URL,
            headers=self._headers(),
            json={"add": [{"value": rule_value, "tag": "btc-sentiment"}]},
            timeout=30,
        )
        resp.raise_for_status()
        logger.info(f"[STREAM] Regra configurada: {rule_value}")

    def _run(self):
        """Loop principal do stream com reconexao automatica."""
        backoff = 1
        max_backoff = 300

        while self._running:
            try:
                self._setup_rules()
                self._listen()
                backoff = 1
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    retry_after = int(e.response.headers.get("retry-after", backoff))
                    logger.warning(f"[STREAM] Rate limited — aguardando {retry_after}s")
                    time.sleep(retry_after)
                elif e.response is not None and e.response.status_code in (401, 403):
                    logger.error(f"[STREAM] Erro de autenticacao ({e.response.status_code}) — verificar X_BEARER_TOKEN")
                    self._running = False
                    return
                else:
                    logger.warning(f"[STREAM] HTTP {e} — reconectando em {backoff}s")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                logger.warning(f"[STREAM] Conexao perdida: {e} — reconectando em {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except Exception as e:
                logger.error(f"[STREAM] Erro inesperado: {e} — reconectando em {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)

    def _listen(self):
        """Conecta ao stream e processa tweets."""
        params = {"tweet.fields": "created_at,author_id,public_metrics,text"}
        resp = requests.get(
            STREAM_URL,
            headers=self._headers(),
            params=params,
            stream=True,
            timeout=90,
        )
        resp.raise_for_status()
        logger.info("[STREAM] Conectado ao X API v2 filtered stream")

        for line in resp.iter_lines():
            if not self._running:
                break
            if not line:
                continue
            try:
                data = json.loads(line)
                tweet = data.get("data", {})
                text = tweet.get("text", "")
                if not text:
                    continue

                urgency = _classify_urgency(text)
                metrics = tweet.get("public_metrics", {})
                retweets = metrics.get("retweet_count", 0)
                likes = metrics.get("like_count", 0)

                if retweets > 1000 or likes > 5000:
                    urgency = min(10, urgency + 2)

                if urgency >= self._urgency_threshold:
                    alert = {
                        "text": text[:500],
                        "urgency": urgency,
                        "author_id": tweet.get("author_id", ""),
                        "retweets": retweets,
                        "likes": likes,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    self._queue.put(alert)
                    logger.info(
                        f"[STREAM] ALERTA urgencia={urgency}/10 | "
                        f"RT={retweets} Likes={likes} | {text[:100]}..."
                    )

            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.warning(f"[STREAM] Erro ao processar tweet: {e}")


class StreamAlertManager:
    """Gerencia alertas do stream e integra com o bot."""

    def __init__(self):
        self.queue: queue.Queue = queue.Queue()
        self._listener: XStreamListener | None = None
        self._recent_alerts: list[dict] = []

    def start(self):
        """Inicia o listener se configurado."""
        if not cfg.X_STREAM_ENABLED:
            logger.info("[STREAM] X Stream desativado (X_STREAM_ENABLED=false)")
            return
        self._listener = XStreamListener(self.queue)
        self._listener.start()

    def stop(self):
        """Para o listener."""
        if self._listener:
            self._listener.stop()

    def check_alerts(self) -> bool:
        """Checa se ha alertas urgentes na fila. Retorna True se houver."""
        found = False
        while not self.queue.empty():
            try:
                alert = self.queue.get_nowait()
                self._recent_alerts.append(alert)
                if len(self._recent_alerts) > 20:
                    self._recent_alerts = self._recent_alerts[-20:]
                found = True
            except queue.Empty:
                break
        return found

    def format_for_llm(self) -> str:
        """Formata alertas recentes como texto para a LLM."""
        if not self._recent_alerts:
            return ""
        lines = ["=== ALERTAS DO X (TEMPO REAL) ==="]
        for a in self._recent_alerts[-5:]:
            lines.append(
                f"[urgencia={a['urgency']}/10 | RT={a['retweets']} Likes={a['likes']}] "
                f"{a['text'][:200]}"
            )
        return "\n".join(lines)

    def clear_alerts(self):
        """Limpa alertas apos serem processados pela LLM."""
        self._recent_alerts.clear()
