"""
Sentimento de mercado — Fear & Greed Index (alternative.me).
Cache em memoria para evitar chamadas desnecessarias (API atualiza 1x/dia).
"""
import time
import logging
import requests

import config as cfg

logger = logging.getLogger(__name__)

FNG_API_URL = "https://api.alternative.me/fng/"


class SentimentData:
    """Coleta e formata dados de sentimento de mercado."""

    def __init__(self):
        self._fng_cache: dict | None = None
        self._fng_cache_time: float = 0

    def get_fear_greed(self) -> dict | None:
        """Busca Fear & Greed Index com cache. Retorna None se falhar."""
        now = time.time()
        cache_seconds = cfg.FNG_CACHE_MINUTES * 60

        if self._fng_cache and (now - self._fng_cache_time) < cache_seconds:
            return self._fng_cache

        try:
            resp = requests.get(FNG_API_URL, params={"limit": 2}, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            if data.get("metadata", {}).get("error"):
                logger.warning(f"[FNG] API retornou erro: {data['metadata']['error']}")
                return self._fng_cache

            items = data.get("data", [])
            if not items:
                logger.warning("[FNG] API retornou lista vazia")
                return self._fng_cache

            today = items[0]
            yesterday = items[1] if len(items) > 1 else None

            result = {
                "value": int(today["value"]),
                "classification": today["value_classification"],
                "yesterday_value": int(yesterday["value"]) if yesterday else None,
                "yesterday_classification": yesterday["value_classification"] if yesterday else None,
            }

            self._fng_cache = result
            self._fng_cache_time = now
            logger.info(f"[FNG] Fear & Greed: {result['value']}/100 ({result['classification']})")
            return result

        except Exception as e:
            logger.warning(f"[FNG] Erro ao buscar Fear & Greed: {e}")
            return self._fng_cache

    def format_for_llm(self) -> str:
        """Formata sentimento como texto para a LLM. Retorna string vazia se desabilitado."""
        if not cfg.USE_SENTIMENT:
            return ""

        fng = self.get_fear_greed()
        if not fng:
            return ""

        lines = ["=== SENTIMENTO DE MERCADO ==="]
        lines.append(f"Fear & Greed Index: {fng['value']}/100 ({fng['classification']})")

        if fng["yesterday_value"] is not None:
            diff = fng["value"] - fng["yesterday_value"]
            trend = "subindo" if diff > 0 else "caindo" if diff < 0 else "estavel"
            lines.append(f"Ontem: {fng['yesterday_value']}/100 ({fng['yesterday_classification']}) - tendencia: {trend}")

        return "\n".join(lines)
