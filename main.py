"""
AI Trader Bot — Main Loop
==========================
Bot de trading que usa LLM (Claude/Gemini/GPT) para analisar o mercado e decidir trades.

Uso:
    python main.py              # dry run (padrao)
    python main.py --live       # ordens reais
    python main.py --once       # roda uma vez e sai
"""
import os
import sys
import io
import time
import signal
import logging
import argparse
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import config as cfg
from pybit.unified_trading import HTTP
from market_data import MarketData
from analyst import create_analyst
from executor import TradeExecutor
from monitor import show_status
from sentiment import SentimentData
from stream import StreamAlertManager
from leader import LeaderSignal, SIGNAL_WARMUP_BARS
import db

# Logging — console + arquivo
os.makedirs("logs", exist_ok=True)
log_format = '%(asctime)s | %(levelname)-8s | %(message)s'
log_datefmt = '%Y-%m-%d %H:%M:%S'

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    datefmt=log_datefmt,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def _sync_bybit_time():
    """Sincroniza timestamp local com servidor Bybit via monkey-patch."""
    try:
        import pybit._helpers as _helpers
        client = HTTP(testnet=cfg.USE_TESTNET)
        result = client.get_server_time()
        server_ms = int(result["result"]["timeSecond"]) * 1000
        local_ms = int(time.time() * 1000)
        offset = server_ms - local_ms
        if abs(offset) > 500:  # Mais de 500ms de drift
            _original = _helpers.generate_timestamp
            _helpers.generate_timestamp = lambda: _original() + offset
            logger.info(f"[SYNC] Clock drift detectado: {offset:+d}ms — timestamp ajustado")
        else:
            logger.info(f"[SYNC] Clock OK (drift: {offset:+d}ms)")
    except Exception as e:
        logger.warning(f"[SYNC] Falha ao sincronizar timestamp: {e}")


def print_banner(dry_run: bool, analyst):
    mode = "DRY RUN" if dry_run else "LIVE"
    symbols_str = ", ".join(cfg.SYMBOLS)
    if cfg.USE_LLM:
        title = ">>> AI TRADER BOT — LLM-Powered Trading <<<"
        engine = f"Provider: {analyst.provider_name} | Model: {analyst.model}"
        steps = ("|  2. Envia a LLM para analise (um por simbolo)                |\n"
                 "    |  3. LLM decide: LONG / SHORT / HOLD / CLOSE                  |")
    else:
        title = ">>> AI TRADER BOT — Mecanico (lider BTC/ETH, sem LLM) <<<"
        engine = f"Estrategia: direcao BTC/ETH | TF: {cfg.TIMEFRAME}min"
        steps = ("|  2. Le a direcao do lider BTC/ETH (flip)                     |\n"
                 "    |  3. Regra: bullish->LONG, bearish->SHORT, fecha na virada    |")
    print(f"""
    +==============================================================+
    |  {title:<60s}|
    +--------------------------------------------------------------+
    |  {engine:<60s}|
    |  Symbols: {symbols_str:<48s}|
    |  Mode: {mode:<51s}|
    +--------------------------------------------------------------+
    |  1. Coleta dados de cada simbolo (klines + indicadores)      |
    {steps}
    |  4. Executor abre/fecha posicao na Bybit                     |
    |  5. Repete a cada {cfg.CHECK_INTERVAL}s                                       |
    +==============================================================+
    """)


def build_mechanical_decision(sym, market, active_trade, leader, trader):
    """Decisao SEM LLM: guiada pela direcao do lider BTC/ETH.
    - flat + lider vira p/ uma direcao (flip) -> LONG/SHORT
    - posicao aberta -> HOLD; CLOSE se o lider inverte (LEADER_CLOSE_ON_FLIP)
    Estado do flip guardado em trader['last_dir']. Entry/SL a partir do preco e ATR
    atuais (o executor recalcula o TP em MIN_RR_RATIO:1)."""
    d = leader.get_direction(sym)
    prev = trader.get("last_dir", 0)
    trader["last_dir"] = d
    dtxt = "bullish" if d > 0 else "bearish" if d < 0 else "neutro"

    if active_trade:
        if cfg.LEADER_CLOSE_ON_FLIP and d != 0:
            side = active_trade.get("side")
            opposes = (side == "Buy" and d < 0) or (side == "Sell" and d > 0)
            if opposes:
                return {"action": "CLOSE", "reason": f"lider inverteu para {dtxt} — fechando"}
        return {"action": "HOLD", "reason": "posicao aberta, lider mantido"}

    # Flat: entra apenas quando o lider MUDA para uma direcao (flip)
    if d != 0 and d != prev:
        try:
            df = market.get_klines(limit=SIGNAL_WARMUP_BARS, closed_only=True)
            ind = market.calc_indicators(df)
            price = float(df["close"].iloc[-1])
            atr = float(ind["atr"].iloc[-1])
        except Exception as e:
            return {"action": "HOLD", "reason": f"falha ao obter dados p/ entrada: {e}"}
        if not (atr > 0 and price > 0):
            return {"action": "HOLD", "reason": "ATR/preco invalido"}
        sl_frac = max(cfg.SL_MIN_PCT / 100, min(cfg.SL_MAX_PCT / 100, 1.5 * atr / price))
        sign = 1 if d > 0 else -1
        sl = price * (1 - sign * sl_frac)
        tp = price * (1 + sign * sl_frac * cfg.MIN_RR_RATIO)
        return {"action": "LONG" if d > 0 else "SHORT",
                "entry": price, "stop_loss": sl, "take_profit": tp,
                "confidence": 1.0, "reason": f"lider {dtxt} (mecanico, sem LLM)"}

    return {"action": "HOLD", "reason": "sem novo flip do lider"}


def _pullback_ok(df, ind, direction) -> bool:
    """Setup de pullback no alt (mesma regra do backtest): lider da a direcao,
    o alt da o timing. LONG: tendencia de alta (close>EMA50) + RSI cruzando 50
    p/ cima + ADX forte. SHORT: espelho."""
    if df is None or len(df) < 2:
        return False   # historico insuficiente (ex: simbolo recem-listado)
    try:
        import pandas as pd
        ema50 = ind["ema50"].iloc[-1]
        rsi, rsi_p = ind["rsi"].iloc[-1], ind["rsi"].iloc[-2]
        adx = ind["adx"].iloc[-1]
        close = df["close"].iloc[-1]
        if any(pd.isna(v) for v in (ema50, rsi, rsi_p, adx)):
            return False
        if adx < cfg.ADX_RANGING_THRESHOLD:
            return False
        if direction > 0:
            return close > ema50 and rsi_p < 50 <= rsi
        return close < ema50 and rsi_p > 50 >= rsi
    except Exception:
        return False


def build_mechanical_setup(sym, market, leader, sentiment) -> int:
    """Detecta um setup mecanico (barato, sem LLM). Retorna +1 LONG, -1 SHORT, 0 nenhum.
    Camadas: direcao do lider BTC/ETH -> regime macro -> Fear&Greed -> pullback no alt.
    So se TODAS passarem ha setup — e so entao vale gastar uma chamada da LLM."""
    d = leader.get_direction(sym)
    if d == 0:
        return 0

    # Regime macro — igual ao backtest: exige regime CONHECIDO e a favor da direcao
    # (reg==0 = MA indisponivel/falha -> bloqueia; conservador e alinhado a validacao)
    if cfg.HYBRID_REGIME_FILTER:
        reg = leader.get_regime()
        if reg == 0 or (d > 0 and reg < 0) or (d < 0 and reg > 0):
            return 0

    # Fear & Greed contrarian (da API gratis, ja cacheado)
    fng = sentiment.get_fear_greed()
    if fng and fng.get("value") is not None:
        v = fng["value"]
        if d > 0 and v >= cfg.FNG_GREED_MAX:
            return 0
        if d < 0 and v <= cfg.FNG_FEAR_MIN:
            return 0

    # Pullback no alt (timing) — velas FECHADAS (evita sinal que repinta)
    try:
        df = market.get_klines(limit=SIGNAL_WARMUP_BARS, closed_only=True)
        ind = market.calc_indicators(df)
    except Exception:
        return 0
    if not _pullback_ok(df, ind, d):
        return 0

    # Funding (swing): evita segurar posicao pagando funding alto contra ela
    if cfg.FUNDING_FILTER:
        fr = market.get_funding_rate()
        if fr is not None:
            fr_pct = fr * 100
            thr = cfg.FUNDING_MAX_ABS
            if (d > 0 and fr_pct >= thr) or (d < 0 and fr_pct <= -thr):
                logger.info(f"[{sym}] [FUNDING] {fr_pct:+.4f}%/8h contra "
                            f"{'LONG' if d > 0 else 'SHORT'} (limite {thr}%) — setup pulado")
                return 0
            logger.info(f"[{sym}] [FUNDING] {fr_pct:+.4f}%/8h — ok p/ {'LONG' if d > 0 else 'SHORT'}")
    return d


def llm_analyze(sym, market, analyst, leader, sentiment_text, stream_text, setup_dir=None):
    """Monta o prompt (dados + sentimento/F&G + lider), chama a LLM e aplica os
    filtros de direcao. Retorna a decisao. Usado no modo LLM puro e no hibrido.
    setup_dir (hibrido): +1/-1 da direcao do setup mecanico — a LLM so pode
    APROVAR essa direcao (ou vetar via HOLD); nunca abrir na direcao oposta."""
    market_text = market.format_for_llm()
    if sentiment_text:
        market_text += "\n\n" + sentiment_text
    if stream_text:
        market_text += "\n\n" + stream_text
    leader_text = leader.format_for_llm(sym)
    if leader_text:
        market_text += "\n\n" + leader_text

    logger.info(f"[{sym}] [LLM] Analisando com {analyst.provider_name} {analyst.model}...")
    decision = analyst.analyze(market_text, symbol=sym)

    # Reconciliacao com o setup (hibrido): a LLM so aprova a direcao do setup.
    # Qualquer outra resposta (direcao oposta, HOLD, CLOSE) vira HOLD (veto).
    if setup_dir and isinstance(decision, dict):
        expected = "LONG" if setup_dir > 0 else "SHORT"
        if decision.get("action") != expected:
            orig = decision.get("reason", "")
            logger.info(f"[{sym}] [HIBRIDO] LLM nao confirmou o setup {expected} "
                        f"(respondeu {decision.get('action')}) -> HOLD")
            decision["action"] = "HOLD"
            decision["reason"] = (orig + " | " if orig else "") + f"LLM nao confirmou setup {expected}"

    # Filtro duro do lider: veta LONG/SHORT contra a direcao do lider
    act = decision.get("action") if isinstance(decision, dict) else None
    if act in ("LONG", "SHORT"):
        bias = leader.get_bias(sym)
        vetoed = (act == "LONG" and not bias["allow_long"]) or \
                 (act == "SHORT" and not bias["allow_short"])
        if vetoed:
            veto = f"VETO {bias['reason']}: {act} contra o lider -> HOLD"
            logger.info(f"[{sym}] [LEADER] {veto}")
            orig = decision.get("reason", "")
            decision["action"] = "HOLD"
            decision["reason"] = (orig + " | " if orig else "") + veto
    return decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Modo live (ordens reais)")
    parser.add_argument("--once", action="store_true", help="Roda uma vez e sai")
    args = parser.parse_args()

    if args.live:
        cfg.DRY_RUN = False
        if not cfg.BYBIT_API_KEY or not cfg.BYBIT_API_SECRET:
            logger.error("[CONFIG] BYBIT_API_KEY e BYBIT_API_SECRET obrigatorias no modo --live")
            sys.exit(1)
    dry_run = cfg.DRY_RUN

    # Sincronizar timestamp com servidor Bybit (corrige clock drift)
    _sync_bybit_time()

    # Init components
    db.init_db()
    sentiment = SentimentData()
    stream_mgr = StreamAlertManager()
    leader = LeaderSignal()
    analyst = create_analyst()

    # Cria um trader (market + executor) por simbolo. Falha de um simbolo
    # nao impede os outros — apenas e logada.
    traders = []
    failed = []
    for sym in cfg.SYMBOLS:
        try:
            traders.append({
                "symbol": sym,
                "market": MarketData(sym),
                "executor": TradeExecutor(sym),
            })
        except Exception as e:
            failed.append(sym)
            logger.error(f"[INIT] Falha ao inicializar trader {sym}: {e}")

    if not traders:
        logger.critical("[INIT] Nenhum trader inicializado com sucesso. Encerrando.")
        sys.exit(1)

    logger.info(f"[INIT] {len(traders)} simbolos OK: {[t['symbol'] for t in traders]}"
                + (f" | {len(failed)} falharam: {failed}" if failed else ""))

    print_banner(dry_run, analyst)

    # Iniciar stream do X (se configurado)
    stream_mgr.start()

    # Handler para SIGTERM (shutdown gracioso)
    def _signal_handler(signum, frame):
        logger.info(f"[SIGNAL] Sinal {signum} recebido, encerrando...")
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _signal_handler)

    iteration = 0
    consecutive_errors = 0
    leader_sent_ts = 0.0   # ultimo refresh do sentimento do lider (BTC/ETH) via Grok
    max_consecutive_errors = 5

    if cfg.USE_LLM and cfg.LLM_AS_FILTER:
        logger.info(f"[START] AI Trader iniciado | MODO HIBRIDO — setup mecanico (lider+pullback+F&G+regime) "
                    f"filtrado pela LLM {analyst.provider_name} {analyst.model} | {len(traders)} simbolos "
                    f"| TF={cfg.TIMEFRAME}min | {'DRY RUN' if dry_run else 'LIVE'}")
    elif cfg.USE_LLM:
        logger.info(f"[START] AI Trader iniciado | {analyst.provider_name} {analyst.model} | {'DRY RUN' if dry_run else 'LIVE'}")
    else:
        logger.info(f"[START] AI Trader iniciado | MODO MECANICO (sem LLM) — direcao pelo lider BTC/ETH "
                    f"| TF={cfg.TIMEFRAME}min | {'DRY RUN' if dry_run else 'LIVE'}")

    while True:
        try:
            iteration += 1
            now = datetime.now(timezone.utc)

            # Limpa tela — informacoes atualizam in-place
            os.system("cls" if os.name == "nt" else "clear")
            print_banner(dry_run, analyst)
            logger.info(f"[ITER {iteration}] {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # Sentimento e stream so alimentam o prompt da LLM — pulados sem LLM.
            # Drena fila do stream PRIMEIRO e limpa imediatamente para evitar
            # descartar alertas que cheguem durante o processamento dos simbolos
            if cfg.USE_LLM:
                sentiment_text = sentiment.format_for_llm()
                stream_mgr.check_alerts()
                stream_text = stream_mgr.format_for_llm()
                if stream_text:
                    stream_mgr.clear_alerts()
            else:
                sentiment_text = ""
                stream_text = ""

            # Lider de mercado (BTC/ETH) — refresh 1x por iteracao (compartilhado).
            # Fail-open interno: nunca deixa uma falha do lider parar o ciclo.
            try:
                leader.refresh()
            except Exception as e:
                logger.warning(f"[LEADER] Erro no refresh, filtro fail-open: {e}")

            # Sentimento do LIDER (BTC/ETH) via Grok search — PERIODICO (a cada N horas).
            # Fica em cache e entra no prompt quando um setup dispara. So xAI c/ search.
            if (cfg.LEADER_SENTIMENT_HOURS > 0 and cfg.USE_LLM
                    and hasattr(analyst, "refresh_leader_sentiment")):
                if (now.timestamp() - leader_sent_ts) >= cfg.LEADER_SENTIMENT_HOURS * 3600:
                    try:
                        resumo = analyst.refresh_leader_sentiment()
                        logger.info(f"[SENTIMENTO BTC/ETH via Grok] {resumo}")
                        leader_sent_ts = now.timestamp()
                    except Exception as e:
                        logger.warning(f"[SENTIMENTO BTC/ETH] Falha na busca: {e}")

            # Conta erros por simbolo para detectar falha sistemica
            symbol_errors = 0

            # Itera por cada simbolo sequencialmente
            for trader in traders:
                sym = trader["symbol"]
                market = trader["market"]
                executor = trader["executor"]

                try:
                    logger.info(f"[{sym}] === Iniciando ciclo ===")

                    # 0. Verifica se posicao foi fechada por TP/SL da Bybit
                    executor.check_closed_by_exchange()

                    # 0b. Gestao de TP parcial: move SL p/ breakeven apos TP1
                    executor.manage_open_position()

                    # 0c. Decisao — 3 modos (gestao de posicao acima ja rodou):
                    #  HIBRIDO (LLM-filtro): mecanica seleciona o setup, LLM aprova/veta.
                    #  LLM puro: LLM analisa todo ciclo (com economia opcional).
                    #  MECANICO: so a direcao do lider, sem LLM.
                    if cfg.USE_LLM and cfg.LLM_AS_FILTER:
                        if executor.active_trade is not None:
                            # Posicao aberta: sem LLM. Saidas por TP/SL e CLOSE mecanico na virada.
                            decision = build_mechanical_decision(sym, market, executor.active_trade, leader, trader)
                        else:
                            setup = build_mechanical_setup(sym, market, leader, sentiment)
                            if setup == 0:
                                logger.info(f"[{sym}] [HIBRIDO] Sem setup mecanico — LLM nao chamada")
                                continue
                            logger.info(f"[{sym}] [HIBRIDO] Setup {'LONG' if setup > 0 else 'SHORT'} — consultando a LLM...")
                            decision = llm_analyze(sym, market, analyst, leader, sentiment_text, stream_text, setup_dir=setup)

                    elif cfg.USE_LLM:
                        # Economia: pula a analise quando ha posicao aberta ou dentro do intervalo
                        if cfg.LLM_SKIP_WHEN_IN_POSITION and executor.active_trade is not None:
                            logger.info(f"[{sym}] [LLM] Pulado — posicao aberta (economia; saida via TP/SL/breakeven)")
                            continue
                        if cfg.LLM_INTERVAL_MINUTES > 0:
                            elapsed_min = (now.timestamp() - trader.get("last_llm_ts", 0.0)) / 60
                            if elapsed_min < cfg.LLM_INTERVAL_MINUTES:
                                logger.info(f"[{sym}] [LLM] Pulado — intervalo de {cfg.LLM_INTERVAL_MINUTES}min "
                                            f"(faltam {cfg.LLM_INTERVAL_MINUTES - elapsed_min:.0f}min, economia)")
                                continue
                        logger.info(f"[{sym}] [DATA] Coletando dados de mercado...")
                        try:
                            decision = llm_analyze(sym, market, analyst, leader, sentiment_text, stream_text)
                            trader["last_llm_ts"] = now.timestamp()   # sucesso: intervalo cheio
                        except Exception:
                            # Falha: backoff curto (~2min) — nem martela todo ciclo, nem some pelo intervalo inteiro
                            trader["last_llm_ts"] = now.timestamp() - max(0, cfg.LLM_INTERVAL_MINUTES - 2) * 60
                            raise

                    else:
                        # Modo mecanico (sem LLM): decisao pela direcao do lider BTC/ETH
                        logger.info(f"[{sym}] [MECANICO] Decisao pela direcao do lider BTC/ETH...")
                        decision = build_mechanical_decision(sym, market, executor.active_trade, leader, trader)

                    # 3. Executa decisao
                    result = executor.execute(decision)
                    logger.info(f"[{sym}] [EXEC] {result}")

                    # 4. Registra no historico do simbolo (so alimenta contexto da LLM)
                    if cfg.USE_LLM:
                        analyst.record_decision(decision, now.strftime("%H:%M"), result, symbol=sym)
                except Exception as e:
                    symbol_errors += 1
                    logger.error(f"[{sym}] [ERROR] Erro no ciclo: {e}", exc_info=True)
                    # Erro em um simbolo nao para o bot — continua com os outros

            # Se TODOS os simbolos falharam, conta como erro consecutivo do bot
            if symbol_errors > 0 and symbol_errors == len(traders):
                raise RuntimeError(f"Todos os {len(traders)} simbolos falharam neste ciclo")

            # Status apos execucao (usa client do primeiro trader para queries de saldo)
            try:
                show_status(traders[0]["executor"].client)
            except Exception as e:
                logger.warning(f"[MONITOR] Erro ao exibir status: {e}")

            logger.info(f"[STATS] Iteracao {iteration} completa ({len(traders)} simbolos)")
            consecutive_errors = 0

        except KeyboardInterrupt:
            logger.info("[STOP] Bot parado pelo usuario")
            stream_mgr.stop()
            break
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"[ERROR] ({consecutive_errors}/{max_consecutive_errors}) {e}", exc_info=True)
            if args.once:
                logger.error("[ONCE] Modo --once falhou, saindo com erro")
                sys.exit(1)
            if consecutive_errors >= max_consecutive_errors:
                logger.critical(f"[FATAL] {max_consecutive_errors} erros consecutivos, encerrando bot")
                break

        if args.once:
            logger.info("[ONCE] Modo --once, saindo")
            break

        # Sleep — intervalo dinamico: 60s se algum trader tem posicao aberta, CHECK_INTERVAL caso contrario
        # Monitores: stream do X (a cada 1s) + polling xAI (a cada SENTIMENT_MONITOR_INTERVAL)
        any_open = any(t["executor"].active_trade for t in traders)
        interval = 60 if any_open else cfg.CHECK_INTERVAL
        monitor_interval = cfg.SENTIMENT_MONITOR_INTERVAL
        use_xai_monitor = (cfg.USE_LLM
                           and hasattr(analyst, 'check_sentiment_shift')
                           and analyst.use_search
                           and not any_open
                           and monitor_interval > 0)
        use_stream = cfg.X_STREAM_ENABLED

        parts = []
        if use_stream:
            parts.append("stream X")
        if use_xai_monitor:
            parts.append(f"polling xAI {monitor_interval}s")
        if parts:
            logger.info(f"[SLEEP] Aguardando {interval}s (monitorando: {', '.join(parts)})...")
        else:
            logger.info(f"[SLEEP] Aguardando {interval}s{'  (posicao aberta)' if any_open else ''}...")

        try:
            elapsed = 0
            while elapsed < interval:
                # Stream do X: checa fila a cada segundo (< 1s de delay)
                if use_stream and stream_mgr.check_alerts():
                    logger.info("[ALERT] Tweet urgente detectado — forcando analise!")
                    break

                # Polling xAI: checa sentimento periodicamente
                if use_xai_monitor and elapsed > 0 and elapsed % monitor_interval == 0:
                    if analyst.check_sentiment_shift():
                        logger.info("[ALERT] Mudanca brusca de sentimento — forcando analise!")
                        break

                time.sleep(1)
                elapsed += 1
        except KeyboardInterrupt:
            logger.info("[STOP] Bot parado pelo usuario")
            stream_mgr.stop()
            break


if __name__ == "__main__":
    main()
