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
    print(f"""
    +==============================================================+
    |    >>> AI TRADER BOT — LLM-Powered Trading <<<               |
    +--------------------------------------------------------------+
    |  Provider: {analyst.provider_name:<47s}|
    |  Model: {analyst.model:<50s}|
    |  Symbols: {symbols_str:<48s}|
    |  Mode: {mode:<51s}|
    +--------------------------------------------------------------+
    |  1. Coleta dados de cada simbolo (klines + indicadores)      |
    |  2. Envia a LLM para analise (um por simbolo)                |
    |  3. LLM decide: LONG / SHORT / HOLD / CLOSE                 |
    |  4. Executor abre/fecha posicao na Bybit                     |
    |  5. Repete a cada {cfg.CHECK_INTERVAL}s                                       |
    +==============================================================+
    """)


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
    analyst = create_analyst()

    # Cria um trader (market + executor) por simbolo
    traders = []
    for sym in cfg.SYMBOLS:
        traders.append({
            "symbol": sym,
            "market": MarketData(sym),
            "executor": TradeExecutor(sym),
        })
    logger.info(f"[INIT] {len(traders)} simbolos configurados: {[t['symbol'] for t in traders]}")

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
    max_consecutive_errors = 5

    logger.info(f"[START] AI Trader iniciado | {analyst.provider_name} {analyst.model} | {'DRY RUN' if dry_run else 'LIVE'}")

    while True:
        try:
            iteration += 1
            now = datetime.now(timezone.utc)

            # Limpa tela — informacoes atualizam in-place
            os.system("cls" if os.name == "nt" else "clear")
            print_banner(dry_run, analyst)
            logger.info(f"[ITER {iteration}] {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # Sentimento e stream sao compartilhados entre todos os simbolos
            sentiment_text = sentiment.format_for_llm()
            stream_mgr.check_alerts()
            stream_text = stream_mgr.format_for_llm()

            # Itera por cada simbolo sequencialmente
            for trader in traders:
                sym = trader["symbol"]
                market = trader["market"]
                executor = trader["executor"]

                try:
                    logger.info(f"[{sym}] === Iniciando ciclo ===")

                    # 0. Verifica se posicao foi fechada por TP/SL da Bybit
                    executor.check_closed_by_exchange()

                    # 1. Coleta dados do simbolo
                    logger.info(f"[{sym}] [DATA] Coletando dados de mercado...")
                    market_text = market.format_for_llm()

                    # 1b. Adiciona sentimento (compartilhado)
                    if sentiment_text:
                        market_text += "\n\n" + sentiment_text
                    if stream_text:
                        market_text += "\n\n" + stream_text

                    # 2. Envia a LLM com historico do simbolo
                    logger.info(f"[{sym}] [LLM] Analisando com {analyst.provider_name} {analyst.model}...")
                    decision = analyst.analyze(market_text, symbol=sym)

                    # 3. Executa decisao
                    result = executor.execute(decision)
                    logger.info(f"[{sym}] [EXEC] {result}")

                    # 4. Registra no historico do simbolo
                    analyst.record_decision(decision, now.strftime("%H:%M"), result, symbol=sym)
                except Exception as e:
                    logger.error(f"[{sym}] [ERROR] Erro no ciclo: {e}", exc_info=True)
                    # Erro em um simbolo nao para o bot — continua com os outros

            # Apos processar todos os simbolos, limpa alertas do stream
            if stream_text:
                stream_mgr.clear_alerts()

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
        use_xai_monitor = (hasattr(analyst, 'check_sentiment_shift')
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
