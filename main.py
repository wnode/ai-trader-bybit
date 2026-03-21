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
from market_data import MarketData
from analyst import create_analyst
from executor import TradeExecutor
from monitor import show_status
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


def print_banner(dry_run: bool, analyst):
    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"""
    +==============================================================+
    |    >>> AI TRADER BOT — LLM-Powered Trading <<<               |
    +--------------------------------------------------------------+
    |  Provider: {analyst.provider_name:<47s}|
    |  Model: {analyst.model:<50s}|
    |  Symbol: {cfg.SYMBOL:<49s}|
    |  Mode: {mode:<51s}|
    +--------------------------------------------------------------+
    |  1. Coleta dados de mercado (klines + indicadores)           |
    |  2. Envia a LLM para analise                                 |
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

    # Init components
    db.init_db()
    market = MarketData()
    analyst = create_analyst()
    executor = TradeExecutor()

    print_banner(dry_run, analyst)

    # Status inicial: conta, posicao e historico
    try:
        show_status(executor.client)
    except Exception as e:
        logger.warning(f"[MONITOR] Erro ao exibir status: {e}")

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
            logger.info(f"{'='*60}")
            logger.info(f"[ITER {iteration}] {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # 0. Verifica se posicao foi fechada por TP/SL da Bybit
            executor.check_closed_by_exchange()

            # 1. Coleta dados
            logger.info("[DATA] Coletando dados de mercado...")
            market_text = market.format_for_llm()

            # 2. Envia a LLM
            logger.info(f"[LLM] Analisando com {analyst.provider_name} {analyst.model}...")
            decision = analyst.analyze(market_text)

            # 3. Executa decisao
            result = executor.execute(decision)
            logger.info(f"[EXEC] {result}")

            # 4. Registra
            analyst.record_decision(decision, now.strftime("%H:%M"), result)

            # Status apos execucao
            try:
                show_status(executor.client)
            except Exception as e:
                logger.warning(f"[MONITOR] Erro ao exibir status: {e}")

            logger.info(f"[STATS] Iteracao {iteration} completa")
            consecutive_errors = 0

        except KeyboardInterrupt:
            logger.info("[STOP] Bot parado pelo usuario")
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

        # Sleep
        logger.info(f"[SLEEP] Aguardando {cfg.CHECK_INTERVAL}s...")
        try:
            time.sleep(cfg.CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("[STOP] Bot parado pelo usuario")
            break


if __name__ == "__main__":
    main()
