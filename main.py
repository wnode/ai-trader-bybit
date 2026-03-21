"""
AI Trader Bot — Main Loop
==========================
Bot de trading que usa LLM (Claude/Gemini/GPT) para analisar o mercado e decidir trades.

Uso:
    python main.py              # dry run (padrao)
    python main.py --live       # ordens reais
    python main.py --once       # roda uma vez e sai
"""
import sys
import io
import time
import logging
import argparse
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from config import CHECK_INTERVAL, LLM_PROVIDER, SYMBOL
from market_data import MarketData
from analyst import create_analyst
from executor import TradeExecutor

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
    |  Symbol: {SYMBOL:<49s}|
    |  Mode: {mode:<51s}|
    +--------------------------------------------------------------+
    |  1. Coleta dados de mercado (klines + indicadores)           |
    |  2. Envia a LLM para analise                                 |
    |  3. LLM decide: LONG / SHORT / HOLD / CLOSE                 |
    |  4. Executor abre/fecha posicao na Bybit                     |
    |  5. Repete a cada {CHECK_INTERVAL}s                                       |
    +==============================================================+
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Modo live (ordens reais)")
    parser.add_argument("--once", action="store_true", help="Roda uma vez e sai")
    args = parser.parse_args()

    import config as cfg
    if args.live:
        cfg.DRY_RUN = False
    dry_run = cfg.DRY_RUN

    # Init components
    market = MarketData()
    analyst = create_analyst()
    executor = TradeExecutor()

    print_banner(dry_run, analyst)

    iteration = 0

    logger.info(f"[START] AI Trader iniciado | {analyst.provider_name} {analyst.model} | {'DRY RUN' if dry_run else 'LIVE'}")

    while True:
        try:
            iteration += 1
            now = datetime.now(timezone.utc)
            logger.info(f"{'='*60}")
            logger.info(f"[ITER {iteration}] {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")

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

            logger.info(f"[STATS] Iteracao {iteration} completa")

        except KeyboardInterrupt:
            logger.info("[STOP] Bot parado pelo usuario")
            break
        except Exception as e:
            logger.error(f"[ERROR] {e}", exc_info=True)

        if args.once:
            logger.info("[ONCE] Modo --once, saindo")
            break

        # Sleep
        logger.info(f"[SLEEP] Aguardando {CHECK_INTERVAL}s...")
        try:
            time.sleep(CHECK_INTERVAL)
        except KeyboardInterrupt:
            logger.info("[STOP] Bot parado pelo usuario")
            break


if __name__ == "__main__":
    main()
