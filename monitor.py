"""
Monitor de trades — exibe posicao atual, historico e estatisticas.
Uso: python monitor.py
"""
import time
import logging
from datetime import datetime, timezone
from pybit.unified_trading import HTTP
from pybit.exceptions import FailedRequestError

import config as cfg
from executor import ORDER_PREFIX
import db

logger = logging.getLogger(__name__)


def _api_call(client: HTTP, method_name: str, **kwargs):
    """Chamada API com retry e backoff exponencial."""
    for attempt in range(3):
        try:
            method = getattr(client, method_name)
            return method(**kwargs)
        except (ConnectionError, ConnectionResetError, OSError, FailedRequestError) as e:
            logger.warning(f"[MONITOR RETRY] Tentativa {attempt+1}/3: {e}")
            if attempt < 2:
                time.sleep(2 * (2 ** attempt))
            else:
                raise


def show_balance(client: HTTP):
    result = _api_call(client, "get_wallet_balance", accountType="UNIFIED")
    coins = result.get("result", {}).get("list", [])
    if not coins:
        print("  Saldo: indisponivel (resposta API vazia)")
        return 0.0
    for coin in coins[0].get("coin", []):
        if coin["coin"] == "USDT":
            balance = float(coin["walletBalance"])
            print(f"  Saldo USDT: ${balance:,.2f}")
            return balance
    return 0.0


def show_position(client: HTTP):
    result = _api_call(client, "get_positions", category="linear", symbol=cfg.SYMBOL)
    positions = result.get("result", {}).get("list", [])
    if not positions:
        print("  Posicao: indisponivel (resposta API vazia)")
        return
    for p in positions:
        if float(p["size"]) > 0:
            entry = float(p["avgPrice"])
            mark = float(p["markPrice"])
            pnl = float(p["unrealisedPnl"])
            sl_str = p.get("stopLoss") or ""
            tp_str = p.get("takeProfit") or ""
            sl = float(sl_str) if sl_str else 0.0
            tp = float(tp_str) if tp_str else 0.0
            side = "SHORT" if p["side"] == "Sell" else "LONG"
            print(f"  {side} {p['size']} BTC @ ${entry:,.2f}")
            print(f"  Preco atual: ${mark:,.2f}")
            if sl or tp:
                print(f"  TP: ${tp:,.2f} | SL: ${sl:,.2f}")
            pnl_icon = "+" if pnl >= 0 else ""
            print(f"  PnL: {pnl_icon}${pnl:,.2f}")
            return
    print("  Nenhuma posicao aberta")


def show_history():
    trades = db.get_all_trades()
    if not trades:
        print("  Sem trades registrados")
        return

    header = f"  {'#':<3} {'Data':<16} {'Side':<6} {'Qty':<8} {'Entry':>11} {'Exit':>11} {'PnL':>9} {'Tipo':<6} {'LLM'}"
    print(header)
    print("  " + "-" * 90)

    for i, t in enumerate(trades):
        opened = t["opened_at"]
        try:
            # Compativel com Python < 3.11 (remove +00:00 se presente)
            if opened.endswith("+00:00"):
                opened = opened.replace("+00:00", "")
            dt = datetime.fromisoformat(opened)
        except (ValueError, TypeError, AttributeError):
            dt = datetime(2000, 1, 1)  # fallback para datas malformadas
        pnl = t["pnl"] if t["pnl"] is not None else 0.0
        exit_p = t["exit_price"] if t["exit_price"] is not None else 0.0
        close_type = t.get("close_type") or "?"
        llm = t.get("llm_provider") or ""

        print(
            f"  {i+1:<3} {dt.strftime('%d/%m %H:%M'):<16} {t['side']:<6} {t['qty']:<8}"
            f" {t['entry_price']:>11,.2f} {exit_p:>11,.2f}"
            f" {pnl:>+9.2f} {close_type:<6} {llm}"
        )

    print("  " + "-" * 90)


def show_stats():
    stats = db.get_stats()
    if not stats:
        print("  Sem dados para estatisticas")
        return

    print(f"  Trades: {stats['total_trades']} | Wins: {stats['win_count']} | Losses: {stats['loss_count']}")
    print(f"  Win rate: {stats['win_rate']:.1f}%")
    print(f"  PnL total: ${stats['total_pnl']:+,.2f}")
    print(f"  Media win: ${stats['avg_win']:+,.2f} | Media loss: ${stats['avg_loss']:+,.2f}")
    pf = stats['profit_factor']
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"  Profit factor: {pf_str}")
    print(f"  Max drawdown: ${stats['max_drawdown']:,.2f}")
    print(f"  Fechamentos: TP={stats['tp_count']} | SL={stats['sl_count']} | CLOSE={stats['close_count']}")


def show_status(client: HTTP):
    """Exibe status completo: conta, posicao e historico."""
    print()
    print("=" * 50)
    print(f"  MONITOR — {cfg.SYMBOL}")
    print(f"  {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * 50)

    print()
    print(">> CONTA")
    show_balance(client)

    print()
    print(">> POSICAO ABERTA")
    show_position(client)

    print()
    print(">> HISTORICO DE TRADES")
    show_history()

    print()
    print(">> ESTATISTICAS")
    show_stats()
    print()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
    db.init_db()
    client = HTTP(
        testnet=cfg.USE_TESTNET,
        api_key=cfg.BYBIT_API_KEY,
        api_secret=cfg.BYBIT_API_SECRET,
        recv_window=20000,
        timeout=30,
    )
    show_status(client)
