"""
Monitor de trades — exibe posicao atual, historico e estatisticas.
Uso: python monitor.py
"""
import time
import logging
from datetime import datetime
from pybit.unified_trading import HTTP
from config import SYMBOL
from executor import ORDER_PREFIX

logger = logging.getLogger(__name__)


def _api_call(client, method, **kwargs):
    """Chamada API com retry."""
    for attempt in range(3):
        try:
            return method(**kwargs)
        except (ConnectionError, ConnectionResetError, OSError) as e:
            logger.warning(f"[MONITOR RETRY] Tentativa {attempt+1}/3: {e}")
            if attempt < 2:
                time.sleep(2)
            else:
                raise


def show_balance(client: HTTP):
    result = _api_call(client, client.get_wallet_balance, accountType="UNIFIED")
    for coin in result["result"]["list"][0]["coin"]:
        if coin["coin"] == "USDT":
            balance = float(coin["walletBalance"])
            print(f"  Saldo USDT: ${balance:,.2f}")
            return balance
    return 0.0


def show_position(client: HTTP):
    result = _api_call(client, client.get_positions, category="linear", symbol=SYMBOL)
    for p in result["result"]["list"]:
        if float(p["size"]) > 0:
            entry = float(p["avgPrice"])
            mark = float(p["markPrice"])
            pnl = float(p["unrealisedPnl"])
            sl = p.get("stopLoss", "0")
            tp = p.get("takeProfit", "0")
            side = "SHORT" if p["side"] == "Sell" else "LONG"
            print(f"  {side} {p['size']} BTC @ ${entry:,.2f}")
            print(f"  Preco atual: ${mark:,.2f}")
            print(f"  TP: ${float(tp):,.2f} | SL: ${float(sl):,.2f}")
            pnl_icon = "+" if pnl >= 0 else ""
            print(f"  PnL: {pnl_icon}${pnl:,.2f}")
            return
    print("  Nenhuma posicao aberta")


def show_history(client: HTTP):
    closed = _api_call(client, client.get_closed_pnl, category="linear", symbol=SYMBOL, limit=50)
    orders = _api_call(client, client.get_order_history, category="linear", symbol=SYMBOL, limit=50)

    # Mapear ordens do bot e TP/SL filled
    bot_order_ids = set()
    tp_sl_filled = {}
    for o in orders["result"]["list"]:
        link = o.get("orderLinkId", "")
        if link.startswith(ORDER_PREFIX):
            bot_order_ids.add(o["orderId"])
        if o["orderStatus"] == "Filled" and o["stopOrderType"] in ("TakeProfit", "StopLoss"):
            tp_sl_filled[o["createdTime"]] = o["stopOrderType"]

    # Filtrar apenas trades do bot
    trades = [t for t in closed["result"]["list"] if t.get("orderId") in bot_order_ids]
    if not trades:
        print("  Sem trades do bot")
        return

    tp_count = 0
    sl_count = 0
    total_pnl = 0.0

    header = f"  {'#':<3} {'Data':<16} {'Side':<6} {'Qty':<8} {'Entry':>11} {'Exit':>11} {'PnL':>9} {'Tipo'}"
    print(header)
    print("  " + "-" * 78)

    for i, t in enumerate(reversed(trades)):
        dt = datetime.fromtimestamp(int(t["updatedTime"]) / 1000)
        pnl = float(t["closedPnl"])
        total_pnl += pnl

        # side no closed PnL e o lado de fechamento, inverter para abertura
        side = "SHORT" if t["side"] == "Buy" else "LONG"

        # Determinar TP ou SL
        created = t["createdTime"]
        if created in tp_sl_filled:
            tipo = "TP" if tp_sl_filled[created] == "TakeProfit" else "SL"
        elif pnl > 0:
            tipo = "TP*"
        else:
            tipo = "SL*"

        if "TP" in tipo:
            tp_count += 1
        else:
            sl_count += 1

        print(
            f"  {i+1:<3} {dt.strftime('%d/%m %H:%M'):<16} {side:<6} {t['qty']:<8}"
            f" {float(t['avgEntryPrice']):>11,.2f} {float(t['avgExitPrice']):>11,.2f}"
            f" {pnl:>+9.2f} {tipo}"
        )

    print("  " + "-" * 78)
    total = tp_count + sl_count
    wr = (tp_count / total * 100) if total > 0 else 0
    print(f"  Trades: {total} | TP: {tp_count} | SL: {sl_count} | Win rate: {wr:.1f}%")
    print(f"  PnL total: ${total_pnl:+,.2f}")
    print()
    print("  * = inferido pelo PnL (sem match exato na ordem)")


def show_status(client: HTTP):
    """Exibe status completo: conta, posicao e historico."""
    print()
    print("=" * 50)
    print(f"  MONITOR — {SYMBOL}")
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
    show_history(client)
    print()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    from config import BYBIT_API_KEY, BYBIT_API_SECRET, USE_TESTNET
    client = HTTP(
        testnet=USE_TESTNET,
        api_key=BYBIT_API_KEY,
        api_secret=BYBIT_API_SECRET,
        recv_window=10000,
        timeout=30,
    )
    show_status(client)
