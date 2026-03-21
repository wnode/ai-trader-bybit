"""
Executor de trades — recebe decisao da LLM e executa na Bybit.
Inclui retry com reconexao para ConnectionResetError.
"""
import logging
import time
from pybit.unified_trading import HTTP

from config import (
    BYBIT_API_KEY, BYBIT_API_SECRET, USE_TESTNET,
    SYMBOL, LEVERAGE, RISK_PER_TRADE, DRY_RUN
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2


class TradeExecutor:
    """Executa trades na Bybit baseado nas decisoes da LLM."""

    def __init__(self):
        self.client = self._create_client()
        self.active_trade = None

    def _create_client(self) -> HTTP:
        return HTTP(
            testnet=USE_TESTNET,
            api_key=BYBIT_API_KEY,
            api_secret=BYBIT_API_SECRET,
            recv_window=10000,
            timeout=30
        )

    def _api_call(self, method, **kwargs):
        """Executa chamada API com retry e reconexao."""
        for attempt in range(MAX_RETRIES):
            try:
                return method(**kwargs)
            except (ConnectionError, ConnectionResetError, OSError) as e:
                logger.warning(f"[RETRY] Tentativa {attempt+1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    self.client = self._create_client()
                else:
                    raise

    def get_balance(self) -> float:
        result = self._api_call(
            self.client.get_wallet_balance,
            accountType="UNIFIED"
        )
        for coin in result["result"]["list"][0]["coin"]:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0

    def calc_position_size(self, entry: float, sl: float) -> float:
        """Calcula tamanho da posicao baseado no risco."""
        balance = self.get_balance()
        risk_amount = balance * RISK_PER_TRADE
        sl_dist = abs(entry - sl) / entry
        if sl_dist == 0:
            return 0.0
        notional = (risk_amount / sl_dist) * LEVERAGE
        qty = notional / entry
        # Bybit min: 0.001 BTC
        qty = max(0.001, round(qty, 3))
        return qty

    def execute(self, decision: dict) -> str:
        """Executa a decisao da LLM. Retorna status string."""
        action = decision["action"]

        if action == "HOLD":
            return "HOLD — nenhuma acao"

        if action == "CLOSE":
            return self._close_position()

        if action in ("LONG", "SHORT"):
            return self._open_position(decision)

        return f"Acao desconhecida: {action}"

    def _open_position(self, decision: dict) -> str:
        """Abre posicao LONG ou SHORT."""
        # Checa se ja tem posicao
        pos = self._get_position()
        if pos:
            return f"Ja existe posicao {pos['side']} aberta — ignorando"

        action = decision["action"]
        entry = decision.get("entry")
        sl = decision.get("stop_loss")
        tp = decision.get("take_profit")

        if not all([entry, sl, tp]):
            return "Entry/SL/TP nao definidos — ignorando"

        qty = self.calc_position_size(entry, sl)
        side = "Buy" if action == "LONG" else "Sell"

        if DRY_RUN:
            self.active_trade = {
                "side": side, "entry": entry, "sl": sl, "tp": tp,
                "qty": qty, "reason": decision.get("reason", ""),
            }
            return (f"[DRY] {action} {qty} BTC @ ${entry:,.2f} "
                    f"SL=${sl:,.2f} TP=${tp:,.2f} "
                    f"(conf={decision.get('confidence', 0):.1f})")

        try:
            result = self._api_call(
                self.client.place_order,
                category="linear",
                symbol=SYMBOL,
                side=side,
                orderType="Market",
                qty=str(qty),
                stopLoss=str(round(sl, 2)),
                takeProfit=str(round(tp, 2)),
                positionIdx=0,
            )
            if result["retCode"] == 0:
                self.active_trade = {
                    "side": side, "entry": entry, "sl": sl, "tp": tp,
                    "qty": qty, "order_id": result["result"]["orderId"],
                }
                return (f"{action} {qty} BTC @ market "
                        f"SL=${sl:,.2f} TP=${tp:,.2f} "
                        f"OrderID={result['result']['orderId']}")
            else:
                return f"Erro Bybit: {result['retMsg']}"
        except Exception as e:
            return f"Erro ao executar: {e}"

    def _close_position(self) -> str:
        """Fecha posicao aberta."""
        pos = self._get_position()
        if not pos:
            self.active_trade = None
            return "Nenhuma posicao para fechar"

        if DRY_RUN:
            self.active_trade = None
            return f"[DRY] Fechando {pos['side']} {pos['size']} BTC"

        try:
            close_side = "Sell" if pos["side"] == "Buy" else "Buy"
            result = self._api_call(
                self.client.place_order,
                category="linear",
                symbol=SYMBOL,
                side=close_side,
                orderType="Market",
                qty=str(pos["size"]),
                reduceOnly=True,
                positionIdx=0,
            )
            if result["retCode"] == 0:
                self.active_trade = None
                return f"Posicao fechada — OrderID={result['result']['orderId']}"
            else:
                return f"Erro ao fechar: {result['retMsg']}"
        except Exception as e:
            return f"Erro ao fechar: {e}"

    def _get_position(self) -> dict | None:
        """Retorna posicao aberta."""
        result = self._api_call(
            self.client.get_positions,
            category="linear", symbol=SYMBOL
        )
        for p in result["result"]["list"]:
            if float(p["size"]) > 0:
                return {
                    "side": p["side"],
                    "size": float(p["size"]),
                    "entry": float(p["avgPrice"]),
                }
        return None
