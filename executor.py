"""
Executor de trades — recebe decisao da LLM e executa na Bybit.
Inclui retry com reconexao para ConnectionResetError.
"""
import logging
import time
import uuid
from pybit.unified_trading import HTTP
from pybit.exceptions import FailedRequestError

import config as cfg
import db

logger = logging.getLogger(__name__)

ORDER_PREFIX = "aitbot"
MAX_RETRIES = 3
RETRY_DELAY = 2


class TradeExecutor:
    """Executa trades na Bybit baseado nas decisoes da LLM."""

    def __init__(self):
        self.client = self._create_client()
        self.active_trade = None
        self._restore_active_trade()

    def _create_client(self) -> HTTP:
        return HTTP(
            testnet=cfg.USE_TESTNET,
            api_key=cfg.BYBIT_API_KEY,
            api_secret=cfg.BYBIT_API_SECRET,
            recv_window=20000,
            timeout=30
        )

    def _api_call(self, method_name: str, **kwargs):
        """Executa chamada API com retry, backoff exponencial e reconexao."""
        for attempt in range(MAX_RETRIES):
            try:
                method = getattr(self.client, method_name)
                return method(**kwargs)
            except (ConnectionError, ConnectionResetError, OSError, FailedRequestError) as e:
                logger.warning(f"[RETRY] Tentativa {attempt+1}/{MAX_RETRIES}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))
                    self.client = self._create_client()
                else:
                    raise

    def _restore_active_trade(self):
        """Restaura active_trade do DB se houver trade aberto e posicao na exchange."""
        try:
            open_trade = db.get_open_trade()
            if not open_trade:
                return
            pos = self._get_position()
            if pos:
                self.active_trade = {
                    "side": pos["side"],
                    "entry": pos["entry"],
                    "sl": open_trade.get("stop_loss"),
                    "tp": open_trade.get("take_profit"),
                    "qty": pos["size"],
                    "order_id": open_trade["order_id"],
                }
                logger.info(f"[RESTORE] Trade restaurado do DB: {open_trade['side']} {pos['size']} BTC")
            else:
                # Posicao nao existe mais, fechar trade orfao no DB
                self.check_closed_by_exchange_for_order(open_trade["order_id"])
        except Exception as e:
            logger.warning(f"[RESTORE] Erro ao restaurar trade: {e}")

    def get_balance(self) -> float:
        result = self._api_call(
            "get_wallet_balance",
            accountType="UNIFIED"
        )
        for coin in result["result"]["list"][0]["coin"]:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0

    def calc_position_size(self, entry: float, sl: float) -> float:
        """Calcula tamanho da posicao baseado no risco."""
        balance = self.get_balance()
        risk_amount = balance * cfg.RISK_PER_TRADE
        sl_dist = abs(entry - sl) / entry
        if sl_dist == 0:
            return 0.0
        notional = risk_amount / sl_dist
        qty = notional / entry
        qty = round(qty, 3)
        # Bybit min: 0.001 BTC — se qty minima excede o risco, pular trade
        if qty < 0.001:
            min_risk = 0.001 * abs(entry - sl)
            logger.warning(f"[RISK] Qty calculada {qty} < minimo 0.001. "
                           f"Risco minimo seria ${min_risk:,.2f} vs budget ${risk_amount:,.2f}")
            return 0.0
        return qty

    def execute(self, decision: dict) -> str:
        """Executa a decisao da LLM. Retorna status string."""
        if not isinstance(decision, dict) or "action" not in decision:
            return f"Decisao invalida: {decision}"
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

        if entry is None or sl is None or tp is None:
            return "Entry/SL/TP nao definidos — ignorando"

        qty = self.calc_position_size(entry, sl)
        if qty <= 0:
            return "Qty zero — risco muito baixo para abrir posicao"

        side = "Buy" if action == "LONG" else "Sell"

        if cfg.DRY_RUN:
            self.active_trade = {
                "side": side, "entry": entry, "sl": sl, "tp": tp,
                "qty": qty, "reason": decision.get("reason", ""),
            }
            return (f"[DRY] {action} {qty} BTC @ ${entry:,.2f} "
                    f"SL=${sl:,.2f} TP=${tp:,.2f} "
                    f"(conf={decision.get('confidence', 0):.1f})")

        try:
            link_id = f"{ORDER_PREFIX}-{uuid.uuid4().hex[:16]}"
            result = self._api_call(
                "place_order",
                category="linear",
                symbol=cfg.SYMBOL,
                side=side,
                orderType="Market",
                qty=str(qty),
                stopLoss=str(round(sl, 2)),
                takeProfit=str(round(tp, 2)),
                positionIdx=0,
                orderLinkId=link_id,
            )
            if result["retCode"] == 0:
                order_id = result["result"]["orderId"]

                # Buscar preco real de fill
                real_entry = self._get_fill_price(order_id, fallback=entry)

                try:
                    db.record_open(
                        side=action, qty=qty, entry_price=real_entry,
                        stop_loss=sl, take_profit=tp,
                        confidence=decision.get("confidence", 0),
                        reason=decision.get("reason", ""),
                        order_id=order_id,
                        llm_provider=cfg.LLM_PROVIDER,
                        llm_model=self._get_llm_model(),
                    )
                except Exception as e:
                    logger.error(f"[DB] Falha ao registrar abertura no DB (posicao JA aberta na exchange) order_id={order_id}: {e}")
                self.active_trade = {
                    "side": side, "entry": real_entry, "sl": sl, "tp": tp,
                    "qty": qty, "order_id": order_id,
                }
                return (f"{action} {qty} BTC @ ${real_entry:,.2f} "
                        f"SL=${sl:,.2f} TP=${tp:,.2f} "
                        f"OrderID={order_id}")
            else:
                return f"Erro Bybit: {result['retMsg']}"
        except Exception as e:
            return f"Erro ao executar: {e}"

    def _close_position(self, close_type: str = "CLOSE") -> str:
        """Fecha posicao aberta."""
        pos = self._get_position()
        if not pos:
            self.active_trade = None
            return "Nenhuma posicao para fechar"

        if cfg.DRY_RUN:
            self.active_trade = None
            return f"[DRY] Fechando {pos['side']} {pos['size']} BTC"

        try:
            close_side = "Sell" if pos["side"] == "Buy" else "Buy"
            link_id = f"{ORDER_PREFIX}-{uuid.uuid4().hex[:16]}"
            result = self._api_call(
                "place_order",
                category="linear",
                symbol=cfg.SYMBOL,
                side=close_side,
                orderType="Market",
                qty=str(pos["size"]),
                reduceOnly=True,
                positionIdx=0,
                orderLinkId=link_id,
            )
            if result["retCode"] == 0:
                close_order_id = result["result"]["orderId"]
                # Buscar preco real de fechamento
                exit_price = self._get_fill_price(close_order_id, fallback=pos["mark"])
                pnl = self._calc_close_pnl(pos, exit_price)

                db_ok = False
                if self.active_trade and self.active_trade.get("order_id"):
                    try:
                        updated = db.record_close(
                            order_id=self.active_trade["order_id"],
                            exit_price=exit_price, pnl=pnl, close_type=close_type,
                        )
                        db_ok = updated
                        if not updated:
                            logger.warning(f"[DB] record_close retornou False — trade ja fechado ou order_id nao encontrado")
                    except Exception as e:
                        logger.error(f"[DB] Falha ao registrar fechamento order_id={self.active_trade.get('order_id')}: {e}")
                if db_ok or not self.active_trade:
                    self.active_trade = None
                else:
                    # Guardar dados de fechamento para retry direto no proximo ciclo
                    self.active_trade["pending_close"] = {
                        "exit_price": exit_price, "pnl": pnl, "close_type": close_type,
                    }
                    logger.warning("[STATE] active_trade mantido com pending_close — DB nao confirmou fechamento")
                return f"Posicao fechada @ ${exit_price:,.2f} PnL=${pnl:+,.2f} — OrderID={close_order_id}"
            else:
                # Posicao pode ter sido fechada entre get_position e place_order
                recheck = self._get_position()
                if not recheck:
                    logger.warning(f"[CLOSE] Ordem falhou ({result['retMsg']}) mas posicao ja fechada — reconciliando")
                    self.check_closed_by_exchange()
                return f"Erro ao fechar: {result['retMsg']}"
        except Exception as e:
            return f"Erro ao fechar: {e}"

    def _get_fill_price(self, order_id: str, fallback: float) -> float:
        """Busca preco real de fill de uma ordem."""
        try:
            time.sleep(0.5)  # Aguarda fill propagar
            result = self._api_call(
                "get_executions",
                category="linear", symbol=cfg.SYMBOL, orderId=order_id, limit=1,
            )
            executions = result.get("result", {}).get("list", [])
            if executions:
                return float(executions[0]["execPrice"])
        except Exception as e:
            logger.warning(f"[FILL] Erro ao buscar fill price: {e}")
        return fallback

    def _calc_close_pnl(self, pos: dict, exit_price: float) -> float:
        """Calcula PnL estimado do fechamento."""
        entry = pos["entry"]
        size = pos["size"]
        if pos["side"] == "Buy":  # LONG
            return (exit_price - entry) * size
        else:  # SHORT
            return (entry - exit_price) * size

    def _get_position(self) -> dict | None:
        """Retorna posicao aberta."""
        result = self._api_call(
            "get_positions",
            category="linear", symbol=cfg.SYMBOL
        )
        for p in result["result"]["list"]:
            if float(p["size"]) > 0:
                return {
                    "side": p["side"],
                    "size": float(p["size"]),
                    "entry": float(p["avgPrice"]),
                    "mark": float(p.get("markPrice", p["avgPrice"])),
                }
        return None

    def check_closed_by_exchange(self):
        """Verifica se a posicao foi fechada por TP/SL da Bybit e registra no DB."""
        if not self.active_trade or not self.active_trade.get("order_id"):
            return
        pos = self._get_position()
        if pos:
            return  # ainda aberta

        order_id = self.active_trade["order_id"]

        # Se temos dados de fechamento pendentes (DB falhou no ciclo anterior), tentar direto
        pending = self.active_trade.get("pending_close")
        if pending:
            try:
                updated = db.record_close(
                    order_id=order_id,
                    exit_price=pending["exit_price"],
                    pnl=pending["pnl"],
                    close_type=pending["close_type"],
                )
                if updated:
                    logger.info(f"[DB] Retry pending_close OK: PnL ${pending['pnl']:+,.2f}")
                else:
                    logger.warning(f"[DB] Retry pending_close: record_close retornou False — trade ja fechado?")
                self.active_trade = None
                return
            except Exception as e:
                logger.warning(f"[STATE] Retry pending_close falhou — active_trade mantido: {e}")
                return

        try:
            self.check_closed_by_exchange_for_order(order_id)
            self.active_trade = None
        except Exception as e:
            logger.warning(f"[STATE] Erro ao registrar fechamento no DB — active_trade mantido: {e}")

    def check_closed_by_exchange_for_order(self, order_id: str):
        """Busca dados de fechamento para um order_id especifico."""
        try:
            # Buscar nas ordens do bot para encontrar o fechamento correto
            orders = self._api_call(
                "get_order_history",
                category="linear", symbol=cfg.SYMBOL, limit=20,
            )

            # Encontrar a ordem de TP ou SL que foi Filled
            # Filtra por ordens do bot (prefixo aitbot) para evitar casar com ordens de terceiros
            close_type = None
            for o in orders["result"]["list"]:
                link = o.get("orderLinkId", "")
                if (o["orderStatus"] == "Filled"
                        and o["stopOrderType"] in ("TakeProfit", "StopLoss")
                        and link.startswith(ORDER_PREFIX)):
                    close_type = "TP" if o["stopOrderType"] == "TakeProfit" else "SL"
                    break

            # Buscar PnL real no closed_pnl — filtrar pelo orderId de abertura
            closed = self._api_call(
                "get_closed_pnl",
                category="linear", symbol=cfg.SYMBOL, limit=10,
            )
            matched = None
            for t in closed["result"]["list"]:
                if t.get("orderId") == order_id:
                    matched = t
                    break

            if not matched:
                raise RuntimeError(f"closed_pnl nao encontrado para order_id={order_id}")

            exit_price = float(matched["avgExitPrice"])
            pnl = float(matched["closedPnl"])

            if close_type is None:
                close_type = "EXCHANGE"

            updated = db.record_close(
                order_id=order_id,
                exit_price=exit_price, pnl=pnl, close_type=close_type,
            )
            if updated:
                logger.info(f"[DB] Posicao fechada pela Bybit ({close_type}): PnL ${pnl:+,.2f}")
            else:
                logger.warning(f"[DB] record_close retornou False para order_id={order_id} — trade ja fechado?")
        except Exception as e:
            raise RuntimeError(f"Erro ao verificar fechamento: {e}") from e

    def _get_llm_model(self) -> str:
        """Retorna modelo LLM atual da config."""
        try:
            models = {
                "google": cfg.GOOGLE_MODEL,
                "anthropic": cfg.ANTHROPIC_MODEL,
                "openai": cfg.OPENAI_MODEL,
            }
            return models.get(cfg.LLM_PROVIDER.lower(), "")
        except Exception:
            return ""
