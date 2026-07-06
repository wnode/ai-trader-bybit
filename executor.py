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
    """Executa trades na Bybit baseado nas decisoes da LLM. Um executor por simbolo."""

    def __init__(self, symbol: str = None):
        self.symbol = symbol or cfg.SYMBOL
        self.client = self._create_client()
        self.active_trade = None
        self._min_qty: float | None = None
        self._qty_step: float | None = None
        self._tick_size: float | None = None
        self._price_decimals: int = 2
        self._fetch_instrument_info()
        self._restore_active_trade()

    def _fetch_instrument_info(self):
        """Busca min_qty, qty_step e tick_size do simbolo na Bybit. Cacheado por instancia."""
        try:
            result = self._api_call(
                "get_instruments_info",
                category="linear", symbol=self.symbol,
            )
            items = result.get("result", {}).get("list", [])
            if items:
                lot = items[0].get("lotSizeFilter", {})
                self._min_qty = float(lot.get("minOrderQty", "0.001"))
                self._qty_step = float(lot.get("qtyStep", "0.001"))
                price = items[0].get("priceFilter", {})
                self._tick_size = float(price.get("tickSize", "0.01"))
                # Calcular casas decimais a partir do tickSize
                tick_str = price.get("tickSize", "0.01")
                if "." in tick_str:
                    self._price_decimals = len(tick_str.split(".")[1].rstrip("0"))
                else:
                    self._price_decimals = 0
                logger.info(f"[{self.symbol}] minQty={self._min_qty} qtyStep={self._qty_step} "
                            f"tickSize={self._tick_size} priceDecimals={self._price_decimals}")
        except Exception as e:
            logger.warning(f"[{self.symbol}] Erro ao buscar instrument info: {e} — usando defaults")
            self._min_qty = 0.001
            self._qty_step = 0.001
            self._tick_size = 0.01
            self._price_decimals = 2

    def _round_price(self, price: float) -> float:
        """Arredonda preco respeitando o tickSize do simbolo."""
        tick = self._tick_size or 0.01
        rounded = round(price / tick) * tick
        return round(rounded, self._price_decimals)

    def _round_qty(self, qty: float) -> float:
        """Arredonda qty para baixo respeitando o qty_step do simbolo."""
        step = self._qty_step or 0.001
        q = (qty // step) * step
        decimals = len(f"{step:.10f}".rstrip("0").split(".")[1]) if "." in f"{step}" else 0
        return round(q, max(0, decimals))

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

    def _find_recent_entry_order_id(self) -> str | None:
        """Melhor-esforco: order_id da ordem de ABERTURA mais recente do bot
        (aitbot, Filled, nao reduceOnly) — p/ reconstruir um trade orfao."""
        try:
            orders = self._api_call("get_order_history", category="linear", symbol=self.symbol, limit=20)
            for o in orders.get("result", {}).get("list", []):
                link = str(o.get("orderLinkId", ""))
                if (o.get("orderStatus") == "Filled" and not o.get("reduceOnly")
                        and not o.get("stopOrderType") and link.startswith(ORDER_PREFIX)
                        and not link.startswith(f"{ORDER_PREFIX}-tp")
                        and not link.startswith(f"{ORDER_PREFIX}-emg")):
                    return o.get("orderId")
        except Exception:
            pass
        return None

    def _restore_active_trade(self):
        """Restaura active_trade do DB se houver trade aberto e posicao na exchange.
        Recupera tambem posicoes vivas SEM registro no DB (record_open falhou)."""
        try:
            open_trade = db.get_open_trade(self.symbol)
            pos = self._get_position()

            if not open_trade:
                if pos:
                    # Posicao viva sem registro no DB (record_open falhou) — reconstroi
                    # E registra no DB, para ser rastreada/fechavel e nao re-reconstruir.
                    order_id = self._find_recent_entry_order_id() or f"restored-{self.symbol}-{int(time.time())}"
                    side_lbl = "LONG" if pos["side"] == "Buy" else "SHORT"
                    logger.warning(f"[{self.symbol}] [RESTORE] Posicao viva SEM registro no DB "
                                   f"({pos['side']} {pos['size']}) — reconstruindo e registrando (order_id={order_id})")
                    try:
                        db.record_open(
                            symbol=self.symbol, side=side_lbl, qty=pos["size"], entry_price=pos["entry"],
                            stop_loss=None, take_profit=None, confidence=0,
                            reason="recuperado (record_open original falhou)", order_id=order_id,
                            llm_provider=cfg.LLM_PROVIDER, llm_model="",
                        )
                    except Exception as e:
                        logger.error(f"[{self.symbol}] [RESTORE] Falha ao registrar trade recuperado: {e}")
                    self.active_trade = {
                        "side": pos["side"], "entry": pos["entry"],
                        "sl": None, "tp": None, "qty": pos["size"], "init_qty": pos["size"],
                        "order_id": order_id, "partial": cfg.PARTIAL_TP_ENABLED,
                        "tp1": None, "tp2": None, "qty1": None, "qty2": None,
                        "breakeven_done": False,
                    }
                return

            if pos:
                db_qty = open_trade.get("qty") or pos["size"]
                partial = cfg.PARTIAL_TP_ENABLED and db_qty > 0
                # Se a posicao ja reduziu, o TP1 parcial ja foi atingido -> breakeven ja movido
                reduced = pos["size"] < db_qty * 0.99
                self.active_trade = {
                    "side": pos["side"],
                    "entry": pos["entry"],
                    "sl": open_trade.get("stop_loss"),
                    "tp": open_trade.get("take_profit"),
                    "qty": pos["size"],
                    "init_qty": db_qty,
                    "order_id": open_trade["order_id"],
                    "partial": partial,
                    "tp1": None, "tp2": open_trade.get("take_profit"),
                    "qty1": None, "qty2": None,
                    "breakeven_done": reduced,
                }
                logger.info(f"[{self.symbol}] [RESTORE] Trade restaurado do DB: {open_trade['side']} {pos['size']}"
                            + (" (TP parcial)" if partial else "")
                            + (" [TP1 ja atingido -> breakeven]" if reduced else ""))
            else:
                # Posicao nao existe mais, fechar trade orfao no DB
                try:
                    self.check_closed_by_exchange_for_order(open_trade["order_id"])
                except Exception as e:
                    # Nao reconciliou (fechamento antigo demais p/ closed_pnl) — fecha a linha
                    # como UNKNOWN p/ nao re-restaurar um fantasma a cada reinicio.
                    logger.warning(f"[{self.symbol}] [RESTORE] Nao reconciliou order_id={open_trade['order_id']} "
                                   f"— marcando UNKNOWN p/ liberar: {e}")
                    db.record_close(open_trade["order_id"],
                                    exit_price=open_trade.get("entry_price") or 0.0,
                                    pnl=0.0, close_type="UNKNOWN")
        except Exception as e:
            logger.warning(f"[{self.symbol}] [RESTORE] Erro ao restaurar trade: {e}")

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
        """Calcula tamanho da posicao baseado no risco. Respeita min_qty e qty_step do simbolo."""
        balance = self.get_balance()
        risk_amount = balance * cfg.RISK_PER_TRADE
        sl_dist = abs(entry - sl) / entry
        if sl_dist == 0:
            return 0.0
        notional = risk_amount / sl_dist
        qty = notional / entry

        # Arredondar para qty_step do simbolo
        min_qty = self._min_qty or 0.001
        qty = self._round_qty(qty)

        if qty < min_qty:
            min_risk = min_qty * abs(entry - sl)
            logger.warning(f"[{self.symbol}] [RISK] Qty calculada {qty} < minimo {min_qty}. "
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

        # SL e alvos respeitando tickSize. Runner (TP2) sempre em MIN_RR_RATIO:1.
        sl = self._round_price(sl)
        sl_dist = abs(entry - sl)
        direction = 1 if action == "LONG" else -1
        tp_runner = self._round_price(entry + direction * sl_dist * cfg.MIN_RR_RATIO)

        qty = self.calc_position_size(entry, sl)
        if qty <= 0:
            return "Qty zero — risco muito baixo para abrir posicao"

        side = "Buy" if action == "LONG" else "Sell"

        # Decide se usa TP parcial + runner ou TP unico
        use_partial = cfg.PARTIAL_TP_ENABLED
        tp1 = qty1 = qty2 = None
        if use_partial:
            tp1 = self._round_price(entry + direction * sl_dist * cfg.TP1_RR_RATIO)
            qty1 = self._round_qty(qty * cfg.TP1_SIZE_PCT)
            qty2 = self._round_qty(qty - qty1)
            min_qty = self._min_qty or 0.001
            if qty1 < min_qty or qty2 < min_qty:
                logger.warning(f"[{self.symbol}] [TP parcial] qty {qty} nao divisivel "
                               f"(qty1={qty1}, qty2={qty2}, min={min_qty}) — usando TP unico {cfg.MIN_RR_RATIO}:1")
                use_partial = False

        if use_partial:
            logger.info(f"[{self.symbol}] [TP parcial] entry=${entry:,.4f} SL=${sl:,.4f} "
                        f"TP1=${tp1:,.4f} ({qty1}@{cfg.TP1_RR_RATIO}:1) "
                        f"TP2=${tp_runner:,.4f} ({qty2}@{cfg.MIN_RR_RATIO}:1)")
        else:
            logger.info(f"[{self.symbol}] [TP] entry=${entry:,.4f} SL=${sl:,.4f} "
                        f"TP=${tp_runner:,.4f} (R:R={cfg.MIN_RR_RATIO}:1)")

        if cfg.DRY_RUN:
            self.active_trade = {
                "side": side, "entry": entry, "sl": sl, "tp": tp_runner,
                "qty": qty, "init_qty": qty, "reason": decision.get("reason", ""),
                "partial": use_partial, "tp1": tp1, "tp2": tp_runner,
                "qty1": qty1, "qty2": qty2, "breakeven_done": False,
            }
            extra = (f" TP1=${tp1:,.4f}({qty1}) TP2=${tp_runner:,.4f}({qty2})"
                     if use_partial else f" TP=${tp_runner:,.4f}")
            return (f"[{self.symbol}] [DRY] {action} {qty} @ ${entry:,.4f} "
                    f"SL=${sl:,.4f}{extra} (conf={decision.get('confidence', 0):.1f})")

        try:
            link_id = f"{ORDER_PREFIX}-{uuid.uuid4().hex[:16]}"
            order_params = {
                "category": "linear",
                "symbol": self.symbol,
                "side": side,
                "orderType": "Market",
                "qty": str(qty),
                "positionIdx": 0,
                "orderLinkId": link_id,
            }
            result = self._api_call("place_order", **order_params)
            if result["retCode"] == 0:
                order_id = result["result"]["orderId"]

                # Preco real de fill — SL/TP recalculados a partir DELE (nao do sugerido)
                real_entry = self._get_fill_price(order_id, fallback=entry)

                # Valida que o SL esta do lado correto do fill (evita stop invertido em gap)
                if (direction == 1 and sl >= real_entry) or (direction == -1 and sl <= real_entry):
                    self._emergency_close(side, qty, f"SL ${sl:,.4f} do lado errado do fill ${real_entry:,.4f}")
                    return f"[{self.symbol}] SL invalido apos fill (${real_entry:,.4f}) — posicao fechada por seguranca"

                # Recalcula os alvos a partir do fill real, preservando a distancia do SL
                sl_dist_real = abs(real_entry - sl)
                tp_runner = self._round_price(real_entry + direction * sl_dist_real * cfg.MIN_RR_RATIO)
                if use_partial:
                    tp1 = self._round_price(real_entry + direction * sl_dist_real * cfg.TP1_RR_RATIO)
                    # SL Market (Full mode) cobre a posicao inteira e encolhe com ela.
                    # Se o SL falhar, fecha a posicao — nunca deixar posicao sem stop.
                    if not self._set_sl_only(sl):
                        self._emergency_close(side, qty, "Falha ao definir SL (TP parcial)")
                        return f"[{self.symbol}] Falha ao definir SL — posicao fechada por seguranca"
                    # TP1/TP2 como ordens reduceOnly Limit (maker fee) em niveis distintos
                    self._place_reduce_limit(tp1, qty1, side)
                    self._place_reduce_limit(tp_runner, qty2, side)
                else:
                    # SL (Market) + TP (Limit) via set_trading_stop. Fecha se falhar.
                    if not self._set_sl_tp(sl, tp_runner):
                        self._emergency_close(side, qty, "Falha ao definir SL/TP")
                        return f"[{self.symbol}] Falha ao definir SL/TP — posicao fechada por seguranca"

                try:
                    db.record_open(
                        symbol=self.symbol,
                        side=action, qty=qty, entry_price=real_entry,
                        stop_loss=sl, take_profit=tp_runner,
                        confidence=decision.get("confidence", 0),
                        reason=decision.get("reason", ""),
                        order_id=order_id,
                        llm_provider=cfg.LLM_PROVIDER,
                        llm_model=self._get_llm_model(),
                    )
                except Exception as e:
                    logger.error(f"[{self.symbol}] [DB] Falha ao registrar abertura no DB (posicao JA aberta na exchange) order_id={order_id}: {e}")
                self.active_trade = {
                    "side": side, "entry": real_entry, "sl": sl, "tp": tp_runner,
                    "qty": qty, "init_qty": qty, "order_id": order_id,
                    "partial": use_partial, "tp1": tp1, "tp2": tp_runner,
                    "qty1": qty1, "qty2": qty2, "breakeven_done": False,
                }
                if use_partial:
                    return (f"[{self.symbol}] {action} {qty} @ ${real_entry:,.4f} "
                            f"SL=${sl:,.4f} TP1=${tp1:,.4f}({qty1}) "
                            f"TP2=${tp_runner:,.4f}({qty2}) OrderID={order_id}")
                return (f"[{self.symbol}] {action} {qty} @ ${real_entry:,.4f} "
                        f"SL=${sl:,.4f} TP=${tp_runner:,.4f}(Limit) "
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
            return f"[{self.symbol}] [DRY] Fechando {pos['side']} {pos['size']}"

        # Com TP parcial ha ordens reduceOnly Limit pendentes — cancela antes de
        # fechar a mercado, senao a Bybit pode rejeitar (qty reduceOnly > posicao).
        if self.active_trade and self.active_trade.get("partial"):
            self._cancel_open_orders()
            # Re-le o tamanho: o TP1 pode ter enchido entre o snapshot e o cancelamento
            pos2 = self._get_position()
            if not pos2:
                logger.info(f"[{self.symbol}] [CLOSE] Posicao ja fechada apos cancelar ordens — reconciliando")
                self.check_closed_by_exchange()
                return "Posicao ja fechada (reconciliada)"
            pos = pos2

        try:
            close_side = "Sell" if pos["side"] == "Buy" else "Buy"
            link_id = f"{ORDER_PREFIX}-{uuid.uuid4().hex[:16]}"
            result = self._api_call(
                "place_order",
                category="linear",
                symbol=self.symbol,
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

    def _set_sl_tp(self, sl: float, tp: float) -> bool:
        """Define SL (Market) e TP (Limit, maker fee 0.020%) via set_trading_stop.
        Usa tpslMode=Partial para permitir tpOrderType=Limit.
        Se Limit falhar, tenta TP Market como fallback. Retorna True se OK, False
        se nem o fallback definiu o SL (posicao ficaria sem protecao)."""
        sl_str = str(self._round_price(sl))
        tp_str = str(self._round_price(tp))
        try:
            result = self._api_call(
                "set_trading_stop",
                category="linear",
                symbol=self.symbol,
                stopLoss=sl_str,
                slTriggerBy="LastPrice",
                takeProfit=tp_str,
                tpTriggerBy="LastPrice",
                tpslMode="Partial",
                tpOrderType="Limit",
                tpLimitPrice=tp_str,
                tpSize="0",
                slSize="0",
                positionIdx=0,
            )
            if result["retCode"] == 0:
                logger.info(f"[SL/TP] SL Market @ ${sl:,.2f} | TP Limit @ ${tp:,.2f} (maker fee)")
                return True
            logger.warning(f"[SL/TP] Falha Partial: {result['retMsg']} — tentando Full mode")
        except Exception as e:
            logger.warning(f"[SL/TP] Erro Partial: {e} — tentando Full mode")
        # Fallback: Full mode (SL + TP ambos Market)
        try:
            result = self._api_call(
                "set_trading_stop",
                category="linear",
                symbol=self.symbol,
                stopLoss=sl_str,
                slTriggerBy="LastPrice",
                takeProfit=tp_str,
                tpTriggerBy="LastPrice",
                positionIdx=0,
            )
            if result["retCode"] == 0:
                logger.info(f"[SL/TP] SL Market @ ${sl:,.2f} | TP Market @ ${tp:,.2f} (fallback)")
                return True
            logger.error(f"[SL/TP] Falha fallback: {result['retMsg']} — posicao SEM SL/TP!")
        except Exception as e:
            logger.error(f"[SL/TP] Erro fallback: {e} — posicao SEM SL/TP!")
        return False

    def _emergency_close(self, side: str, qty: float, reason: str):
        """Fecha a mercado (reduceOnly) uma posicao recem-aberta que ficou sem
        protecao (SL falhou / invalido). Re-le o tamanho REAL e so limpa o
        active_trade se a Bybit CONFIRMAR o fechamento (retCode==0). Em falha,
        mantem/reconstroi active_trade para nao deixar posicao orfã sem gestao."""
        if cfg.DRY_RUN:
            self.active_trade = None
            return
        pos = self._get_position()
        if not pos:
            self.active_trade = None
            return  # posicao nunca abriu / ja fechou
        close_side = "Sell" if pos["side"] == "Buy" else "Buy"
        try:
            result = self._api_call(
                "place_order",
                category="linear", symbol=self.symbol, side=close_side,
                orderType="Market", qty=str(pos["size"]), reduceOnly=True, positionIdx=0,
                orderLinkId=f"{ORDER_PREFIX}-emg-{uuid.uuid4().hex[:10]}",
            )
            if result and result.get("retCode") == 0:
                logger.error(f"[{self.symbol}] [SAFETY] {reason} — posicao fechada a mercado")
                self.active_trade = None
                return
            msg = result.get("retMsg") if result else "sem resposta"
            logger.critical(f"[{self.symbol}] [SAFETY] {reason} — fechamento REJEITADO ({msg}) "
                            f"— POSICAO ABERTA SEM STOP! Mantendo controle p/ nova tentativa.")
        except Exception as e:
            logger.critical(f"[{self.symbol}] [SAFETY] {reason} — FALHA ao fechar: {e} "
                            f"— POSICAO PODE ESTAR SEM STOP! Mantendo controle.")
        # Fechamento falhou: reconstroi active_trade minimo a partir da posicao viva
        # para que o proximo ciclo/monitor a veja (nao orfanar em memoria).
        self.active_trade = {
            "side": pos["side"], "entry": pos["entry"], "qty": pos["size"],
            "init_qty": pos["size"], "sl": None, "tp": None, "order_id": None,
            "partial": False, "breakeven_done": True, "needs_emergency_close": True,
        }

    def _set_sl_only(self, sl: float) -> bool:
        """Define apenas o SL (Market, Full mode) cobrindo toda a posicao.
        Usado com TP parcial, onde os TPs sao ordens reduceOnly Limit separadas.
        O SL em Full mode encolhe junto com a posicao quando o TP1 e atingido."""
        sl_str = str(self._round_price(sl))
        try:
            result = self._api_call(
                "set_trading_stop",
                category="linear",
                symbol=self.symbol,
                stopLoss=sl_str,
                slTriggerBy="LastPrice",
                tpslMode="Full",
                positionIdx=0,
            )
            if result["retCode"] == 0:
                logger.info(f"[{self.symbol}] [SL] Market @ ${sl:,.4f} (Full)")
                return True
            logger.error(f"[{self.symbol}] [SL] Falha: {result['retMsg']} — posicao SEM SL!")
        except Exception as e:
            logger.error(f"[{self.symbol}] [SL] Erro: {e} — posicao SEM SL!")
        return False

    def _place_reduce_limit(self, price: float, qty: float, side: str) -> str | None:
        """Coloca ordem de TP reduceOnly Limit (maker fee). Retorna orderId ou None.
        side = lado da POSICAO (Buy/Sell); a ordem de TP e do lado oposto."""
        close_side = "Sell" if side == "Buy" else "Buy"
        price_str = str(self._round_price(price))
        try:
            link_id = f"{ORDER_PREFIX}-tp-{uuid.uuid4().hex[:12]}"
            result = self._api_call(
                "place_order",
                category="linear",
                symbol=self.symbol,
                side=close_side,
                orderType="Limit",
                qty=str(qty),
                price=price_str,
                reduceOnly=True,
                timeInForce="GTC",
                positionIdx=0,
                orderLinkId=link_id,
            )
            if result["retCode"] == 0:
                logger.info(f"[{self.symbol}] [TP] reduceOnly Limit {qty} @ ${price:,.4f} OK")
                return result["result"]["orderId"]
            logger.error(f"[{self.symbol}] [TP] Falha reduceOnly Limit {qty} @ ${price:,.4f}: {result['retMsg']}")
        except Exception as e:
            logger.error(f"[{self.symbol}] [TP] Erro reduceOnly Limit: {e}")
        return None

    def _cancel_open_orders(self):
        """Cancela ordens abertas do simbolo (ex: TP runner orfao apos SL/CLOSE).
        Seguro: opera uma posicao por simbolo, entao so cancela ordens deste trade."""
        try:
            self._api_call("cancel_all_orders", category="linear", symbol=self.symbol)
        except Exception as e:
            logger.warning(f"[{self.symbol}] [CANCEL] Erro ao cancelar ordens orfas: {e}")

    def manage_open_position(self):
        """Gestao de posicao aberta com TP parcial: quando o TP1 e atingido
        (posicao reduzida em relacao ao tamanho inicial), move o SL para
        breakeven, tornando o runner um trade de risco zero."""
        at = self.active_trade
        if not at:
            return
        # Retry do fechamento de emergencia (SL falhou e o close anterior foi rejeitado)
        if at.get("needs_emergency_close") and not cfg.DRY_RUN:
            self._emergency_close(at.get("side"), at.get("qty", 0), "retry fechamento de emergencia")
            return
        if not at.get("partial") or at.get("breakeven_done"):
            return
        if not cfg.BREAKEVEN_AFTER_TP1 or cfg.DRY_RUN:
            return

        pos = self._get_position()
        if not pos:
            return  # fechada — check_closed_by_exchange cuida do registro

        init_qty = at.get("init_qty") or at.get("qty") or 0
        if init_qty <= 0 or pos["size"] >= init_qty * 0.99:
            return  # TP1 ainda nao atingido (posicao quase intacta)

        # TP1 atingido — move SL para breakeven (entry + offset p/ cobrir fees)
        entry = at["entry"]
        offset = entry * (cfg.BREAKEVEN_OFFSET_PCT / 100.0)
        be = entry + offset if at["side"] == "Buy" else entry - offset
        be = self._round_price(be)
        if self._set_sl_only(be):
            at["sl"] = be
            at["breakeven_done"] = True
            logger.info(f"[{self.symbol}] [BREAKEVEN] TP1 atingido (resta {pos['size']}) — "
                        f"SL movido para ${be:,.4f} (runner risco-zero)")

    def _get_fill_price(self, order_id: str, fallback: float) -> float:
        """Busca preco real de fill de uma ordem."""
        try:
            time.sleep(0.5)  # Aguarda fill propagar
            result = self._api_call(
                "get_executions",
                category="linear", symbol=self.symbol, orderId=order_id, limit=1,
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
            category="linear", symbol=self.symbol
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

        # Posicao fechada: cancela ordens orfas (ex: TP runner pendente apos SL)
        if self.active_trade.get("partial"):
            self._cancel_open_orders()

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
                # Nao travar o simbolo para sempre: apos N tentativas, desiste
                # (a posicao ja esta fechada na exchange; so o registro de PnL se perde).
                pending["retries"] = pending.get("retries", 0) + 1
                if pending["retries"] >= 5:
                    logger.critical(f"[STATE] Retry pending_close falhou {pending['retries']}x — "
                                    f"desistindo do registro e liberando o simbolo: {e}")
                    self.active_trade = None
                else:
                    logger.warning(f"[STATE] Retry pending_close falhou ({pending['retries']}/5) — mantido: {e}")
                return

        try:
            self.check_closed_by_exchange_for_order(order_id)
            self.active_trade = None
        except Exception as e:
            logger.warning(f"[STATE] Erro ao registrar fechamento no DB — active_trade mantido: {e}")

    def check_closed_by_exchange_for_order(self, order_id: str):
        """Busca dados de fechamento para um order_id especifico do simbolo.
        Valida que o trade fechado e do nosso ciclo via timestamp da abertura."""
        try:
            # Buscar a abertura para validar que fechamento e do nosso trade
            opened_ms = None
            try:
                open_orders = self._api_call(
                    "get_order_history",
                    category="linear", symbol=self.symbol, orderId=order_id, limit=1,
                )
                items = open_orders.get("result", {}).get("list", [])
                if items:
                    opened_ms = int(items[0].get("createdTime", "0"))
            except Exception:
                pass

            # Buscar nas ordens do simbolo para encontrar o fechamento correto
            orders = self._api_call(
                "get_order_history",
                category="linear", symbol=self.symbol, limit=20,
            )

            # Encontrar como a posicao fechou (ordem mais recente = fechamento final).
            # SL vem via set_trading_stop (stopOrderType=StopLoss). TPs parciais sao
            # ordens reduceOnly Limit com prefixo aitbot-tp (stopOrderType vazio).
            close_type = None
            close_ms = None   # horario do fechamento final (limite superior da agregacao)
            for o in orders["result"]["list"]:
                if o["orderStatus"] != "Filled":
                    continue
                if opened_ms and int(o.get("updatedTime", "0")) < opened_ms:
                    continue  # fechamento antigo, ignorar
                sot = o.get("stopOrderType", "")
                link = o.get("orderLinkId", "")
                if sot == "StopLoss":
                    close_type = "SL"
                    close_ms = int(o.get("updatedTime", "0"))
                    break
                if sot == "TakeProfit" or link.startswith(f"{ORDER_PREFIX}-tp"):
                    close_type = "TP"
                    close_ms = int(o.get("updatedTime", "0"))
                    break
                if link.startswith(f"{ORDER_PREFIX}-emg"):
                    close_type = "EMG"   # fechamento de emergencia (SL falhou)
                    close_ms = int(o.get("updatedTime", "0"))
                    break
                if o.get("reduceOnly"):
                    # Fechamento a mercado externo / liquidacao — tambem delimita a janela
                    close_type = "EXCHANGE"
                    close_ms = int(o.get("updatedTime", "0"))
                    break

            # Buscar PnL real no closed_pnl — somar TODOS os fechamentos do trade
            # (com TP parcial ha 2+ fills: TP1 + TP2/SL), mas SO dentro da janela
            # [opened_ms, close_ms + buffer] para nao pegar um trade posterior do mesmo simbolo.
            closed = self._api_call(
                "get_closed_pnl",
                category="linear", symbol=self.symbol, limit=20,
            )
            upper_ms = (close_ms + 5000) if close_ms else None
            total_pnl = 0.0
            weighted_exit = 0.0
            total_qty = 0.0
            matched_any = False
            for item in closed["result"]["list"]:
                # closed_pnl tem orderId da ordem de fechamento, nao da abertura.
                # Validamos pela janela temporal do trade: opened_ms <= createdTime <= upper_ms
                item_ms = int(item.get("createdTime", "0"))
                if opened_ms and item_ms < opened_ms:
                    continue  # fechamento de trade anterior, ignorar
                if upper_ms and item_ms > upper_ms:
                    continue  # fechamento de trade POSTERIOR, ignorar (evita double-count)
                qty_c = float(item.get("qty", "0") or 0)
                total_pnl += float(item["closedPnl"])
                weighted_exit += float(item["avgExitPrice"]) * qty_c
                total_qty += qty_c
                matched_any = True
                if opened_ms is None:
                    break  # sem timestamp de abertura (orfa): usa so o fechamento mais recente

            if not matched_any:
                raise RuntimeError(f"closed_pnl vazio para {self.symbol} apos {opened_ms}")

            exit_price = (weighted_exit / total_qty) if total_qty > 0 else 0.0
            pnl = total_pnl

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
                "xai": cfg.XAI_MODEL,
            }
            return models.get(cfg.LLM_PROVIDER.lower(), "")
        except Exception:
            return ""
