"""
Coleta dados de mercado da Bybit para enviar a LLM.
Inclui retry com reconexao para lidar com ConnectionResetError.
"""
import logging
import time
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from pybit.exceptions import FailedRequestError
from datetime import datetime, timezone

import config as cfg

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2


class MarketData:
    """Coleta e formata dados de mercado."""

    def __init__(self):
        self.client = self._create_client()

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
            except Exception as e:
                if "retryable" in str(e).lower() or "recv_window" in str(e).lower():
                    # Nao recriar client — pybit ajusta timestamp_offset internamente
                    logger.warning(f"[RETRY] Tentativa {attempt+1}/{MAX_RETRIES} (timestamp): {e}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY * (2 ** attempt))
                    else:
                        raise
                else:
                    raise

    def get_klines(self, interval: str = None, limit: int = None) -> pd.DataFrame:
        """Busca klines e retorna DataFrame."""
        interval = interval or cfg.TIMEFRAME
        limit = limit or cfg.KLINES_TO_SEND
        result = self._api_call(
            "get_kline",
            category="linear", symbol=cfg.SYMBOL,
            interval=interval, limit=limit
        )
        if not result or result.get("retCode") != 0:
            msg = result.get("retMsg", "unknown") if result else "sem resposta"
            raise RuntimeError(f"Erro API get_kline: {msg}")
        rows = []
        for k in result["result"]["list"]:
            ts, o, h, l, c, vol, turnover = k
            rows.append({
                "timestamp": datetime.fromtimestamp(int(ts) / 1000, tz=timezone.utc),
                "open": float(o), "high": float(h), "low": float(l),
                "close": float(c), "volume": float(vol),
            })
        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        return df

    def calc_indicators(self, df: pd.DataFrame) -> dict:
        """Calcula indicadores tecnicos."""
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # EMAs
        ema9 = close.ewm(span=cfg.EMA_FAST, adjust=False).mean()
        ema21 = close.ewm(span=cfg.EMA_MID, adjust=False).mean()
        ema50 = close.ewm(span=cfg.EMA_SLOW, adjust=False).mean()

        # RSI — Wilder's smoothing
        rsi_p = cfg.RSI_PERIOD
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss_raw = (-delta.where(delta < 0, 0))
        avg_gain = gain.ewm(alpha=1/rsi_p, min_periods=rsi_p, adjust=False).mean()
        avg_loss = loss_raw.ewm(alpha=1/rsi_p, min_periods=rsi_p, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # Neutro ate ter dados suficientes

        # MACD
        ema_fast = close.ewm(span=cfg.MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=cfg.MACD_SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=cfg.MACD_SIGNAL, adjust=False).mean()
        macd_hist = macd - signal

        # ATR — Wilder's smoothing
        atr_p = cfg.ATR_PERIOD
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/atr_p, min_periods=atr_p, adjust=False).mean()
        atr = atr.replace(0, np.finfo(float).eps)  # Evita divisao por zero no DI

        # ADX — corrigido: mutual exclusion antes de zerar negativos
        adx_p = cfg.ADX_PERIOD
        plus_dm_raw = high.diff()
        minus_dm_raw = -low.diff()
        plus_dm = pd.Series(0.0, index=df.index)
        minus_dm = pd.Series(0.0, index=df.index)
        cond_plus = (plus_dm_raw > 0) & (plus_dm_raw > minus_dm_raw)
        cond_minus = (minus_dm_raw > 0) & (minus_dm_raw > plus_dm_raw)
        plus_dm[cond_plus] = plus_dm_raw[cond_plus]
        minus_dm[cond_minus] = minus_dm_raw[cond_minus]

        smooth_plus_dm = plus_dm.ewm(alpha=1/adx_p, min_periods=adx_p, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(alpha=1/adx_p, min_periods=adx_p, adjust=False).mean()
        # ATR para DI usa mesmo periodo do ADX (pode diferir do ATR_PERIOD)
        atr_for_adx = tr.ewm(alpha=1/adx_p, min_periods=adx_p, adjust=False).mean()
        atr_for_adx = atr_for_adx.replace(0, np.finfo(float).eps)
        plus_di = 100 * (smooth_plus_dm / atr_for_adx)
        minus_di = 100 * (smooth_minus_dm / atr_for_adx)
        di_sum = plus_di + minus_di
        dx = 100 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
        dx = dx.fillna(0)
        adx = dx.ewm(alpha=1/adx_p, min_periods=adx_p, adjust=False).mean()
        adx = adx.fillna(0)

        # Bollinger Bands (ddof=0 para population std)
        bb_period = min(cfg.BB_PERIOD, len(close))
        bb_mid = close.rolling(bb_period).mean()
        bb_std = close.rolling(bb_period).std(ddof=0)
        bb_upper = bb_mid + cfg.BB_STD * bb_std
        bb_lower = bb_mid - cfg.BB_STD * bb_std

        # Volume avg
        vol_avg = df["volume"].rolling(cfg.VOL_AVG_PERIOD).mean()

        return {
            "ema9": ema9, "ema21": ema21, "ema50": ema50,
            "rsi": rsi, "macd": macd, "macd_signal": signal,
            "macd_hist": macd_hist, "atr": atr, "adx": adx,
            "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
            "vol_avg": vol_avg,
        }

    def get_position(self) -> dict | None:
        """Retorna posicao aberta ou None."""
        result = self._api_call(
            "get_positions",
            category="linear", symbol=cfg.SYMBOL
        )
        if not result or result.get("retCode") != 0:
            msg = result.get("retMsg", "unknown") if result else "sem resposta"
            raise RuntimeError(f"Erro API get_positions: {msg}")
        for p in result["result"]["list"]:
            if float(p["size"]) > 0:
                sl_val = p.get("stopLoss") or ""
                tp_val = p.get("takeProfit") or ""
                return {
                    "side": p["side"],
                    "size": float(p["size"]),
                    "entry": float(p["avgPrice"]),
                    "pnl": float(p["unrealisedPnl"]),
                    "sl": float(sl_val) if sl_val.strip() else None,
                    "tp": float(tp_val) if tp_val.strip() else None,
                }
        return None

    def get_balance(self) -> float:
        """Retorna saldo USDT."""
        result = self._api_call(
            "get_wallet_balance",
            accountType="UNIFIED"
        )
        if not result or result.get("retCode") != 0:
            msg = result.get("retMsg", "unknown") if result else "sem resposta"
            raise RuntimeError(f"Erro API get_wallet_balance: {msg}")
        for coin in result["result"]["list"][0]["coin"]:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0

    def format_for_llm(self) -> str:
        """Formata todos os dados num texto para enviar a LLM."""
        # 15min klines (24h)
        df = self.get_klines(cfg.TIMEFRAME, cfg.KLINES_TO_SEND)
        if df.empty:
            raise ValueError(f"Nenhum dado de klines recebido para {cfg.SYMBOL} ({cfg.TIMEFRAME})")
        ind = self.calc_indicators(df)

        # Daily klines (30 dias)
        df_daily = self.get_klines("D", cfg.DAILY_KLINES)
        if df_daily.empty:
            raise ValueError(f"Nenhum dado de klines diarios recebido para {cfg.SYMBOL}")

        # Current state
        price = df["close"].iloc[-1]
        position = self.get_position()
        balance = self.get_balance()

        # Build text
        lines = []
        lines.append(f"=== MARKET DATA — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===")
        lines.append(f"Symbol: {cfg.SYMBOL} | Price: ${price:,.2f} | Balance: ${balance:,.2f}")
        lines.append("")

        # Position
        if position:
            lines.append(f"POSICAO ABERTA: {position['side']} {position['size']} BTC")
            lines.append(f"  Entry: ${position['entry']:,.2f} | PnL: ${position['pnl']:.4f}")
            if position['sl'] is not None and position['tp'] is not None:
                lines.append(f"  SL: ${position['sl']:,.2f} | TP: ${position['tp']:,.2f}")
        else:
            lines.append("POSICAO: Nenhuma")
        lines.append("")

        # Current indicators — tratar NaN
        rsi_val = ind['rsi'].iloc[-1]
        adx_val = ind['adx'].iloc[-1]

        atr_val = ind['atr'].iloc[-1]
        macd_val = ind['macd'].iloc[-1]
        macd_sig_val = ind['macd_signal'].iloc[-1]
        macd_hist_val = ind['macd_hist'].iloc[-1]
        bb_upper_val = ind['bb_upper'].iloc[-1]
        bb_mid_val = ind['bb_mid'].iloc[-1]
        bb_lower_val = ind['bb_lower'].iloc[-1]

        lines.append("=== INDICADORES ATUAIS (15min) ===")
        lines.append(f"EMA9={ind['ema9'].iloc[-1]:,.2f} EMA21={ind['ema21'].iloc[-1]:,.2f} EMA50={ind['ema50'].iloc[-1]:,.2f}")
        lines.append(f"RSI(14)={'N/A' if pd.isna(rsi_val) else f'{rsi_val:.1f}'} | ADX={'N/A' if pd.isna(adx_val) else f'{adx_val:.1f}'}")
        lines.append(f"MACD={'N/A' if pd.isna(macd_val) else f'{macd_val:.2f}'} Signal={'N/A' if pd.isna(macd_sig_val) else f'{macd_sig_val:.2f}'} Hist={'N/A' if pd.isna(macd_hist_val) else f'{macd_hist_val:.2f}'}")
        if pd.isna(atr_val):
            lines.append("ATR(14)=N/A")
        else:
            lines.append(f"ATR(14)=${atr_val:,.2f} ({atr_val/price*100:.3f}%)")
        if pd.isna(bb_upper_val):
            lines.append("BB: N/A")
        else:
            lines.append(f"BB: Upper=${bb_upper_val:,.2f} Mid=${bb_mid_val:,.2f} Lower=${bb_lower_val:,.2f}")
        lines.append("")

        # Recent 15min candles (last 16 = 4h)
        lines.append("=== ULTIMOS 16 CANDLES (15min, 4h) ===")
        lines.append("Timestamp            | Open     | High     | Low      | Close    | Vol     | RSI  | MACD_H")
        for i in range(max(0, len(df) - 16), len(df)):
            row = df.iloc[i]
            rsi_i = ind['rsi'].iloc[i]
            macd_h_i = ind['macd_hist'].iloc[i]
            lines.append(
                f"{row['timestamp'].strftime('%H:%M')} "
                f"| {row['open']:>9,.2f} | {row['high']:>9,.2f} | {row['low']:>9,.2f} "
                f"| {row['close']:>9,.2f} | {row['volume']:>7,.0f} "
                f"| {'N/A' if pd.isna(rsi_i) else f'{rsi_i:>4.1f}'} "
                f"| {'  N/A' if pd.isna(macd_h_i) else f'{macd_h_i:>+6.1f}'}"
            )
        lines.append("")

        # Daily candles (last 10)
        lines.append("=== ULTIMOS 10 CANDLES DIARIOS ===")
        lines.append("Date       | Open     | High     | Low      | Close    | Range%")
        for i in range(max(0, len(df_daily) - 10), len(df_daily)):
            row = df_daily.iloc[i]
            rng = (row["high"] - row["low"]) / row["low"] * 100
            lines.append(
                f"{row['timestamp'].strftime('%Y-%m-%d')} "
                f"| {row['open']:>9,.2f} | {row['high']:>9,.2f} | {row['low']:>9,.2f} "
                f"| {row['close']:>9,.2f} | {rng:.2f}%"
            )

        return "\n".join(lines)
