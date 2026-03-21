"""
Coleta dados de mercado da Bybit para enviar a LLM.
Inclui retry com reconexao para lidar com ConnectionResetError.
"""
import logging
import time
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP
from datetime import datetime, timezone

from config import (
    BYBIT_API_KEY, BYBIT_API_SECRET, USE_TESTNET,
    SYMBOL, TIMEFRAME, KLINES_TO_SEND, DAILY_KLINES
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2


class MarketData:
    """Coleta e formata dados de mercado."""

    def __init__(self):
        self.client = self._create_client()

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

    def get_klines(self, interval: str = None, limit: int = None) -> pd.DataFrame:
        """Busca klines e retorna DataFrame."""
        interval = interval or TIMEFRAME
        limit = limit or KLINES_TO_SEND
        result = self._api_call(
            self.client.get_kline,
            category="linear", symbol=SYMBOL,
            interval=interval, limit=limit
        )
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
        ema9 = close.ewm(span=9, adjust=False).mean()
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()

        # RSI(14)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal

        # ATR(14)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()

        # ADX(14)
        plus_dm = high.diff().where(lambda x: x > 0, 0)
        minus_dm = (-low.diff()).where(lambda x: x > 0, 0)
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(14).mean()

        # Bollinger Bands
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std

        # Volume avg
        vol_avg = df["volume"].rolling(20).mean()

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
            self.client.get_positions,
            category="linear", symbol=SYMBOL
        )
        for p in result["result"]["list"]:
            if float(p["size"]) > 0:
                return {
                    "side": p["side"],
                    "size": float(p["size"]),
                    "entry": float(p["avgPrice"]),
                    "pnl": float(p["unrealisedPnl"]),
                    "sl": float(p["stopLoss"]) if p["stopLoss"] else None,
                    "tp": float(p["takeProfit"]) if p["takeProfit"] else None,
                }
        return None

    def get_balance(self) -> float:
        """Retorna saldo USDT."""
        result = self._api_call(
            self.client.get_wallet_balance,
            accountType="UNIFIED"
        )
        for coin in result["result"]["list"][0]["coin"]:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0

    def format_for_llm(self) -> str:
        """Formata todos os dados num texto para enviar a LLM."""
        # 15min klines (24h)
        df = self.get_klines(TIMEFRAME, KLINES_TO_SEND)
        ind = self.calc_indicators(df)

        # Daily klines (30 dias)
        df_daily = self.get_klines("D", DAILY_KLINES)

        # Current state
        price = df["close"].iloc[-1]
        position = self.get_position()
        balance = self.get_balance()

        # Build text
        lines = []
        lines.append(f"=== MARKET DATA — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===")
        lines.append(f"Symbol: {SYMBOL} | Price: ${price:,.2f} | Balance: ${balance:,.2f}")
        lines.append("")

        # Position
        if position:
            lines.append(f"POSICAO ABERTA: {position['side']} {position['size']} BTC")
            lines.append(f"  Entry: ${position['entry']:,.2f} | PnL: ${position['pnl']:.4f}")
            if position['sl']:
                lines.append(f"  SL: ${position['sl']:,.2f} | TP: ${position['tp']:,.2f}")
        else:
            lines.append("POSICAO: Nenhuma")
        lines.append("")

        # Current indicators
        lines.append("=== INDICADORES ATUAIS (15min) ===")
        lines.append(f"EMA9={ind['ema9'].iloc[-1]:,.2f} EMA21={ind['ema21'].iloc[-1]:,.2f} EMA50={ind['ema50'].iloc[-1]:,.2f}")
        lines.append(f"RSI(14)={ind['rsi'].iloc[-1]:.1f} | ADX={ind['adx'].iloc[-1]:.1f}")
        lines.append(f"MACD={ind['macd'].iloc[-1]:.2f} Signal={ind['macd_signal'].iloc[-1]:.2f} Hist={ind['macd_hist'].iloc[-1]:.2f}")
        lines.append(f"ATR(14)=${ind['atr'].iloc[-1]:,.2f} ({ind['atr'].iloc[-1]/price*100:.3f}%)")
        lines.append(f"BB: Upper=${ind['bb_upper'].iloc[-1]:,.2f} Mid=${ind['bb_mid'].iloc[-1]:,.2f} Lower=${ind['bb_lower'].iloc[-1]:,.2f}")
        lines.append("")

        # Recent 15min candles (last 16 = 4h)
        lines.append("=== ULTIMOS 16 CANDLES (15min, 4h) ===")
        lines.append("Timestamp            | Open     | High     | Low      | Close    | Vol     | RSI  | MACD_H")
        for i in range(max(0, len(df) - 16), len(df)):
            row = df.iloc[i]
            lines.append(
                f"{row['timestamp'].strftime('%H:%M')} "
                f"| {row['open']:>9,.2f} | {row['high']:>9,.2f} | {row['low']:>9,.2f} "
                f"| {row['close']:>9,.2f} | {row['volume']:>7,.0f} "
                f"| {ind['rsi'].iloc[i]:>4.1f} | {ind['macd_hist'].iloc[i]:>+6.1f}"
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
