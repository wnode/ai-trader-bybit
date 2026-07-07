"""
Lider de Mercado (BTC/ETH) — filtro direcional para as altcoins.

Tese: altcoins amplificam os movimentos de BTC/ETH (beta alto). Este modulo
calcula o vies (bullish/bearish/neutral) do BTC (lider de todas as alts) e do
ETH (reforco para alts do ecossistema ETH) e expoe:
  - format_for_llm(): bloco de contexto injetado no prompt de cada alt
  - get_bias(symbol): gate rigido {allow_long, allow_short} usado no main loop

Seguranca: se a coleta de dados de um lider falhar, o filtro e FAIL-OPEN
(nao bloqueia trades) — nunca travar tudo por falha de dado.
Reutiliza MarketData.get_klines/calc_indicators (sem duplicar indicadores).
"""
import logging
import pandas as pd

import config as cfg
from market_data import MarketData

logger = logging.getLogger(__name__)

BULLISH, BEARISH, NEUTRAL = "bullish", "bearish", "neutral"

# Warmup de indicadores: EMA/MACD/ADX sao recursivos; poucas barras dao valores
# diferentes do backtest (historico longo). ~400 barras estabilizam o transiente.
SIGNAL_WARMUP_BARS = 400


class LeaderSignal:
    """Coleta e avalia o vies direcional dos lideres de mercado (BTC/ETH)."""

    def __init__(self):
        self._markets: dict[str, MarketData] = {}   # symbol -> MarketData
        self._cache: dict[str, dict] = {}           # symbol -> {"bias", "detail"} (por iteracao)
        self._regime: int = 0                        # +1 BTC>MA200D, -1 abaixo, 0 desconhecido
        # Constroi os lideres se o filtro esta ligado, se operamos sem LLM
        # (modo mecanico), ou no modo hibrido (LLM-como-filtro dos setups).
        if cfg.LEADER_FILTER_ENABLED or not cfg.USE_LLM or cfg.LLM_AS_FILTER:
            self._ensure_market(cfg.LEADER_BTC_SYMBOL)
            # So constroi o lider ETH se algum simbolo do ecossistema ETH usa
            if cfg.LEADER_ETH_SYMBOLS:
                self._ensure_market(cfg.LEADER_ETH_SYMBOL)

    def _ensure_market(self, symbol: str):
        """Cria (lazy) um MarketData para o lider. Falha -> lider ausente (fail-open)."""
        if symbol in self._markets:
            return
        try:
            self._markets[symbol] = MarketData(symbol)
            logger.info(f"[LEADER] Lider inicializado: {symbol}")
        except Exception as e:
            logger.warning(f"[LEADER] Falha ao inicializar lider {symbol}, fail-open: {e}")

    def refresh(self):
        """Recalcula o vies de cada lider. Chamado 1x por iteracao (compartilhado).
        Cada lider e isolado: falha em um nao afeta o outro (fail-open)."""
        if not self._markets:
            return
        self._cache = {}
        for sym, market in self._markets.items():
            try:
                self._cache[sym] = self._compute_bias(market)
            except Exception as e:
                # Fail-open: sem entrada no cache -> get_bias trata como nao-bloqueante
                logger.warning(f"[LEADER] Falha ao obter dados de {sym}, filtro fail-open: {e}")

        # Regime macro (BTC vs MA de N dias) — usado no filtro hibrido
        if cfg.HYBRID_REGIME_FILTER and (cfg.LLM_AS_FILTER or not cfg.USE_LLM):
            self._regime = self._compute_regime()

    def _compute_regime(self) -> int:
        """+1 se BTC acima da MA(REGIME_MA_DAYS) diaria, -1 se abaixo. Em falha
        transitoria (fetch/NaN) MANTEM o ultimo regime bom (self._regime) para nao
        paralisar as entradas das 25 moedas por um soluco de API. So e 0 no startup."""
        btc = self._markets.get(cfg.LEADER_BTC_SYMBOL)
        if not btc:
            return self._regime
        try:
            df = btc.get_klines("D", cfg.REGIME_MA_DAYS + 50, closed_only=True)
            ma = df["close"].rolling(cfg.REGIME_MA_DAYS).mean().iloc[-1]
            if pd.isna(ma):
                return self._regime   # MA nao pronta — mantem ultimo
            reg = 1 if df["close"].iloc[-1] > ma else -1
            logger.info(f"[LEADER] Regime macro: BTC {'ACIMA' if reg > 0 else 'ABAIXO'} da MA{cfg.REGIME_MA_DAYS}D")
            return reg
        except Exception as e:
            logger.warning(f"[LEADER] Falha ao obter regime — mantendo ultimo ({self._regime}): {e}")
            return self._regime   # transitorio: nao paralisa (evita halt silencioso de todas as moedas)

    def get_regime(self) -> int:
        """Regime macro atual: +1 bull, -1 bear, 0 desconhecido (fail-open)."""
        return self._regime

    def _compute_bias(self, market: MarketData) -> dict:
        """Calcula o vies de um lider a partir dos indicadores existentes."""
        df = market.get_klines(cfg.LEADER_TIMEFRAME, SIGNAL_WARMUP_BARS, closed_only=True)
        ind = market.calc_indicators(df)

        ema9 = ind["ema9"].iloc[-1]
        ema21 = ind["ema21"].iloc[-1]
        ema50 = ind["ema50"].iloc[-1]
        macd_hist = ind["macd_hist"].iloc[-1]
        adx = ind["adx"].iloc[-1]
        rsi = ind["rsi"].iloc[-1]

        # NaN em barras iniciais -> neutro (default seguro)
        if any(pd.isna(v) for v in (ema9, ema21, ema50, macd_hist, adx)):
            bias = NEUTRAL
        else:
            trend_up = ema9 > ema21 > ema50
            trend_down = ema9 < ema21 < ema50
            strong = adx >= cfg.ADX_RANGING_THRESHOLD   # abaixo = lateral -> neutro
            if trend_up and macd_hist > 0 and strong:
                bias = BULLISH
            elif trend_down and macd_hist < 0 and strong:
                bias = BEARISH
            else:
                bias = NEUTRAL

        detail = {
            "trend": ("up" if ema9 > ema21 > ema50 else "down" if ema9 < ema21 < ema50 else "flat"),
            "macd_hist": float(macd_hist) if not pd.isna(macd_hist) else 0.0,
            "adx": float(adx) if not pd.isna(adx) else 0.0,
            "rsi": float(rsi) if not pd.isna(rsi) else 50.0,
        }
        return {"bias": bias, "detail": detail}

    def _leaders_for(self, symbol: str) -> list[tuple[str, str]]:
        """Lista de (label, symbol) dos lideres aplicaveis a um simbolo.
        BTC sempre; ETH extra so p/ simbolos do ecossistema ETH. O proprio
        lider nao e filtrado contra si mesmo."""
        sym = symbol.upper()
        leaders = []
        if sym != cfg.LEADER_BTC_SYMBOL:
            leaders.append(("BTC", cfg.LEADER_BTC_SYMBOL))
        if sym in cfg.LEADER_ETH_SYMBOLS and sym != cfg.LEADER_ETH_SYMBOL:
            leaders.append(("ETH", cfg.LEADER_ETH_SYMBOL))
        return leaders

    def get_bias(self, symbol: str) -> dict:
        """Gate direcional para um simbolo. Retorna {allow_long, allow_short, reason}.
        Semantica: LONG so se nenhum lider bearish; SHORT so se nenhum lider bullish.
        Lider ausente (falha de coleta) = fail-open (nao bloqueia)."""
        if not cfg.LEADER_FILTER_ENABLED:
            return {"allow_long": True, "allow_short": True, "reason": ""}

        allow_long = True
        allow_short = True
        reasons = []
        for label, lsym in self._leaders_for(symbol):
            entry = self._cache.get(lsym)
            if entry is None:
                reasons.append(f"{label}=indisponivel(fail-open)")
                continue
            bias = entry["bias"]
            if bias == BULLISH:
                allow_short = False
                reasons.append(f"{label}=bullish")
            elif bias == BEARISH:
                allow_long = False
                reasons.append(f"{label}=bearish")
            else:  # NEUTRAL
                reasons.append(f"{label}=neutral")
                if cfg.LEADER_BLOCK_ON_NEUTRAL:
                    allow_long = False
                    allow_short = False

        return {
            "allow_long": allow_long,
            "allow_short": allow_short,
            "reason": "LIDER(" + ", ".join(reasons) + ")" if reasons else "",
        }

    def get_direction(self, symbol: str) -> int:
        """Direcao imposta pelos lideres para operar (modo mecanico sem LLM):
        +1 = LONG, -1 = SHORT, 0 = flat/sem sinal. BTC manda; ETH (extra p/
        ecossistema) precisa nao opor. Conflito BTC x ETH -> 0."""
        if not self._markets:
            return 0
        btc_entry = self._cache.get(cfg.LEADER_BTC_SYMBOL)
        btc = btc_entry["bias"] if btc_entry else NEUTRAL
        eco = symbol.upper() in cfg.LEADER_ETH_SYMBOLS
        eth = None
        if eco:
            eth_entry = self._cache.get(cfg.LEADER_ETH_SYMBOL)
            eth = eth_entry["bias"] if eth_entry else NEUTRAL
        if btc == BULLISH:
            return 0 if (eco and eth == BEARISH) else 1
        if btc == BEARISH:
            return 0 if (eco and eth == BULLISH) else -1
        return 0

    def format_for_llm(self, symbol: str = None) -> str:
        """Bloco de contexto do lider para o prompt. Vazio se desabilitado."""
        if not cfg.LEADER_FILTER_ENABLED or not self._markets:
            return ""

        lines = ["=== LIDER DE MERCADO (BTC/ETH) ==="]
        labels = {cfg.LEADER_BTC_SYMBOL: "BTC", cfg.LEADER_ETH_SYMBOL: "ETH"}
        for lsym in self._markets:
            label = labels.get(lsym, lsym)
            entry = self._cache.get(lsym)
            if entry is None:
                lines.append(f"{label}: dados indisponiveis (filtro fail-open)")
                continue
            d = entry["detail"]
            mom = "+" if d["macd_hist"] > 0 else "-" if d["macd_hist"] < 0 else "0"
            force = "FORTE" if d["adx"] >= cfg.ADX_RANGING_THRESHOLD else "LATERAL"
            lines.append(
                f"{label}: tendencia={entry['bias']} (EMAs={d['trend']}), "
                f"momentum={mom} (MACD_hist), ADX={d['adx']:.1f} [{force}]"
            )

        lines.append("Regra: entradas de alts contra a direcao do lider sao VETADAS "
                     "(bullish => so LONG; bearish => so SHORT).")

        if symbol:
            bias = self.get_bias(symbol)
            perm = []
            perm.append("LONG permitido" if bias["allow_long"] else "LONG vetado")
            perm.append("SHORT permitido" if bias["allow_short"] else "SHORT vetado")
            lines.append(f"Para {symbol}: {', '.join(perm)}. {bias['reason']}")

        return "\n".join(lines)
