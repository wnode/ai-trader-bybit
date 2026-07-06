"""
Configuracao do AI Trader Bot.
Suporta multiplos providers: anthropic, google, openai, xai
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()


def _get_int(key: str, default: str) -> int:
    val = os.getenv(key, default).strip()
    try:
        return int(val)
    except ValueError:
        print(f"[CONFIG] ERRO: {key} deve ser inteiro (valor: '{val}')")
        sys.exit(1)


def _get_float(key: str, default: str) -> float:
    val = os.getenv(key, default).strip()
    try:
        return float(val)
    except ValueError:
        print(f"[CONFIG] ERRO: {key} deve ser numerico (valor: '{val}')")
        sys.exit(1)


def _get_bool(key: str, default: str) -> bool:
    val = os.getenv(key, default).strip().lower()
    if val not in ("true", "false"):
        print(f"[CONFIG] ERRO: {key} deve ser 'true' ou 'false' (valor: '{val}')")
        sys.exit(1)
    return val == "true"


# Bybit
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
USE_TESTNET = _get_bool("USE_TESTNET", "true")
DRY_RUN = _get_bool("DRY_RUN", "true")

# Trading
# SYMBOLS: lista separada por virgula (ex: BTCUSDT,ETHUSDT,SOLUSDT)
# SYMBOL (legado): se SYMBOLS nao definido, usa SYMBOL como unico simbolo
_symbols_env = os.getenv("SYMBOLS", "").strip()
if _symbols_env:
    SYMBOLS = [s.strip().upper() for s in _symbols_env.split(",") if s.strip()]
else:
    SYMBOLS = [os.getenv("SYMBOL", "BTCUSDT").strip().upper()]
SYMBOL = SYMBOLS[0]  # Mantem compat com codigo que ainda le cfg.SYMBOL direto

for _s in SYMBOLS:
    if not _s.endswith("USDT"):
        print(f"[CONFIG] ERRO: simbolo deve terminar em USDT (valor: '{_s}')")
        sys.exit(1)

TIMEFRAME = os.getenv("TIMEFRAME", "15")
VALID_TIMEFRAMES = ("1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M")
if TIMEFRAME not in VALID_TIMEFRAMES:
    print(f"[CONFIG] ERRO: TIMEFRAME deve ser um de {VALID_TIMEFRAMES} (valor: '{TIMEFRAME}')")
    sys.exit(1)
CHECK_INTERVAL = _get_int("CHECK_INTERVAL", "300")
LEVERAGE = _get_int("LEVERAGE", "25")
RISK_PER_TRADE = _get_float("RISK_PER_TRADE", "0.0001")
KLINES_TO_SEND = _get_int("KLINES_TO_SEND", "96")
DAILY_KLINES = _get_int("DAILY_KLINES", "30")

# Indicadores tecnicos
RSI_PERIOD = _get_int("RSI_PERIOD", "14")
EMA_FAST = _get_int("EMA_FAST", "9")
EMA_MID = _get_int("EMA_MID", "21")
EMA_SLOW = _get_int("EMA_SLOW", "50")
BB_PERIOD = _get_int("BB_PERIOD", "20")
BB_STD = _get_float("BB_STD", "2.0")
ADX_PERIOD = _get_int("ADX_PERIOD", "14")
ATR_PERIOD = _get_int("ATR_PERIOD", "14")
MACD_FAST = _get_int("MACD_FAST", "12")
MACD_SLOW = _get_int("MACD_SLOW", "26")
MACD_SIGNAL = _get_int("MACD_SIGNAL", "9")
VOL_AVG_PERIOD = _get_int("VOL_AVG_PERIOD", "20")

# Regras de trading
SL_MIN_PCT = _get_float("SL_MIN_PCT", "0.3")
SL_MAX_PCT = _get_float("SL_MAX_PCT", "1.0")
MIN_RR_RATIO = _get_float("MIN_RR_RATIO", "1.0")
MIN_CONFIDENCE = _get_float("MIN_CONFIDENCE", "0.7")
ADX_RANGING_THRESHOLD = _get_float("ADX_RANGING_THRESHOLD", "15")

# Estrategia de Risco:Retorno alto — TP parcial + runner com breakeven
# Quando ligado: fecha TP1_SIZE_PCT da posicao em TP1_RR_RATIO:1 (trava lucro)
# e deixa o restante correr ate o runner em MIN_RR_RATIO:1. Apos o TP1,
# o SL e movido para breakeven (entry + BREAKEVEN_OFFSET_PCT% p/ cobrir fees).
PARTIAL_TP_ENABLED = _get_bool("PARTIAL_TP_ENABLED", "false")
TP1_RR_RATIO = _get_float("TP1_RR_RATIO", "1.5")
TP1_SIZE_PCT = _get_float("TP1_SIZE_PCT", "0.5")
BREAKEVEN_AFTER_TP1 = _get_bool("BREAKEVEN_AFTER_TP1", "true")
BREAKEVEN_OFFSET_PCT = _get_float("BREAKEVEN_OFFSET_PCT", "0.05")

if PARTIAL_TP_ENABLED:
    if not (0.0 < TP1_SIZE_PCT < 1.0):
        print(f"[CONFIG] ERRO: TP1_SIZE_PCT deve estar entre 0 e 1 (valor: {TP1_SIZE_PCT})")
        sys.exit(1)
    if TP1_RR_RATIO >= MIN_RR_RATIO:
        print(f"[CONFIG] ERRO: TP1_RR_RATIO ({TP1_RR_RATIO}) deve ser menor que "
              f"MIN_RR_RATIO ({MIN_RR_RATIO}) — TP1 e o alvo parcial, MIN_RR_RATIO e o runner")
        sys.exit(1)

# Sentimento de mercado
USE_SENTIMENT = _get_bool("USE_SENTIMENT", "false")
FNG_CACHE_MINUTES = _get_int("FNG_CACHE_MINUTES", "60")
XAI_SEARCH = _get_bool("XAI_SEARCH", "false")

# Lider de mercado (BTC/ETH) — filtro direcional das altcoins
# BTC e o gate de todas as alts; ETH e reforco p/ alts do ecossistema ETH.
# So permite LONG se lider bullish, SHORT se bearish. CLOSE nunca e vetado.
# Fail-open: falha de coleta do lider NAO bloqueia trades.
LEADER_FILTER_ENABLED = _get_bool("LEADER_FILTER_ENABLED", "false")
_leader_eth_env = os.getenv("LEADER_ETH_SYMBOLS", "").strip()
LEADER_ETH_SYMBOLS = [s.strip().upper() for s in _leader_eth_env.split(",") if s.strip()]
LEADER_TIMEFRAME = os.getenv("LEADER_TIMEFRAME", TIMEFRAME).strip()
LEADER_BLOCK_ON_NEUTRAL = _get_bool("LEADER_BLOCK_ON_NEUTRAL", "false")
LEADER_BTC_SYMBOL = os.getenv("LEADER_BTC_SYMBOL", "BTCUSDT").strip().upper()
LEADER_ETH_SYMBOL = os.getenv("LEADER_ETH_SYMBOL", "ETHUSDT").strip().upper()
# Modo mecanico (sem LLM): fecha a posicao quando o lider inverte de direcao
LEADER_CLOSE_ON_FLIP = _get_bool("LEADER_CLOSE_ON_FLIP", "true")

# USE_LLM: se false, o bot NAO chama a LLM (custo zero). A decisao passa a ser
# mecanica, guiada pela direcao do lider BTC/ETH (flat+bullish->LONG, bearish->SHORT,
# fecha na virada). O codigo da LLM fica intacto para religar (USE_LLM=true).
USE_LLM = _get_bool("USE_LLM", "true")

# Economia de custo da LLM (so afeta USE_LLM=true):
# - LLM_INTERVAL_MINUTES: intervalo minimo entre chamadas da LLM por simbolo.
#   0 = chama todo ciclo (padrao antigo). Ex: 15 = 1x por vela de 15min (sem desperdicio).
# - LLM_SKIP_WHEN_IN_POSITION: nao chama a LLM enquanto ha posicao aberta — as saidas
#   ja sao geridas por TP/SL/breakeven na Bybit. Corta o maior custo (ciclos de 60s).
LLM_INTERVAL_MINUTES = _get_int("LLM_INTERVAL_MINUTES", "0")
LLM_SKIP_WHEN_IN_POSITION = _get_bool("LLM_SKIP_WHEN_IN_POSITION", "false")

# Hibrido: LLM como FILTRO dos setups mecanicos (custo baixo com muitos simbolos).
# So vale com USE_LLM=true. A camada mecanica (direcao do lider BTC/ETH + pullback
# no alt + Fear&Greed + regime macro) detecta os setups; a LLM so e chamada NESSES
# setups para aprovar/vetar (traz o julgamento p/ elevar o WR). Sem setup = sem
# chamada paga. Saidas ficam por conta de TP/SL (e CLOSE mecanico na virada do lider).
LLM_AS_FILTER = _get_bool("LLM_AS_FILTER", "false")
# Fear & Greed (contrarian): bloqueia LONG em ganancia extrema, SHORT em medo extremo
FNG_GREED_MAX = _get_float("FNG_GREED_MAX", "80")
FNG_FEAR_MIN = _get_float("FNG_FEAR_MIN", "20")
# Regime macro: so LONG com BTC acima da media de N dias, so SHORT abaixo
HYBRID_REGIME_FILTER = _get_bool("HYBRID_REGIME_FILTER", "true")
REGIME_MA_DAYS = _get_int("REGIME_MA_DAYS", "200")

# Verificacao PERIODICA de sentimento do LIDER (BTC/ETH) via Grok search (xAI).
# A cada N horas o Grok busca noticias/sentimento de BTC/ETH e injeta no prompt dos
# setups (as alts seguem o lider, entao nao precisa buscar sentimento de cada alt).
# 0 = desligado. So funciona com LLM_PROVIDER=xai e XAI_SEARCH=true.
LEADER_SENTIMENT_HOURS = _get_float("LEADER_SENTIMENT_HOURS", "0")

# O lider e usado quando o filtro esta ligado, no modo sem-LLM, OU no hibrido.
if LEADER_FILTER_ENABLED or not USE_LLM or LLM_AS_FILTER:
    if LEADER_TIMEFRAME not in VALID_TIMEFRAMES:
        print(f"[CONFIG] ERRO: LEADER_TIMEFRAME deve ser um de {VALID_TIMEFRAMES} (valor: '{LEADER_TIMEFRAME}')")
        sys.exit(1)
    for _ls in LEADER_ETH_SYMBOLS:
        if not _ls.endswith("USDT"):
            print(f"[CONFIG] ERRO: LEADER_ETH_SYMBOLS deve terminar em USDT (valor: '{_ls}')")
            sys.exit(1)

# Validacao de faixas e combos de flags (evita halt silencioso / TP degenerado)
if MIN_RR_RATIO <= 0:
    print(f"[CONFIG] ERRO: MIN_RR_RATIO deve ser > 0 (valor: {MIN_RR_RATIO})")
    sys.exit(1)
if not (0 <= FNG_FEAR_MIN < FNG_GREED_MAX <= 100):
    print(f"[CONFIG] ERRO: exija 0 <= FNG_FEAR_MIN < FNG_GREED_MAX <= 100 "
          f"(FEAR_MIN={FNG_FEAR_MIN}, GREED_MAX={FNG_GREED_MAX})")
    sys.exit(1)
if REGIME_MA_DAYS < 1:
    print(f"[CONFIG] ERRO: REGIME_MA_DAYS deve ser >= 1 (valor: {REGIME_MA_DAYS})")
    sys.exit(1)
if LLM_INTERVAL_MINUTES < 0:
    print(f"[CONFIG] ERRO: LLM_INTERVAL_MINUTES deve ser >= 0 (valor: {LLM_INTERVAL_MINUTES})")
    sys.exit(1)
if LLM_AS_FILTER and not USE_LLM:
    print("[CONFIG] ERRO: LLM_AS_FILTER=true exige USE_LLM=true")
    sys.exit(1)
if HYBRID_REGIME_FILTER and USE_LLM and not LLM_AS_FILTER:
    print("[CONFIG] AVISO: HYBRID_REGIME_FILTER nao tem efeito no modo LLM puro (so no hibrido)")

# LLM Provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google").strip().lower()

# Anthropic (Claude)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Google (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

# OpenAI (GPT)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# xAI (Grok)
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
XAI_MODEL = os.getenv("XAI_MODEL", "grok-3-mini")
XAI_SEARCH_MODEL = os.getenv("XAI_SEARCH_MODEL", "grok-4-1-fast-non-reasoning")
XAI_SEARCH_CACHE_ITERATIONS = _get_int("XAI_SEARCH_CACHE_ITERATIONS", "6")

# Sentimento
SENTIMENT_MONITOR_INTERVAL = _get_int("SENTIMENT_MONITOR_INTERVAL", "120")

# X API v2 Stream (tempo real)
X_STREAM_ENABLED = _get_bool("X_STREAM_ENABLED", "false")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "")
X_STREAM_KEYWORDS = os.getenv("X_STREAM_KEYWORDS", "bitcoin,btc,crypto,#bitcoin,#btc")
X_STREAM_URGENCY_THRESHOLD = _get_int("X_STREAM_URGENCY_THRESHOLD", "7")

# Validacoes
VALID_PROVIDERS = ("google", "anthropic", "openai", "xai")
if LLM_PROVIDER not in VALID_PROVIDERS:
    print(f"[CONFIG] ERRO: LLM_PROVIDER deve ser um de {VALID_PROVIDERS} (valor: '{LLM_PROVIDER}')")
    sys.exit(1)

if LEVERAGE < 1:
    print(f"[CONFIG] ERRO: LEVERAGE deve ser >= 1 (valor: {LEVERAGE})")
    sys.exit(1)

if not (0 < RISK_PER_TRADE <= 0.1):
    print(f"[CONFIG] ERRO: RISK_PER_TRADE deve estar entre 0 e 0.1 (valor: {RISK_PER_TRADE})")
    sys.exit(1)

if CHECK_INTERVAL < 10:
    print(f"[CONFIG] ERRO: CHECK_INTERVAL deve ser >= 10 (valor: {CHECK_INTERVAL})")
    sys.exit(1)

if not DRY_RUN and (not BYBIT_API_KEY or not BYBIT_API_SECRET):
    print("[CONFIG] ERRO: BYBIT_API_KEY e BYBIT_API_SECRET obrigatorias no modo LIVE")
    sys.exit(1)

if SL_MIN_PCT >= SL_MAX_PCT:
    print(f"[CONFIG] ERRO: SL_MIN_PCT ({SL_MIN_PCT}) deve ser menor que SL_MAX_PCT ({SL_MAX_PCT})")
    sys.exit(1)

if MACD_FAST >= MACD_SLOW:
    print(f"[CONFIG] ERRO: MACD_FAST ({MACD_FAST}) deve ser menor que MACD_SLOW ({MACD_SLOW})")
    sys.exit(1)

if not (EMA_FAST < EMA_MID < EMA_SLOW):
    print(f"[CONFIG] ERRO: EMA periods devem ser EMA_FAST ({EMA_FAST}) < EMA_MID ({EMA_MID}) < EMA_SLOW ({EMA_SLOW})")
    sys.exit(1)

_provider_keys = {
    "google": GOOGLE_API_KEY,
    "anthropic": ANTHROPIC_API_KEY,
    "openai": OPENAI_API_KEY,
    "xai": XAI_API_KEY,
}
if not _provider_keys.get(LLM_PROVIDER):
    print(f"[CONFIG] ERRO: API key para {LLM_PROVIDER} nao configurada")
    sys.exit(1)
