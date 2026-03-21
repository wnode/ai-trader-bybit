"""
Configuracao do AI Trader Bot.
Suporta multiplos providers: anthropic, google, openai
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
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
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

# LLM Provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google")

# Anthropic (Claude)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Google (Gemini)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")

# OpenAI (GPT)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Validacoes
VALID_PROVIDERS = ("google", "anthropic", "openai")
if LLM_PROVIDER.lower() not in VALID_PROVIDERS:
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

if not DRY_RUN and not BYBIT_API_KEY:
    print("[CONFIG] ERRO: BYBIT_API_KEY obrigatoria no modo LIVE")
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
}
if not _provider_keys.get(LLM_PROVIDER.lower()):
    print(f"[CONFIG] ERRO: API key para {LLM_PROVIDER} nao configurada")
    sys.exit(1)
