"""
Configuracao do AI Trader Bot.
Suporta multiplos providers: anthropic, google, openai
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Bybit
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"
DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"

# Trading
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("TIMEFRAME", "15")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "300"))
LEVERAGE = int(os.getenv("LEVERAGE", "25"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.0001"))
KLINES_TO_SEND = int(os.getenv("KLINES_TO_SEND", "96"))
DAILY_KLINES = int(os.getenv("DAILY_KLINES", "30"))

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
