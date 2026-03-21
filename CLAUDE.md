# AI Trader Bybit

Bot de trading automatizado para BTCUSDT (futuros perpétuos) na Bybit, usando LLM para tomar decisões de LONG/SHORT/HOLD/CLOSE.

## Como rodar

```bash
pip install -r requirements.txt
python main.py          # usa config do .env (DRY_RUN, USE_TESTNET)
python main.py --live   # força DRY_RUN=false
python main.py --once   # roda um ciclo e sai
python monitor.py       # status avulso (conta, posição, histórico)
```

## Estrutura

- `main.py` — loop principal: coleta dados → LLM analisa → executor opera
- `analyst.py` — integração com LLMs (Gemini, Claude, GPT). Prompt dinâmico com regras do `.env`
- `executor.py` — execução de ordens na Bybit. Orders do bot usam prefixo `aitbot-` no `orderLinkId`
- `market_data.py` — coleta klines, indicadores técnicos (EMA, RSI, MACD, ATR, Bollinger, ADX)
- `monitor.py` — exibe conta, posição aberta, histórico e estatísticas
- `db.py` — persistência de trades em SQLite (`data/trades.db`)
- `config.py` — carrega e valida variáveis do `.env`

## Configuração

Tudo via `.env` (ver `.env.example`). Variáveis principais:
- `DRY_RUN` / `USE_TESTNET` — modo de execução
- `LLM_PROVIDER` — google, anthropic ou openai
- `RISK_PER_TRADE` — fração do saldo arriscada por trade (ex: 0.0001 = 0.01%)
- `LEVERAGE` — alavancagem (default 25x). Não afeta o risco, só a margem usada
- `CHECK_INTERVAL` — segundos entre ciclos (default 300)
- Indicadores: `RSI_PERIOD`, `EMA_FAST/MID/SLOW`, `BB_PERIOD/STD`, `ADX_PERIOD`, `ATR_PERIOD`, `MACD_FAST/SLOW/SIGNAL`
- Regras de trading: `SL_MIN_PCT`, `SL_MAX_PCT`, `MIN_RR_RATIO`, `MIN_CONFIDENCE`, `ADX_RANGING_THRESHOLD`

## Convenções

- Código e comentários em português
- Logs vão para console + `logs/bot.log`
- Todos os módulos usam `import config as cfg` (não `from config import ...`) para que `--live` funcione
- TP/SL são definidos pela LLM e enviados à Bybit, que executa automaticamente
- Position sizing: `risk_amount / sl_distance` (sem multiplicar por leverage)
- Uma posição por vez (verifica antes de abrir)
- Entry/exit prices reais são obtidos via `get_executions` da API (não o sugerido pela LLM)
- `active_trade` é restaurado do DB ao reiniciar o bot
