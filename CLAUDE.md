# AI Trader Bybit

Bot de trading automatizado para múltiplos pares (BTC, ETH, SOL) em futuros perpétuos na Bybit, usando LLM para tomar decisões de LONG/SHORT/HOLD/CLOSE. Cada símbolo opera independente, com posição, histórico e gestão de risco próprios.

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
- `SYMBOLS` — lista de símbolos separados por vírgula (ex: `BTCUSDT,ETHUSDT,SOLUSDT`). Cada um opera independente
- `LLM_PROVIDER` — google, anthropic, openai ou xai
- `RISK_PER_TRADE` — fração do saldo arriscada por trade (ex: 0.0001 = 0.01%). Aplicado por símbolo
- `LEVERAGE` — alavancagem (default 25x). Não afeta o risco, só a margem usada
- `CHECK_INTERVAL` — segundos entre ciclos completos (todos os símbolos, default 300)
- Indicadores: `RSI_PERIOD`, `EMA_FAST/MID/SLOW`, `BB_PERIOD/STD`, `ADX_PERIOD`, `ATR_PERIOD`, `MACD_FAST/SLOW/SIGNAL`
- Regras de trading: `SL_MIN_PCT`, `SL_MAX_PCT`, `MIN_RR_RATIO`, `MIN_CONFIDENCE`, `ADX_RANGING_THRESHOLD`

## Convenções

- Código e comentários em português
- Logs vão para console + `logs/bot.log`
- Todos os módulos usam `import config as cfg` (não `from config import ...`) para que `--live` funcione
- TP/SL são definidos pela LLM e enviados à Bybit, que executa automaticamente
- Position sizing: `risk_amount / sl_distance` (sem multiplicar por leverage). `min_qty` e `qty_step` por símbolo via `get_instruments_info`
- Uma posição por vez **por símbolo** (verifica antes de abrir). Múltiplos símbolos podem ter posições simultâneas
- Entry/exit prices reais são obtidos via `get_executions` da API (não o sugerido pela LLM)
- `active_trade` é restaurado do DB ao reiniciar o bot (por símbolo)
- Loop processa símbolos sequencialmente em cada ciclo. Sentimento (Fear & Greed + xAI search) compartilhado entre todos
- Histórico de decisões da LLM mantido por símbolo (`analyst.trade_history[symbol]`)
