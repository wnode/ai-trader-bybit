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
- Risco:Retorno alto (TP parcial + runner): `PARTIAL_TP_ENABLED`, `TP1_RR_RATIO`, `TP1_SIZE_PCT`, `BREAKEVEN_AFTER_TP1`, `BREAKEVEN_OFFSET_PCT`
- Líder de mercado (filtro direcional BTC/ETH): `LEADER_FILTER_ENABLED`, `LEADER_ETH_SYMBOLS`, `LEADER_TIMEFRAME`, `LEADER_BLOCK_ON_NEUTRAL`, `LEADER_BTC_SYMBOL`, `LEADER_ETH_SYMBOL`
- Modo de decisão: `USE_LLM` (false = mecânico, sem custo de API), `LEADER_CLOSE_ON_FLIP`
- Economia de custo da LLM: `LLM_INTERVAL_MINUTES` (intervalo mín. entre chamadas por símbolo), `LLM_SKIP_WHEN_IN_POSITION` (não chama a LLM com posição aberta)
- Híbrido (LLM-como-filtro): `LLM_AS_FILTER`, `HYBRID_REGIME_FILTER`, `REGIME_MA_DAYS`, `FNG_GREED_MAX`, `FNG_FEAR_MIN`

## Convenções

- Código e comentários em português
- Logs vão para console + `logs/bot.log`
- Todos os módulos usam `import config as cfg` (não `from config import ...`) para que `--live` funcione
- TP/SL são definidos pela LLM e enviados à Bybit, que executa automaticamente
- Com `PARTIAL_TP_ENABLED`: TP runner em `MIN_RR_RATIO:1`. Abre fecha `TP1_SIZE_PCT` em `TP1_RR_RATIO:1` (reduceOnly Limit) e deixa o resto correr (reduceOnly Limit). SL em modo Full cobre a posição e encolhe junto. `manage_open_position` move o SL para breakeven quando o TP1 é atingido. `check_closed_by_exchange_for_order` soma os PnL dos fechamentos parciais
- Position sizing: `risk_amount / sl_distance` (sem multiplicar por leverage). `min_qty` e `qty_step` por símbolo via `get_instruments_info`
- Uma posição por vez **por símbolo** (verifica antes de abrir). Múltiplos símbolos podem ter posições simultâneas
- Entry/exit prices reais são obtidos via `get_executions` da API (não o sugerido pela LLM)
- `active_trade` é restaurado do DB ao reiniciar o bot (por símbolo)
- Loop processa símbolos sequencialmente em cada ciclo. Sentimento (Fear & Greed + xAI search) compartilhado entre todos
- Economia da LLM (`USE_LLM=true`): no loop, **depois** de `check_closed_by_exchange`/`manage_open_position` (que sempre rodam), um gate pula a análise (`continue`) se `LLM_SKIP_WHEN_IN_POSITION` e há `active_trade`, ou se `LLM_INTERVAL_MINUTES` ainda não passou desde `trader['last_llm_ts']`. Gestão de posição (breakeven/reconciliação) nunca é pulada — só a chamada paga da LLM
- Modo **híbrido** (`USE_LLM=true` + `LLM_AS_FILTER=true`): flat → `build_mechanical_setup` (líder `get_direction` → `get_regime` → F&G `sentiment.get_fear_greed` → `_pullback_ok`) detecta o setup **sem LLM**; só se houver setup chama `llm_analyze` (mesmo prompt+veto do modo LLM puro, extraído em helper). Com posição aberta → `build_mechanical_decision` (HOLD/CLOSE mecânico, sem LLM). Regime via `LeaderSignal._compute_regime` (BTC diário vs MA`REGIME_MA_DAYS`) em `refresh()`. Corta o custo de N símbolos: LLM só nos poucos setups/semana
- Com `USE_LLM=false` (modo mecânico, sem custo de API): `main.build_mechanical_decision` decide pela direção do líder BTC/ETH (`leader.get_direction`) — flat + flip do líder → LONG/SHORT; posição aberta → HOLD, ou CLOSE se `LEADER_CLOSE_ON_FLIP` e o líder inverteu. Pula `analyst.analyze`/sentiment/stream/monitor xAI. `LeaderSignal` é construído também quando `not USE_LLM`. Estado do flip em `trader['last_dir']`. Config vencedora do backtest: `TIMEFRAME=240` (4h) + `PARTIAL_TP_ENABLED=false` (TP único 3:1). `backtest.py` valida a estratégia mecânica (determinístico, sem LLM): `python backtest.py <dias> <timeframe>`
- Com `LEADER_FILTER_ENABLED`: `leader.py` (`LeaderSignal`) tem `MarketData` próprios de BTC/ETH (fora de `SYMBOLS`), calcula viés bullish/bearish/neutral por líder (EMAs alinhadas + MACD_hist + ADX) via `refresh()` 1x/ciclo. `format_for_llm(sym)` injeta contexto no prompt; `get_bias(sym)` é o gate. O veto é aplicado em `main.py` **entre `analyst.analyze` e `executor.execute`**: LONG/SHORT contra o líder viram HOLD. BTC filtra todas; ETH só as de `LEADER_ETH_SYMBOLS`. CLOSE/HOLD nunca vetados. Fail-open: falha de coleta do líder não bloqueia trades
- Histórico de decisões da LLM mantido por símbolo (`analyst.trade_history[symbol]`)
