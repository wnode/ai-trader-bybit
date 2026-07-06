"""
Backtest da ESTRATEGIA PURA DO LIDER (sem LLM) — 100% deterministico.

Testa a ideia de operar as alts seguindo a direcao do BTC/ETH (beta), sem a
LLM: flat + lider bullish -> LONG; bearish -> SHORT. Vies do lider calculado
igual ao leader.py (EMAs alinhadas + MACD_hist + ADX). Saidas pelas mecanicas
ja implementadas (TP parcial + runner + breakeven, ou TP unico) com taxas.

Como e totalmente mecanico (sem LLM no meio), NAO ha lookahead nem custo de
API — o resultado e reproduzivel e confiavel dentro dos limites do modelo de
execucao (assume SL antes do TP quando a barra abrange ambos; fills a mercado
na entrada/SL/virada e maker nos TPs limit).

Variantes comparadas: entrada {na virada do lider / continua} x saida {TP
unico / TP parcial} x {fecha na virada do lider / nao}. Resultado em R.

Uso: python backtest.py [dias]     (default 180)
"""
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import config as cfg
from market_data import MarketData

BULLISH, BEARISH, NEUTRAL = "bullish", "bearish", "neutral"

# Taxas Bybit perp linear (fracao do notional)
TAKER = 0.00055   # market (entrada, SL)
MAKER = 0.00020   # limit (TPs)

MS_PER_DAY = 86_400_000


# --------------------------------------------------------------------------
# Coleta historica paginada
# --------------------------------------------------------------------------
def fetch_history(md: MarketData, interval: str, days: int) -> pd.DataFrame:
    """Busca `days` dias de klines paginando pelo cursor `end` (mais recente -> antigo)."""
    now_ms = int(time.time() * 1000)
    target_start = now_ms - days * MS_PER_DAY
    end = now_ms
    raw = {}
    while True:
        resp = md._api_call("get_kline", category="linear", symbol=md.symbol,
                            interval=interval, end=end, limit=1000)
        lst = resp.get("result", {}).get("list", []) if resp else []
        if not lst:
            break
        for k in lst:
            ts = int(k[0])
            raw[ts] = k
        oldest = int(lst[-1][0])   # lista vem do mais novo p/ o mais antigo
        if oldest <= target_start or len(lst) < 1000:
            break
        end = oldest - 1

    rows = []
    for ts in sorted(raw):
        if ts < target_start:
            continue
        _, o, h, l, c, vol, _turn = raw[ts]
        rows.append({"ts_ms": ts,
                     "timestamp": datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
                     "open": float(o), "high": float(h), "low": float(l),
                     "close": float(c), "volume": float(vol)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Indicadores + vies do lider (mesma regra do leader.py)
# --------------------------------------------------------------------------
def indicators_arrays(md: MarketData, df: pd.DataFrame) -> dict:
    """Calcula indicadores (reusa calc_indicators do bot) e devolve arrays numpy."""
    ind = md.calc_indicators(df)
    return {k: v.to_numpy() for k, v in ind.items()}


def bias_at(a: dict, i: int) -> str:
    """Vies bullish/bearish/neutral na barra i — identico ao leader._compute_bias."""
    e9, e21, e50 = a["ema9"][i], a["ema21"][i], a["ema50"][i]
    mh, adx = a["macd_hist"][i], a["adx"][i]
    if any(np.isnan(v) for v in (e9, e21, e50, mh, adx)):
        return NEUTRAL
    strong = adx >= cfg.ADX_RANGING_THRESHOLD
    if e9 > e21 > e50 and mh > 0 and strong:
        return BULLISH
    if e9 < e21 < e50 and mh < 0 and strong:
        return BEARISH
    return NEUTRAL


# --------------------------------------------------------------------------
# Direcao pura do lider (a estrategia sem LLM)
# --------------------------------------------------------------------------
def symbol_direction(leader_bias: dict, sym: str, ts: int) -> int:
    """Direcao imposta pelos lideres na barra ts: +1 LONG, -1 SHORT, 0 flat.
    BTC manda; ETH (extra p/ ecossistema) precisa nao opor. Conflito -> 0."""
    btc = leader_bias.get(cfg.LEADER_BTC_SYMBOL, {}).get(ts, NEUTRAL)
    eco = sym.upper() in cfg.LEADER_ETH_SYMBOLS
    eth = leader_bias.get(cfg.LEADER_ETH_SYMBOL, {}).get(ts, NEUTRAL) if eco else None
    if btc == BULLISH:
        if eco and eth == BEARISH:
            return 0   # conflito
        return 1
    if btc == BEARISH:
        if eco and eth == BULLISH:
            return 0   # conflito
        return -1
    return 0


# --------------------------------------------------------------------------
# Simulacao de uma posicao (barra a barra)
# --------------------------------------------------------------------------
def simulate_trade(df, a, entry_idx, action, partial, dir_arr=None, close_on_flip=False):
    """Simula uma posicao a partir de entry_idx. Retorna (exit_idx, r_net, close_type).
    R = risco (distancia entry->SL na posicao cheia). Assume SL antes do TP quando
    a barra abrange ambos (conservador). Entrada no OPEN da barra entry_idx.
    Se close_on_flip, fecha a mercado quando o lider vira contra a posicao."""
    n = len(df)
    entry = df["open"].iloc[entry_idx]
    atr = a["atr"][entry_idx - 1]   # ATR conhecido no fechamento da barra de sinal
    if np.isnan(atr) or entry <= 0:
        return entry_idx, 0.0, "SKIP"

    direction = 1 if action == "LONG" else -1
    sl_frac = 1.5 * atr / entry
    sl_frac = max(cfg.SL_MIN_PCT / 100, min(cfg.SL_MAX_PCT / 100, sl_frac))
    sl_dist = entry * sl_frac
    sl0 = entry - direction * sl_dist

    tp1 = entry + direction * sl_dist * cfg.TP1_RR_RATIO
    tp2 = entry + direction * sl_dist * cfg.MIN_RR_RATIO
    tp_single = tp2
    f1 = cfg.TP1_SIZE_PCT          # fracao no TP1
    f2 = 1 - f1                    # runner

    def r_of(price, frac):
        return direction * (price - entry) / sl_dist * frac

    # Taxas (em R): notional total = 1/sl_frac ; por fracao f = f/sl_frac
    def fee(rate, frac):
        return rate * frac / sl_frac

    fees = fee(TAKER, 1.0)   # entrada market (posicao cheia)

    sl = sl0
    tp1_done = False
    for j in range(entry_idx, n):
        hi, lo = df["high"].iloc[j], df["low"].iloc[j]
        hit_sl = lo <= sl if direction == 1 else hi >= sl

        # Fecha a mercado se o lider virou contra a posicao (tese: segue o BTC/ETH)
        if close_on_flip and dir_arr is not None and j > entry_idx:
            d = dir_arr[j]
            if (direction == 1 and d < 0) or (direction == -1 and d > 0):
                price = df["close"].iloc[j]
                if partial and tp1_done:
                    fees += fee(TAKER, f2)
                    return j, r_of(tp1, f1) + r_of(price, f2) - fees, "FLIP"
                fees += fee(TAKER, 1.0)
                return j, r_of(price, 1.0) - fees, "FLIP"

        if not partial:
            hit_tp = hi >= tp_single if direction == 1 else lo <= tp_single
            if hit_sl:      # conservador: SL antes do TP
                fees += fee(TAKER, 1.0)
                return j, r_of(sl, 1.0) - fees, "SL"
            if hit_tp:
                fees += fee(MAKER, 1.0)
                return j, r_of(tp_single, 1.0) - fees, "TP"
            continue

        # --- modo parcial ---
        if not tp1_done:
            hit_tp1 = hi >= tp1 if direction == 1 else lo <= tp1
            if hit_sl:      # perde as duas metades
                fees += fee(TAKER, 1.0)
                return j, r_of(sl, 1.0) - fees, "SL"
            if hit_tp1:
                tp1_done = True
                fees += fee(MAKER, f1)             # TP1 = limit (maker)
                sl = entry + direction * entry * (cfg.BREAKEVEN_OFFSET_PCT / 100)  # breakeven
                # segue na mesma barra pode bater tp2? conservador: proxima barra
                continue
        else:
            hit_tp2 = hi >= tp2 if direction == 1 else lo <= tp2
            hit_be = lo <= sl if direction == 1 else hi >= sl
            if hit_be:       # runner sai no breakeven (SL market)
                fees += fee(TAKER, f2)
                return j, r_of(tp1, f1) + r_of(sl, f2) - fees, "BE"
            if hit_tp2:      # runner atinge o alvo cheio
                fees += fee(MAKER, f2)
                return j, r_of(tp1, f1) + r_of(tp2, f2) - fees, "TP"
            continue

    # Nao fechou ate o fim dos dados — marca a mercado no ultimo close
    last = df["close"].iloc[-1]
    if partial and tp1_done:
        return n - 1, r_of(tp1, f1) + r_of(last, f2) - fees, "EOD"
    return n - 1, r_of(last, 1.0) - fees, "EOD"


# --------------------------------------------------------------------------
# Passe de backtest (uma variante)
# --------------------------------------------------------------------------
def run_leader_variant(df, a, dir_arr, partial, entry_mode, close_on_flip):
    """Estrategia PURA do lider (sem LLM). entry_mode: 'flip' (so na virada do
    lider) ou 'cont' (sempre que flat e o lider tem direcao). Retorna trades."""
    n = len(df)
    trades = []
    i = 1
    while i < n and np.isnan(a["ema50"][i]):
        i += 1
    while i < n - 1:
        d = dir_arr[i]
        if entry_mode == "flip":
            enter = d != 0 and d != dir_arr[i - 1]
        else:  # continuo
            enter = d != 0
        if enter:
            action = "LONG" if d > 0 else "SHORT"
            exit_idx, r_net, ctype = simulate_trade(df, a, i + 1, action, partial,
                                                    dir_arr=dir_arr, close_on_flip=close_on_flip)
            if ctype != "SKIP":
                trades.append({"action": action, "r": r_net, "type": ctype})
            i = max(exit_idx + 1, i + 1)
        else:
            i += 1
    return trades


def stats(trades, days):
    if not trades:
        return {"n": 0}
    rs = [t["r"] for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    total = sum(rs)
    # equity/drawdown em R
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    gross_w = sum(wins)
    gross_l = -sum(losses)
    return {
        "n": len(trades),
        "per_wk": len(trades) / (days / 7),
        "win": len(wins) / len(trades) * 100,
        "exp": total / len(trades),
        "total": total,
        "pf": (gross_w / gross_l) if gross_l > 0 else float("inf"),
        "mdd": mdd,
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    tf = sys.argv[2] if len(sys.argv) > 2 else cfg.TIMEFRAME
    alts = cfg.SYMBOLS
    leaders = {cfg.LEADER_BTC_SYMBOL, cfg.LEADER_ETH_SYMBOL}

    print(f"=== BACKTEST — ESTRATEGIA PURA DO LIDER (sem LLM) — {days} dias, {tf}min ===")
    print(f"Alts: {alts} | Lideres: BTC={cfg.LEADER_BTC_SYMBOL} ETH={cfg.LEADER_ETH_SYMBOL} "
          f"(ETH extra p/ {cfg.LEADER_ETH_SYMBOLS})")
    print(f"Regra: flat + lider bullish -> LONG; bearish -> SHORT. Saidas: "
          f"TP1={cfg.TP1_RR_RATIO}:1 ({cfg.TP1_SIZE_PCT:.0%}) + runner {cfg.MIN_RR_RATIO}:1 + breakeven.")
    print(f"Vies do lider: EMAs alinhadas + MACD_hist + ADX>={cfg.ADX_RANGING_THRESHOLD}. Taxas incluidas. R = risco/trade.\n")

    # Coleta lideres 1x -> vies por barra
    leader_bias = {}   # symbol -> {ts_ms: bias}
    md_cache = {}
    for lsym in leaders:
        md = MarketData(lsym)
        md_cache[lsym] = md
        dfl = fetch_history(md, tf, days)
        al = indicators_arrays(md, dfl)
        leader_bias[lsym] = {int(dfl["ts_ms"].iloc[k]): bias_at(al, k) for k in range(len(dfl))}
        from collections import Counter
        dist = Counter(leader_bias[lsym].values())
        print(f"  [{lsym}] {len(dfl)} barras ({dfl['timestamp'].iloc[0].date()} -> {dfl['timestamp'].iloc[-1].date()}) "
              f"| vies: bull={dist.get(BULLISH,0)} bear={dist.get(BEARISH,0)} neutro={dist.get(NEUTRAL,0)}")

    # Variantes: (nome, entry_mode, partial, close_on_flip)
    variants = [
        ("flip, TP unico, fecha-virada", "flip", False, True),
        ("flip, TP parcial, fecha-virada", "flip", True, True),
        ("continuo, TP parcial, fecha-virada", "cont", True, True),
        ("flip, TP parcial, SEM fecha-virada", "flip", True, False),
    ]
    agg = {v[0]: [] for v in variants}

    for sym in alts:
        md = md_cache.get(sym) or MarketData(sym)
        df = fetch_history(md, tf, days)
        if len(df) < 100:
            print(f"  [{sym}] poucas barras ({len(df)}) — pulando")
            continue
        a = indicators_arrays(md, df)
        a["volume"] = df["volume"].to_numpy()
        ts_arr = df["ts_ms"].to_numpy()
        dir_arr = np.array([symbol_direction(leader_bias, sym, int(t)) for t in ts_arr])
        print(f"\n[{sym}] {len(df)} barras ({df['timestamp'].iloc[0].date()} -> {df['timestamp'].iloc[-1].date()})")
        for name, emode, partial, cof in variants:
            trades = run_leader_variant(df, a, dir_arr, partial, emode, cof)
            agg[name].extend(trades)
            s = stats(trades, days)
            if s["n"]:
                print(f"  {name:36} n={s['n']:3}  win={s['win']:4.0f}%  "
                      f"exp={s['exp']:+.3f}R  PF={s['pf']:.2f}  DD={s['mdd']:.1f}R  total={s['total']:+.1f}R")
            else:
                print(f"  {name:36} sem trades")

    print("\n=== AGREGADO (todos os alts) ===")
    print(f"{'variante':38}{'trades':>7}{'/sem':>7}{'win%':>7}{'exp(R)':>8}{'PF':>6}{'DD(R)':>7}{'total(R)':>9}")
    for name, _, _, _ in variants:
        s = stats(agg[name], days)
        if not s["n"]:
            print(f"{name:38}{'sem trades':>36}")
            continue
        pf = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
        print(f"{name:38}{s['n']:>7}{s['per_wk']:>7.1f}{s['win']:>7.0f}{s['exp']:>8.3f}{pf:>6}{s['mdd']:>7.1f}{s['total']:>9.1f}")

    print("\nNota: exp = expectancy/trade em R (>0 lucrativo). PF>1 lucrativo. Entrada e saida"
          " 100% deterministicas (sem LLM). 'fecha-virada' = fecha quando o lider inverte.")


if __name__ == "__main__":
    main()
