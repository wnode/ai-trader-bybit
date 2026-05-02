"""
Persistencia de trades em SQLite.
Banco: data/trades.db
"""
import os
import sqlite3
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(DB_DIR, "trades.db")


SCHEMA_VERSION = 2


def _connect() -> sqlite3.Connection:
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Cria tabelas e aplica migracoes se necessario."""
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                symbol TEXT NOT NULL DEFAULT 'BTCUSDT',
                side TEXT NOT NULL,
                qty REAL NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                pnl REAL DEFAULT 0,
                close_type TEXT,
                confidence REAL,
                reason TEXT,
                order_id TEXT,
                llm_provider TEXT,
                llm_model TEXT
            )
        """)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        current_version = row[0] if row else 0

        # Migracao v1 -> v2: adicionar coluna symbol
        if current_version < 2:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)").fetchall()]
            if "symbol" not in cols:
                conn.execute("ALTER TABLE trades ADD COLUMN symbol TEXT NOT NULL DEFAULT 'BTCUSDT'")
                logger.info("[DB] Migracao v2: coluna 'symbol' adicionada (default BTCUSDT para trades existentes)")

        if not row:
            conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        elif current_version < SCHEMA_VERSION:
            conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        conn.commit()
    finally:
        conn.close()
    logger.info(f"[DB] Banco inicializado (v{SCHEMA_VERSION}): {DB_PATH}")


def record_open(symbol: str, side: str, qty: float, entry_price: float,
                stop_loss: float, take_profit: float,
                confidence: float, reason: str, order_id: str,
                llm_provider: str = "", llm_model: str = "") -> int:
    """Registra abertura de trade. Retorna o ID."""
    conn = _connect()
    try:
        cursor = conn.execute("""
            INSERT INTO trades (opened_at, symbol, side, qty, entry_price, stop_loss, take_profit,
                                confidence, reason, order_id, llm_provider, llm_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            symbol, side, qty, entry_price, stop_loss, take_profit,
            confidence, reason, order_id, llm_provider, llm_model,
        ))
        trade_id = cursor.lastrowid
        conn.commit()
    finally:
        conn.close()
    logger.info(f"[DB] Trade #{trade_id} aberto: {symbol} {side} {qty} @ ${entry_price:,.2f}")
    return trade_id


def record_close(order_id: str, exit_price: float, pnl: float, close_type: str) -> bool:
    """Registra fechamento de trade pelo order_id da abertura. Retorna True se atualizou."""
    conn = _connect()
    try:
        cursor = conn.execute("""
            UPDATE trades
            SET closed_at = ?, exit_price = ?, pnl = ?, close_type = ?
            WHERE order_id = ? AND closed_at IS NULL
        """, (
            datetime.now(timezone.utc).isoformat(),
            exit_price, pnl, close_type, order_id,
        ))
        conn.commit()
        updated = cursor.rowcount > 0
    finally:
        conn.close()

    if updated:
        logger.info(f"[DB] Trade fechado ({close_type}): PnL ${pnl:+,.2f}")
    else:
        logger.warning(f"[DB] Nenhum trade aberto encontrado para order_id={order_id}")
    return updated


def get_all_trades(symbol: str | None = None) -> list[dict]:
    """Retorna todos os trades fechados, do mais antigo ao mais recente.
    Se symbol for fornecido, filtra por simbolo."""
    conn = _connect()
    try:
        if symbol:
            rows = conn.execute("""
                SELECT * FROM trades WHERE closed_at IS NOT NULL AND symbol = ?
                ORDER BY opened_at ASC
            """, (symbol,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM trades WHERE closed_at IS NOT NULL
                ORDER BY opened_at ASC
            """).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def get_open_trade(symbol: str | None = None) -> dict | None:
    """Retorna trade aberto (sem closed_at), se houver.
    Se symbol for fornecido, filtra por simbolo."""
    conn = _connect()
    try:
        if symbol:
            rows = conn.execute("""
                SELECT * FROM trades WHERE closed_at IS NULL AND symbol = ?
                ORDER BY opened_at DESC
            """, (symbol,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT * FROM trades WHERE closed_at IS NULL
                ORDER BY opened_at DESC
            """).fetchall()
    finally:
        conn.close()

    if not rows:
        return None
    if len(rows) > 1:
        logger.warning(f"[DB] {len(rows)} trades abertos encontrados (symbol={symbol or 'todos'}) — usando o mais recente")
    return dict(rows[0])


def get_stats(symbol: str | None = None) -> dict:
    """Calcula estatisticas dos trades fechados.
    Se symbol for fornecido, filtra por simbolo."""
    trades = get_all_trades(symbol)
    if not trades:
        return {}

    pnls = [t["pnl"] for t in trades if t["pnl"] is not None]
    if not pnls:
        return {}

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_trades = len(pnls)
    total_pnl = sum(pnls)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    avg_win = sum(wins) / win_count if wins else 0
    avg_loss = sum(losses) / loss_count if losses else 0
    profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Max drawdown (sobre PnL acumulado)
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cumulative += p
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    # Contagem por tipo de fechamento
    tp_count = sum(1 for t in trades if t.get("close_type") == "TP")
    sl_count = sum(1 for t in trades if t.get("close_type") == "SL")
    close_count = sum(1 for t in trades if t.get("close_type") == "CLOSE")

    return {
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "tp_count": tp_count,
        "sl_count": sl_count,
        "close_count": close_count,
    }
