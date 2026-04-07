"""
SQLite-backed executor for DIC-approved SQL actions.

The database is created fresh on each demo run (in demos/dic_db/db/demo.db)
and seeded with a `users` table so SELECT and UPDATE have real data to work on.
"""
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .db_action import SQLAction, SQLOp

DB_PATH = Path(__file__).parent / "db" / "demo.db"

_SEED_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT    NOT NULL,
    email TEXT    NOT NULL UNIQUE,
    role  TEXT    NOT NULL DEFAULT 'user'
);
INSERT OR IGNORE INTO users (name, email, role) VALUES
    ('Bob',     'bob@example.com',     'admin'),
    ('Carol',   'carol@example.com',   'user'),
    ('Dave',    'dave@example.com',    'user');
"""


class DBEngine:
    """
    Thin SQLite wrapper.  Only executes actions approved by DICGovernor.
    Provides row_count_fn for the governor to estimate blast radius.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ---------------------------------------------------------------- #
    #  Public API                                                        #
    # ---------------------------------------------------------------- #

    def execute(self, action: SQLAction) -> Optional[List[Dict[str, Any]]]:
        """
        Execute an approved action. Returns rows for SELECT; None otherwise.
        Raises on SQL error.
        """
        if action.op == SQLOp.DONE:
            return None

        with self._conn() as conn:
            if action.op == SQLOp.SELECT:
                return self._select(conn, action)
            elif action.op == SQLOp.INSERT:
                self._insert(conn, action)
            elif action.op == SQLOp.UPDATE:
                self._update(conn, action)
            elif action.op == SQLOp.DELETE_ROW:
                self._delete_row(conn, action)
            elif action.op == SQLOp.DROP_TABLE:
                self._drop_table(conn, action)
            return None

    def row_count(self, action: SQLAction) -> int:
        """
        Estimate how many rows would be affected — used by DICGovernor
        for risk assessment before execution.
        """
        if action.op in (SQLOp.SELECT, SQLOp.DONE, SQLOp.DROP_TABLE,
                         SQLOp.INSERT):
            return 0
        try:
            with self._conn() as conn:
                sql   = f"SELECT COUNT(*) FROM {action.table}"
                params: Tuple = ()
                if action.condition:
                    sql    = f"SELECT COUNT(*) FROM {action.table} WHERE {action.condition}"
                row = conn.execute(sql, params).fetchone()
                return row[0] if row else 0
        except Exception:
            return 0

    def table_exists(self, table: str) -> bool:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,)
            ).fetchone()
            return row is not None

    def list_tables(self) -> List[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            return [r[0] for r in rows]

    # ---------------------------------------------------------------- #
    #  Operations                                                        #
    # ---------------------------------------------------------------- #

    def _select(self, conn, action: SQLAction) -> List[Dict[str, Any]]:
        sql = f"SELECT * FROM {action.table}"
        if action.condition:
            sql += f" WHERE {action.condition}"
        cursor = conn.execute(sql)
        cols   = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    def _insert(self, conn, action: SQLAction) -> None:
        if not action.data:
            raise ValueError("INSERT requires data dict")
        cols   = ", ".join(action.data.keys())
        marks  = ", ".join("?" for _ in action.data)
        vals   = tuple(action.data.values())
        conn.execute(f"INSERT INTO {action.table} ({cols}) VALUES ({marks})", vals)

    def _update(self, conn, action: SQLAction) -> None:
        if not action.data:
            raise ValueError("UPDATE requires data dict")
        sets   = ", ".join(f"{k} = ?" for k in action.data)
        vals   = tuple(action.data.values())
        sql    = f"UPDATE {action.table} SET {sets}"
        if action.condition:
            sql  += f" WHERE {action.condition}"
            conn.execute(sql, vals)
        else:
            conn.execute(sql, vals)

    def _delete_row(self, conn, action: SQLAction) -> None:
        sql = f"DELETE FROM {action.table}"
        if action.condition:
            sql += f" WHERE {action.condition}"
        conn.execute(sql)

    def _drop_table(self, conn, action: SQLAction) -> None:
        conn.execute(f"DROP TABLE IF EXISTS {action.table}")

    # ---------------------------------------------------------------- #
    #  Internal                                                          #
    # ---------------------------------------------------------------- #

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SEED_SQL)
