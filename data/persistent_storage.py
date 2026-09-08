"""
Persistent Storage Manager for Binance Futures Trading Desk
Saves and restores all trading state to disk via SQLite and JSON backup:
- Account balance, equity, peak balance, fees
- Complete trade history and PnL metrics
- Open positions and orders
- Episodic Trade Memory lessons and market profiles
- User API settings, trading mode (Live/Demo), and VIP fee configurations
Guarantees zero data loss on browser refresh (F5) or server reboot.
"""
import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union


class PersistentStorageManager:
    # Hai chế độ paper/live có sổ sách TÁCH BIỆT: đổi chế độ là chuyển hẳn
    # bộ dữ liệu (số dư, lệnh, bài học AI, mục tiêu tháng), không lẫn lộn.
    # Paper giữ nguyên key cũ (tương thích dữ liệu hiện có); live dùng key
    # có hậu tố _live. Key không thuộc 2 nhóm này (API, key sàn...) dùng chung.
    LIVE_SUFFIX = "_live"
    # Các key sổ sách theo chế độ (phải tách paper/live)
    MODE_SCOPED_PREFIXES = (
        "monthly_",          # mục tiêu tháng: target, profile, deficit, month, start_balance
    )
    MODE_SCOPED_KEYS = {
        "execution_runtime",
    }

    def __init__(self, db_path: Optional[Union[str, Path]] = None, json_backup_path: Optional[Union[str, Path]] = None, secret_provider: Optional[Any] = None, mode: str = "paper"):
        self.mode = "live" if str(mode).lower() == "live" else "paper"
        if db_path is None:
            base_dir = Path(__file__).resolve().parent
            db_path = base_dir / "bot_database.db"
        if json_backup_path is None:
            base_dir = Path(__file__).resolve().parent
            json_backup_path = base_dir / "persistent_state.json"

        self.db_path = Path(db_path)
        self.json_backup_path = Path(json_backup_path)
        self.secret_provider = secret_provider
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # 1. Account State (id=1 paper, id=2 live — sổ sách tách biệt)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS account_state (
                    id INTEGER PRIMARY KEY CHECK (id IN (1, 2)),
                    symbol TEXT DEFAULT 'BTCUSDT',
                    initial_balance REAL DEFAULT 1000.0,
                    current_balance REAL DEFAULT 1000.0,
                    peak_balance REAL DEFAULT 1000.0,
                    total_fees REAL DEFAULT 0.0,
                    is_running INTEGER DEFAULT 1,
                    active_timeframe TEXT DEFAULT '15m',
                    leverage_mode TEXT DEFAULT 'AI_AUTO',
                    manual_leverage INTEGER DEFAULT 3,
                    current_position TEXT,
                    updated_at TEXT
                )
            """)

            # 2. Trades (cột mode: paper/live — lịch sử tách biệt)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    symbol TEXT,
                    timeframe TEXT,
                    direction TEXT,
                    entry_time TEXT,
                    entry_price REAL,
                    breakeven_price REAL,
                    exit_time TEXT,
                    exit_price REAL,
                    fee REAL,
                    reason TEXT,
                    pnl REAL,
                    return_pct REAL,
                    created_at TEXT,
                    mode TEXT DEFAULT 'paper'
                )
            """)

            # 3. Trade Memory Records (LLM_trader / Condor Episodic Memory)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_memory_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER,
                    direction INTEGER,
                    entry_price REAL,
                    net_pnl REAL,
                    exit_reason TEXT,
                    rsi REAL,
                    vwap_status TEXT,
                    absorption_signal TEXT,
                    delta_momentum TEXT,
                    smc_structure TEXT,
                    lesson_learned TEXT,
                    recorded_at TEXT,
                    mode TEXT DEFAULT 'paper'
                )
            """)

            # Di chuyển bảng account_state cũ (CHECK id=1) sang schema mới (id 1/2).
            try:
                old_sql = (cursor.execute(
                    "SELECT sql FROM sqlite_master WHERE name='account_state'").fetchone() or [None])[0] or ""
                if "IN (1, 2)" not in old_sql and "IN (1,2)" not in old_sql:
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS account_state_new (
                            id INTEGER PRIMARY KEY CHECK (id IN (1, 2)),
                            symbol TEXT DEFAULT 'BTCUSDT',
                            initial_balance REAL DEFAULT 1000.0,
                            current_balance REAL DEFAULT 1000.0,
                            peak_balance REAL DEFAULT 1000.0,
                            total_fees REAL DEFAULT 0.0,
                            is_running INTEGER DEFAULT 1,
                            active_timeframe TEXT DEFAULT '15m',
                            leverage_mode TEXT DEFAULT 'AI_AUTO',
                            manual_leverage INTEGER DEFAULT 3,
                            current_position TEXT,
                            updated_at TEXT
                        )
                    """)
                    cursor.execute("INSERT OR IGNORE INTO account_state_new SELECT * FROM account_state")
                    cursor.execute("DROP TABLE account_state")
                    cursor.execute("ALTER TABLE account_state_new RENAME TO account_state")
            except Exception:
                pass
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """)
            trade_columns = {row[1] for row in cursor.execute("PRAGMA table_info(trades)")}
            for name in ("execution_id", "payload_json"):
                if name not in trade_columns:
                    cursor.execute(f"ALTER TABLE trades ADD COLUMN {name} TEXT")
            if "mode" not in trade_columns:
                cursor.execute("ALTER TABLE trades ADD COLUMN mode TEXT DEFAULT 'paper'")
            cursor.execute("UPDATE trades SET mode='paper' WHERE mode IS NULL OR mode=''")
            mem_columns = {row[1] for row in cursor.execute("PRAGMA table_info(trade_memory_records)")}
            if "mode" not in mem_columns:
                cursor.execute("ALTER TABLE trade_memory_records ADD COLUMN mode TEXT DEFAULT 'paper'")
            cursor.execute("UPDATE trade_memory_records SET mode='paper' WHERE mode IS NULL OR mode=''")
            # Dữ liệu cũ không có mode -> thuộc về paper (id=1 giữ nguyên).
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS trades_execution_id ON trades(execution_id)")
            conn.commit()

    # -------------------------------------------------------------
    # Account State
    # -------------------------------------------------------------
    def _mode_id(self) -> int:
        return 2 if self.mode == "live" else 1

    def save_account_state(
        self,
        symbol: str,
        initial_balance: float,
        current_balance: float,
        peak_balance: float,
        total_fees: float,
        is_running: bool,
        active_timeframe: str,
        leverage_mode: str,
        manual_leverage: int,
        current_position: Optional[dict] = None
    ):
        pos_json = json.dumps(current_position) if current_position else None
        now_str = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO account_state (
                    id, symbol, initial_balance, current_balance, peak_balance,
                    total_fees, is_running, active_timeframe, leverage_mode,
                    manual_leverage, current_position, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    symbol=excluded.symbol,
                    initial_balance=excluded.initial_balance,
                    current_balance=excluded.current_balance,
                    peak_balance=excluded.peak_balance,
                    total_fees=excluded.total_fees,
                    is_running=excluded.is_running,
                    active_timeframe=excluded.active_timeframe,
                    leverage_mode=excluded.leverage_mode,
                    manual_leverage=excluded.manual_leverage,
                    current_position=excluded.current_position,
                    updated_at=excluded.updated_at
            """, (
                self._mode_id(), symbol, initial_balance, current_balance, peak_balance,
                total_fees, 1 if is_running else 0, active_timeframe,
                leverage_mode, manual_leverage, pos_json, now_str
            ))
            conn.commit()

        # Update JSON backup
        self._sync_to_json_backup()

    def load_account_state(self) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM account_state WHERE id = ?", (self._mode_id(),))
            row = cursor.fetchone()
            if row:
                pos = json.loads(row["current_position"]) if row["current_position"] else None
                return {
                    "symbol": row["symbol"],
                    "initial_balance": float(row["initial_balance"]),
                    "current_balance": float(row["current_balance"]),
                    "peak_balance": float(row["peak_balance"]),
                    "total_fees": float(row["total_fees"]),
                    "is_running": bool(row["is_running"]),
                    "active_timeframe": row["active_timeframe"],
                    "leverage_mode": row["leverage_mode"],
                    "manual_leverage": int(row["manual_leverage"]),
                    "current_position": pos,
                    "updated_at": row["updated_at"]
                }
        return None

    # -------------------------------------------------------------
    # Trades Persistence
    # -------------------------------------------------------------
    @staticmethod
    def _insert_trade(cursor: sqlite3.Cursor, trade: dict, mode: str = "paper") -> int:
        now_str = str(trade.get("created_at") or datetime.now().isoformat())
        execution_id = str(trade["execution_id"]) if trade.get("execution_id") else None
        exit_time = str(trade.get("closed_at") or trade.get("exit_time", ""))
        close_timestamp = trade.get("closed_at_ts", trade.get("timestamp"))
        if close_timestamp is not None:
            close_timestamp = float(close_timestamp)
            exit_time = datetime.fromtimestamp(
                close_timestamp / 1000.0 if close_timestamp > 1e11 else close_timestamp
            ).isoformat()
        cursor.execute("""
            INSERT OR IGNORE INTO trades (
                trade_id, symbol, timeframe, direction, entry_time,
                entry_price, breakeven_price, exit_time, exit_price,
                fee, reason, pnl, return_pct, created_at, execution_id, payload_json, mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.get("id", trade.get("trade_id", 0)), trade.get("symbol", "BTCUSDT"),
            trade.get("timeframe", "15m"), trade.get("direction", "LONG"),
            str(trade.get("entry_time", "")), float(trade.get("entry_price", 0.0)),
            float(trade.get("breakeven_price", 0.0)), exit_time,
            float(trade.get("exit_price", 0.0)), float(trade.get("fee", 0.0)),
            str(trade.get("reason", "")), float(trade.get("pnl", 0.0)),
            float(trade.get("return_pct", 0.0)), now_str, execution_id,
            json.dumps(trade, ensure_ascii=False, allow_nan=False),
            str(trade.get("mode", mode) or mode),
        ))
        if cursor.rowcount:
            return cursor.lastrowid
        return 0

    def save_trade(self, trade: dict) -> int:
        with self._get_connection() as conn:
            inserted_id = self._insert_trade(conn.cursor(), trade, self.mode)

        self._sync_to_json_backup()
        return inserted_id

    def save_execution_runtime(self, payload: dict) -> int:
        """Atomically checkpoint runtime plus new execution trades; return inserted row count."""
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        inserted = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for trade in payload.get("trades", []):
                if trade.get("execution_id"):
                    inserted += bool(self._insert_trade(cursor, trade, self.mode))
            cursor.execute("""
                INSERT INTO user_settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """, (self._key("execution_runtime"), serialized, datetime.now().isoformat()))
        self._sync_to_json_backup()
        return inserted

    def load_execution_runtime(self) -> Optional[dict]:
        payload = self.get_setting("execution_runtime")
        return payload if isinstance(payload, dict) else None

    def load_trades(self, limit: int = 200) -> List[dict]:
        trades: List[dict] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trades WHERE mode = ? ORDER BY id DESC LIMIT ?
            """, (self.mode, limit,))
            rows = cursor.fetchall()
            for r in reversed(rows):
                payload = json.loads(r["payload_json"]) if r["payload_json"] else {}
                trades.append({**payload,
                    "id": r["trade_id"] or r["id"],
                    "symbol": r["symbol"],
                    "timeframe": r["timeframe"] or "15m",
                    "direction": r["direction"],
                    "entry_time": r["entry_time"],
                    "entry_price": float(r["entry_price"]),
                    "breakeven_price": float(r["breakeven_price"]),
                    "exit_time": r["exit_time"],
                    "exit_price": float(r["exit_price"]),
                    "fee": float(r["fee"]),
                    "reason": r["reason"],
                    "pnl": float(r["pnl"]),
                    "return_pct": float(r["return_pct"]),
                    "created_at": r["created_at"] if "created_at" in r.keys() else "",
                    "execution_id": r["execution_id"],
                })
        return trades

    def get_monthly_realized_pnl(self, year_month: str) -> float:
        """Sum by close month; legacy MM-DD dates require a persisted creation year."""
        try:
            datetime.strptime(year_month, "%Y-%m")
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT COALESCE(SUM(pnl), 0.0) FROM trades
                    WHERE mode = ? AND (CASE
                        WHEN date(substr(exit_time, 1, 10)) IS NOT NULL
                            THEN substr(exit_time, 1, 7)
                        WHEN exit_time GLOB '[0-9][0-9]-[0-9][0-9]*'
                             AND date(substr(created_at, 1, 10)) IS NOT NULL
                            THEN substr(created_at, 1, 4) || '-' || substr(exit_time, 1, 2)
                        ELSE substr(created_at, 1, 7)
                    END = ?)
                """, (self.mode, year_month,))
                res = cursor.fetchone()
                return round(float(res[0]), 2) if res else 0.0
        except Exception as e:
            print(f"[PersistentStorage] Error querying monthly realized PnL: {e}")
            return 0.0

    # -------------------------------------------------------------
    # Trade Memory Records Persistence
    # -------------------------------------------------------------
    def save_trade_memory_record(
        self,
        trade_id: int,
        direction: int,
        entry_price: float,
        net_pnl: float,
        exit_reason: str,
        rsi: float,
        vwap_status: str,
        absorption_signal: str,
        delta_momentum: str,
        smc_structure: str,
        lesson_learned: str
    ):
        now_str = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trade_memory_records (
                    trade_id, direction, entry_price, net_pnl, exit_reason,
                    rsi, vwap_status, absorption_signal, delta_momentum,
                    smc_structure, lesson_learned, recorded_at, mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id, direction, entry_price, net_pnl, exit_reason,
                rsi, vwap_status, absorption_signal, delta_momentum,
                smc_structure, lesson_learned, now_str, self.mode
            ))
            conn.commit()

        # Immediately sync memory records to persistent_state.json backup
        self._sync_to_json_backup()

    def load_trade_memory_records(self, limit: int = 50) -> List[dict]:
        records: List[dict] = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trade_memory_records WHERE mode = ? ORDER BY id DESC LIMIT ?
            """, (self.mode, limit,))
            rows = cursor.fetchall()
            for r in reversed(rows):
                records.append({
                    "trade_id": r["trade_id"],
                    "direction": r["direction"],
                    "entry_price": float(r["entry_price"]),
                    "net_pnl": float(r["net_pnl"]),
                    "exit_reason": r["exit_reason"],
                    "rsi": float(r["rsi"]),
                    "vwap_status": r["vwap_status"],
                    "absorption_signal": r["absorption_signal"],
                    "delta_momentum": r["delta_momentum"],
                    "smc_structure": r["smc_structure"],
                    "lesson_learned": r["lesson_learned"],
                    "recorded_at": r["recorded_at"]
                })
        return records

    # -------------------------------------------------------------
    # Settings & API Keys Persistence
    # -------------------------------------------------------------
    @classmethod
    def scoped_key(cls, key: str, mode: str) -> str:
        """Key sổ sách theo chế độ: paper giữ nguyên, live thêm hậu tố _live."""
        if str(mode).lower() != "live":
            return key
        if key in cls.MODE_SCOPED_KEYS or key.startswith(cls.MODE_SCOPED_PREFIXES):
            return key + cls.LIVE_SUFFIX
        return key

    def _key(self, key: str) -> str:
        return self.scoped_key(key, self.mode)

    def set_mode(self, mode: str) -> None:
        self.mode = "live" if str(mode).lower() == "live" else "paper"

    def save_setting(self, key: str, value: Any):
        key = self._key(key)
        from security.secret_provider import get_secret_provider, make_credential_ref, is_secret_key
        if is_secret_key(key) and isinstance(value, str) and value.strip() and not value.startswith("keyring://"):
            try:
                provider = self.secret_provider or get_secret_provider()
                provider.set_secret(key, value)
                val_str = make_credential_ref("credentials", key)
            except Exception as e:
                print(f"[PersistentStorage] SecretProvider error: {e}; saving reference")
                val_str = make_credential_ref("credentials", key)
        else:
            val_str = json.dumps(value) if not isinstance(value, str) else value

        now_str = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """, (key, val_str, now_str))
            conn.commit()

    def get_setting(self, key: str, default: Any = None, resolve_secrets: bool = True) -> Any:
        key = self._key(key)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM user_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                try:
                    val = json.loads(row["value"])
                except Exception:
                    val = row["value"]
                if isinstance(val, str) and val.startswith("keyring://"):
                    if not resolve_secrets:
                        return val
                    from security.secret_provider import resolve_credential
                    provider = self.secret_provider
                    resolved = resolve_credential(val, provider=provider)
                    return resolved if resolved else default
                return val
        return default

    def get_all_settings(self, sanitize: bool = False) -> Dict[str, Any]:
        res = {}
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT key, value FROM user_settings")
            rows = cursor.fetchall()
            for r in rows:
                try:
                    res[r["key"]] = json.loads(r["value"])
                except Exception:
                    res[r["key"]] = r["value"]
        if sanitize:
            from security.secret_provider import sanitize_settings_for_export
            return sanitize_settings_for_export(res)
        return res

    def save_vibe_config(
        self,
        enabled: bool = True,
        min_votes: int = 3,
        macro_model: str = "gemini-2.0-flash",
        quant_model: str = "gemini-2.0-flash",
        risk_model: str = "gemini-2.0-flash",
        exec_model: str = "gemini-2.0-flash"
    ):
        cfg = {
            "enabled": bool(enabled),
            "min_votes": int(min_votes),
            "macro_model": str(macro_model),
            "quant_model": str(quant_model),
            "risk_model": str(risk_model),
            "exec_model": str(exec_model)
        }
        self.save_setting("vibe_swarm_config", cfg)
        return cfg

    def get_vibe_config(self) -> Dict[str, Any]:
        default_cfg = {
            "enabled": True,
            "min_votes": 3,
            "macro_model": "ag/gemini-3.8-flash-high",
            "quant_model": "ag/gemini-3.8-flash-high",
            "risk_model": "ag/gemini-3.8-flash-high",
            "exec_model": "ag/gemini-3.8-flash-high"
        }
        saved = self.get_setting("vibe_swarm_config", default_cfg)
        if isinstance(saved, dict):
            default_cfg.update(saved)
            # Automatically migrate legacy placeholder names to 9Router format
            for k in ["macro_model", "quant_model", "risk_model", "exec_model"]:
                val = str(default_cfg.get(k, "")).strip()
                if val in ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "deepseek-r1", ""):
                    default_cfg[k] = "ag/gemini-3.8-flash-high"
        return default_cfg

    # -------------------------------------------------------------
    # Database Reset & Purge
    # -------------------------------------------------------------
    def reset_database(self, initial_balance: float = 1000.0, symbol: str = "BTCUSDT"):
        # Reset CHỈ chế độ hiện tại: paper reset không xóa sổ live và ngược lại.
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trades WHERE mode = ?", (self.mode,))
            cursor.execute("DELETE FROM trade_memory_records WHERE mode = ?", (self.mode,))
            cursor.execute("DELETE FROM account_state WHERE id = ?", (self._mode_id(),))
            cursor.execute("DELETE FROM user_settings WHERE key = ?", (self._key("execution_runtime"),))
            for prefix in self.MODE_SCOPED_PREFIXES:
                if self.mode == "live":
                    cursor.execute(
                        "DELETE FROM user_settings WHERE key LIKE ? AND key LIKE ?",
                        (prefix + "%", "%" + self.LIVE_SUFFIX))
                else:
                    cursor.execute(
                        "DELETE FROM user_settings WHERE key LIKE ? AND key NOT LIKE ?",
                        (prefix + "%", "%" + self.LIVE_SUFFIX))
            conn.commit()

        self.save_account_state(
            symbol=symbol,
            initial_balance=initial_balance,
            current_balance=initial_balance,
            peak_balance=initial_balance,
            total_fees=0.0,
            is_running=True,
            active_timeframe="15m",
            leverage_mode="AI_AUTO",
            manual_leverage=3,
            current_position=None
        )
        self._sync_to_json_backup()

    # -------------------------------------------------------------
    # JSON Backup Sync
    # -------------------------------------------------------------
    def _sync_to_json_backup(self):
        try:
            state = self.load_account_state()
            trades = self.load_trades(limit=100)
            memories = self.load_trade_memory_records(limit=100)
            settings = self.get_all_settings()
            safe_settings = dict(settings)
            if "api_secret" in safe_settings and safe_settings["api_secret"]:
                secret = str(safe_settings["api_secret"])
                safe_settings["api_secret_masked"] = secret[:4] + "*" * max(0, len(secret) - 8) + secret[-4:] if len(secret) > 8 else "****"
            if "mexc_api_secret" in safe_settings and safe_settings["mexc_api_secret"]:
                m_secret = str(safe_settings["mexc_api_secret"])
                safe_settings["mexc_api_secret_masked"] = m_secret[:4] + "*" * max(0, len(m_secret) - 8) + m_secret[-4:] if len(m_secret) > 8 else "****"

            data = {
                "backup_time": datetime.now().isoformat(),
                "mode": self.mode,
                "account_state": state,
                "trades_count": len(trades),
                "trades": trades[-20:],
                "ai_lessons_count": len(memories),
                "ai_lessons": memories[-30:],
                "settings_summary": {k: v for k, v in safe_settings.items() if k not in ("api_secret", "mexc_api_secret")}
            }
            with open(self.json_backup_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[PersistentStorageManager] Backup JSON sync error: {e}", flush=True)

    def get_storage_telemetry(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trades WHERE mode = ?", (self.mode,))
            trades_cnt = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM trade_memory_records WHERE mode = ?", (self.mode,))
            mem_cnt = cursor.fetchone()[0]

        sqlite_size_kb = 0.0
        if self.db_path.exists():
            try:
                sqlite_size_kb = round(self.db_path.stat().st_size / 1024, 2)
            except Exception:
                pass

        json_size_kb = 0.0
        json_last_modified = None
        json_trades_cnt = trades_cnt
        json_mem_cnt = mem_cnt
        if self.json_backup_path.exists():
            try:
                json_size_kb = round(self.json_backup_path.stat().st_size / 1024, 2)
                json_last_modified = datetime.fromtimestamp(self.json_backup_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                with open(self.json_backup_path, "r", encoding="utf-8") as f:
                    j_data = json.load(f)
                    json_trades_cnt = j_data.get("trades_count", trades_cnt)
                    json_mem_cnt = j_data.get("ai_lessons_count", mem_cnt)
            except Exception:
                pass

        return {
            "sqlite_trades_count": trades_cnt,
            "sqlite_memory_count": mem_cnt,
            "sqlite_db_name": self.db_path.name,
            "sqlite_db_size_kb": sqlite_size_kb,
            "json_backup_name": self.json_backup_path.name,
            "json_trades_count": json_trades_cnt,
            "json_lessons_count": json_mem_cnt,
            "json_file_size_kb": json_size_kb,
            "json_last_sync": json_last_modified or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_synced": (trades_cnt == json_trades_cnt and mem_cnt == json_mem_cnt)
        }

    # -------------------------------------------------------------
    # Full Data Export & Migration (Di Chuyển Sang Máy Mới)
    # -------------------------------------------------------------
    def export_full_package(self, include_trades: bool = True, include_memory: bool = True, include_settings: bool = True) -> Dict[str, Any]:
        """Exports selected account state, trades, trade memory records, and settings into a single portable dictionary."""
        trades = self.load_trades(limit=10000) if include_trades else []
        memories = self.load_trade_memory_records(limit=5000) if include_memory else []
        settings = self.get_all_settings(sanitize=True) if include_settings else {}
        account = self.load_account_state()
        
        return {
            "version": "1.0",
            "app": "binance_futures_algo_desk",
            "mode": self.mode,
            "exported_at": datetime.now().isoformat(),
            "account_state": account,
            "trades": trades,
            "trades_count": len(trades),
            "trade_memory_records": memories,
            "trade_memory_count": len(memories),
            "settings": settings,
            "settings_count": len(settings),
            "included_sections": {
                "trades": include_trades,
                "memory": include_memory,
                "settings": include_settings
            }
        }

    def import_full_package(self, data: Dict[str, Any], overwrite_account_state: bool = False, mode: str = "merge") -> Dict[str, Any]:
        """Imports and restores trades, memory lessons, and settings from another machine's backup.
        Supports 'merge' (append with deduplication) or 'overwrite' (purge existing data for included sections).
        """
        from security.secret_provider import validate_settings_for_import
        raw_settings = data.get("settings") or data.get("user_settings") or {}
        ok, err_msg = validate_settings_for_import(raw_settings)
        if not ok:
            raise ValueError(f"Secret isolation violation in import: {err_msg}")

        imported_trades = 0
        imported_memories = 0
        imported_settings = 0
        is_overwrite = (str(mode).lower() in ("overwrite", "replace"))

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # In overwrite mode, purge previous data for included sections
            # (CHỈ chế độ hiện tại — không xóa sổ chế độ kia).
            if is_overwrite:
                if "trades" in data:
                    cursor.execute("DELETE FROM trades WHERE mode = ?", (self.mode,))
                if "trade_memory_records" in data or "ai_lessons" in data:
                    cursor.execute("DELETE FROM trade_memory_records WHERE mode = ?", (self.mode,))
                if "settings" in data or "user_settings" in data:
                    cursor.execute("DELETE FROM user_settings WHERE key = ?", (self._key("execution_runtime"),))
                    for prefix in self.MODE_SCOPED_PREFIXES:
                        if self.mode == "live":
                            cursor.execute(
                                "DELETE FROM user_settings WHERE key LIKE ? AND key LIKE ?",
                                (prefix + "%", "%" + self.LIVE_SUFFIX))
                        else:
                            cursor.execute(
                                "DELETE FROM user_settings WHERE key LIKE ? AND key NOT LIKE ?",
                                (prefix + "%", "%" + self.LIVE_SUFFIX))

            # 1. Import Trades (deduplicate by trade_id or entry_time+exit_time if merge mode)
            # Phạm vi CHỈ chế độ hiện tại — không chạm sổ chế độ kia.
            trades = data.get("trades", [])
            for t in trades:
                tid = t.get("id") or t.get("trade_id")
                exists = False
                if not is_overwrite:
                    if tid:
                        cursor.execute("SELECT id FROM trades WHERE trade_id = ? AND mode = ?", (tid, self.mode))
                        if cursor.fetchone():
                            exists = True
                    if not exists and t.get("entry_time") and t.get("exit_time"):
                        cursor.execute("SELECT id FROM trades WHERE entry_time = ? AND exit_time = ? AND mode = ?",
                                       (str(t.get("entry_time", "")), str(t.get("exit_time", "")), self.mode))
                        if cursor.fetchone():
                            exists = True

                if is_overwrite or not exists:
                    imported_trades += bool(self._insert_trade(cursor, {**t, "mode": self.mode}, self.mode))

            # 2. Import Trade Memory Records
            memories = data.get("trade_memory_records") or data.get("ai_lessons") or []
            for m in memories:
                lesson = m.get("lesson_learned", "")
                recorded = m.get("recorded_at", "")
                exists = False
                if not is_overwrite:
                    cursor.execute("SELECT id FROM trade_memory_records WHERE (lesson_learned = ? OR (trade_id = ? AND recorded_at = ?)) AND mode = ?",
                                   (lesson, m.get("trade_id", 0), recorded, self.mode))
                    if cursor.fetchone():
                        exists = True
                if is_overwrite or not exists:
                    cursor.execute("""
                        INSERT INTO trade_memory_records (
                            trade_id, direction, entry_price, net_pnl, exit_reason,
                            rsi, vwap_status, absorption_signal, delta_momentum,
                            smc_structure, lesson_learned, recorded_at, mode
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        m.get("trade_id", 0),
                        m.get("direction", 1),
                        float(m.get("entry_price", 0.0)),
                        float(m.get("net_pnl", 0.0)),
                        str(m.get("exit_reason", "")),
                        float(m.get("rsi", 50.0)),
                        str(m.get("vwap_status", "")),
                        str(m.get("absorption_signal", "")),
                        str(m.get("delta_momentum", "")),
                        str(m.get("smc_structure", "")),
                        lesson,
                        recorded or datetime.now().isoformat(),
                        self.mode
                    ))
                    imported_memories += 1

            # 3. Import Settings (chỉ key sổ sách theo mode mới map key; key dùng
            # chung như API/fees giữ nguyên để không mất cấu hình sàn).
            settings = data.get("settings") or data.get("user_settings") or {}
            for k, v in settings.items():
                val_str = json.dumps(v) if not isinstance(v, str) else v
                cursor.execute("""
                    INSERT INTO user_settings (key, value, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """, (self._key(k), val_str, datetime.now().isoformat()))
                imported_settings += 1

            # 4. Optional: Account State
            if overwrite_account_state and "account_state" in data and data["account_state"]:
                acc = data["account_state"]
                self.save_account_state(
                    symbol=acc.get("symbol", "BTCUSDT"),
                    initial_balance=acc.get("initial_balance", 1000.0),
                    current_balance=acc.get("current_balance", 1000.0),
                    peak_balance=acc.get("peak_balance", 1000.0),
                    total_fees=acc.get("total_fees", 0.0),
                    is_running=acc.get("is_running", True),
                    active_timeframe=acc.get("active_timeframe", "15m"),
                    leverage_mode=acc.get("leverage_mode", "AI_AUTO"),
                    manual_leverage=acc.get("manual_leverage", 3),
                    current_position=acc.get("current_position")
                )

            conn.commit()

        self._sync_to_json_backup()
        mode_text = "Thay thế" if is_overwrite else "Nhập thêm"
        return {
            "status": "ok",
            "success": True,
            "mode": "overwrite" if is_overwrite else "merge",
            "imported_trades": imported_trades,
            "imported_memories": imported_memories,
            "imported_settings": imported_settings,
            "message": f"Khôi phục thành công ({mode_text}): {imported_trades} giao dịch, {imported_memories} bài học AI, {imported_settings} cài đặt!"
        }

