# -*- coding: utf-8 -*-
"""
Bo nho key 9Router dung chung (P0: khong key cung trong source).
Key lay theo thu tu: tham so truyen vao > Settings UI (SecretProvider) > bien moi truong.
"""
from typing import Optional


def load_ninerouter_key() -> Optional[str]:
    try:
        import os
        from pathlib import Path
        import sqlite3
        candidates = [
            os.environ.get("BOT_DATA_DIR", ""),
            str(Path(__file__).resolve().parent.parent / "data"),
        ]
        for base in candidates:
            if not base:
                continue
            db = Path(base) / "bot_database.db"
            if not db.exists():
                continue
            con = sqlite3.connect(str(db))
            try:
                row = con.execute("SELECT value FROM user_settings WHERE key='ninerouter_api_key'").fetchone()
                if row and row[0]:
                    val = str(row[0])
                    if val.startswith("keyring://"):
                        try:
                            from security.secret_provider import resolve_credential
                            val = resolve_credential(val)
                        except Exception:
                            pass
                    if val and "keyring://" not in val:
                        return val
            finally:
                con.close()
    except Exception:
        pass
    return None


class VibeSwarmCouncilKeyLoader:
    """Alias de copilot dung chung ma khong import vong strategy module."""

    @staticmethod
    def load() -> Optional[str]:
        return load_ninerouter_key()
