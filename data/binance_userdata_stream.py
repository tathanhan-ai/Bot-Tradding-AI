# -*- coding: utf-8 -*-
"""Binance Futures user-data stream: fill ve tuc thi (<1s), khong doi reconcile.

Nhan ORDER_TRADE_UPDATE (khop/huy lenh) + ACCOUNT_UPDATE (vi the, UPNL, so du)
qua listenKey (song 60 phut, gia han moi 30 phut). Mang rot thi reconcile
0.5s ke thua — khong mat tin hieu, chi cham hon.

Quy uoc xu ly (khop voi execution.py):
- ORDER_TRADE_UPDATE: tim lenh local theo orderId/clientOrderId -> record
  receipt + apply fill ngay. Khong khop thi log warning (lenh ngoai luong).
- ACCOUNT_UPDATE position section: cap nhat UPNL/vi the theo san ngay.
"""
import asyncio
import json
import threading
import time
from typing import Any, Callable, Dict, List, Optional

try:
    import websockets
except Exception:  # test moi truong khong co websockets
    websockets = None


def parse_order_update(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rut thong tin fill tu ORDER_TRADE_UPDATE. Tra None neu khong phai."""
    try:
        if data.get("e") != "ORDER_TRADE_UPDATE":
            return None
        o = data.get("o", {})
        return {
            "symbol": str(o.get("s", "")),
            "order_id": str(o.get("i", "")),
            "client_order_id": str(o.get("c", "")),
            "side": str(o.get("S", "")),
            "order_type": str(o.get("o", "")),
            "status": str(o.get("X", "")),
            "executed_qty": float(o.get("z", 0.0) or 0.0),
            "cum_quote": float(o.get("q", 0.0) or 0.0),
            "avg_price": float(o.get("ap", 0.0) or 0.0),
            "raw": o,
        }
    except Exception:
        return None


def parse_account_update(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Rut vi the + UPNL tu ACCOUNT_UPDATE. Tra None neu khong phai."""
    try:
        if data.get("e") != "ACCOUNT_UPDATE":
            return None
        a = data.get("a", {})
        positions: List[Dict[str, Any]] = []
        for p in a.get("P", []):
            try:
                amt = float(p.get("pa", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if abs(amt) <= 0:
                continue
            try:
                positions.append({
                    "symbol": str(p.get("s", "")),
                    "amount": amt,
                    "entry_price": float(p.get("ep", 0.0) or 0.0),
                    "unrealized_pnl": float(p.get("up", 0.0) or 0.0),
                })
            except (TypeError, ValueError):
                continue
        balances: Dict[str, float] = {}
        for b in a.get("B", []):
            try:
                balances[str(b.get("a", ""))] = float(b.get("wb", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
        return {"positions": positions, "balances": balances, "raw": a}
    except Exception:
        return None


class BinanceUserDataStream:
    """User-data stream: listenKey + keepalive + reconnect. Thread-safe callbacks."""

    def __init__(
        self,
        api_manager=None,
        on_order_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_account_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.api = api_manager
        self.on_order_update = on_order_update
        self.on_account_update = on_account_update
        self.listen_key = ""
        self.is_running = False
        self.last_message_at = 0.0
        self.last_error = ""
        self.events_received = 0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> bool:
        if self.is_running:
            return True
        if websockets is None:
            self.last_error = "thieu thu vien websockets"
            return False
        ok, key = self._ensure_key()
        if not ok:
            return False
        self.is_running = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="binance-userdata", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self.is_running = False
        self._stop.set()

    def _ensure_key(self):
        try:
            ok, data = self.api.create_listen_key()
            if ok and isinstance(data, dict) and data.get("listenKey"):
                self.listen_key = str(data["listenKey"])
                return True, self.listen_key
            self.last_error = str(data)[:200]
            return False, data
        except Exception as exc:
            self.last_error = type(exc).__name__
            return False, None

    def _run(self):
        backoff = 1.0
        last_keepalive = 0.0
        while self.is_running and not self._stop.is_set():
            try:
                if time.time() - last_keepalive > 1800 and self.listen_key:
                    try:
                        self.api.keepalive_listen_key(self.listen_key)
                    except Exception:
                        pass
                    last_keepalive = time.time()
                url = self.api.user_data_stream_url(self.listen_key)
                asyncio.run(self._consume(url))
                backoff = 1.0
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                # listenKey het han (60 phut khong keepalive duoc) -> xin moi.
                if "401" in str(exc) or "listen" in str(exc).lower():
                    self._ensure_key()
                time.sleep(backoff)
                backoff = min(backoff * 1.5, 30.0)

    async def _consume(self, url: str):
        async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=2 ** 20) as ws:
            while self.is_running and not self._stop.is_set():
                raw = await asyncio.wait_for(ws.recv(), 30)
                try:
                    data = json.loads(raw)
                except Exception:
                    continue
                self.last_message_at = time.time()
                self.events_received += 1
                order = parse_order_update(data)
                if order is not None:
                    try:
                        if self.on_order_update:
                            self.on_order_update(order)
                    except Exception:
                        pass
                    continue
                acct = parse_account_update(data)
                if acct is not None:
                    try:
                        if self.on_account_update:
                            self.on_account_update(acct)
                    except Exception:
                        pass
