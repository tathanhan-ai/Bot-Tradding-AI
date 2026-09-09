# -*- coding: utf-8 -*-
"""P1: deterministic hoa chi thi song 1D/15m cua nguoi dung.

Van de: chi thi Copilot dang chu ("Uu tien song 1D, cho phep 15m khi cung
chieu 1D") chi duoc LLM 9Router tu dien giai — khong bao gio duoc code hoa.
Module nay bien chi thi thanh quy tac deterministic:
- parse_wave_instruction: doc text chi thi bang keyword (khong goi LLM).
- compute_wave_alignment: so huong EMA nhanh/cham tren nen 1d (resample tu
  khung nho neu thieu) va 15m, tra ve huong song tung khung.
- evaluate_wave_gate: quyet dinh VETO / probation / PASS cho candidate.

Mien tru: GRID / reduce_only (khong phai entry directional moi).
"""
from typing import Any, Dict, Optional


def parse_wave_instruction(text: str) -> Dict[str, Any]:
    """Phan tich chi thi song bang keyword, khong goi LLM."""
    t = str(text or "").lower()
    mode = "user_instruction_mode"
    out = {
        "has_instruction": False,
        "prefer_1d": False,
        "allow_15m_aligned": False,
        "block_counter_1d": False,
        "raw": str(text or "")[:300],
    }
    if not t.strip():
        return out
    mentions_wave = any(k in t for k in ("song", "sóng", "wave", "1d", "15m", "d1", "m15",
                                         " khung ", "khung ngay", "khung 15"))
    if not mentions_wave:
        return out
    out["has_instruction"] = True
    if "1d" in t or "d1" in t or "song ngay" in t or "sóng ngày" in t or "khung ngay" in t or ("song" in t and "ngay" in t) or ("sóng" in t and "ngày" in t):
        out["prefer_1d"] = True
    if ("15m" in t or "m15" in t or "15 phut" in t) and any(
            k in t for k in ("cung chieu", "cùng chiều", "cung huong", "cùng hướng",
                             "dong thuan", "đồng thuận", "cho phep", "cho phép",
                             "align", "same direction")):
        out["allow_15m_aligned"] = True
    if any(k in t for k in ("nguoc", "ngược", "nguoc song", "ngược sóng", "counter",
                            "khong danh nguoc", "không đánh ngược", "cam nguoc", "cấm ngược")):
        out["block_counter_1d"] = True
    # Mac dinh an toan: co chi thi song ma khong noi ro -> uu tien 1D + chan nguoc 1D.
    if out["has_instruction"] and not out["allow_15m_aligned"]:
        out["allow_15m_aligned"] = "15m" in t or "m15" in t
    return out


def _ema_direction(frame, fast: int = 20, slow: int = 50) -> int:
    """Huong song: +1 (EMA nhanh tren cham), -1 (duoi), 0 (thieu du lieu)."""
    try:
        if frame is None or len(frame) < slow + 2:
            return 0
        close = frame["close"]
        ema_fast = float(close.ewm(span=fast, adjust=False).mean().iloc[-1])
        ema_slow = float(close.ewm(span=slow, adjust=False).mean().iloc[-1])
        if ema_fast > ema_slow * 1.0005:
            return 1
        if ema_fast < ema_slow * 0.9995:
            return -1
        return 0
    except Exception:
        return 0


def compute_wave_alignment(frames: Dict[str, Any]) -> Dict[str, int]:
    """So huong song 1D va 15m tu khung nen da co (khong goi mang)."""
    frames = frames or {}
    frame_15m = frames.get("15m")
    frame_1d = frames.get("1d")
    if frame_1d is None or len(frame_1d) < 52:
        # Resample tu 1h (24 nen = 1 ngay) neu thieu 1d.
        try:
            import time as _time
            from trading.pipeline import resample_closed_candles
            frame_1h = frames.get("1h")
            if frame_1h is not None and len(frame_1h) >= 48:
                frame_1d = resample_closed_candles(frame_1h, "1d", _time.time())
        except Exception:
            frame_1d = None
    return {
        "wave_1d": _ema_direction(frame_1d),
        "wave_15m": _ema_direction(frame_15m),
    }


def evaluate_wave_gate(direction: int, instruction: Dict[str, Any],
                       alignment: Dict[str, int], order_type: str = "",
                       source: str = "", metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Quy tac cong song. Tra ve dict {verdict, reason, probation, updates}."""
    meta = metadata or {}
    if meta.get("reduce_only") or meta.get("is_exit"):
        return {"verdict": "PASS", "reason": "Risk-reducing exit; wave gate skipped", "updates": {}}
    if (order_type or "").upper() == "GRID" and str(source or "") in ("manual-grid", "auto-grid", "auto"):
        return {"verdict": "PASS", "reason": "Hedge grid; wave gate skipped", "updates": {}}
    if not (instruction or {}).get("has_instruction"):
        return {"verdict": "PASS", "reason": "Khong co chi thi song; cong song bo qua",
                "updates": {"wave_alignment": dict(alignment or {})}}
    if direction not in (-1, 1):
        return {"verdict": "PASS", "reason": "Khong co huong directional; cong song bo qua",
                "updates": {"wave_alignment": dict(alignment or {})}}
    wave_1d = int((alignment or {}).get("wave_1d", 0) or 0)
    wave_15m = int((alignment or {}).get("wave_15m", 0) or 0)
    side = "LONG" if direction == 1 else "SHORT"
    updates = {"wave_alignment": {"wave_1d": wave_1d, "wave_15m": wave_15m},
               "user_instruction_mode": "wave_1d_priority"}
    # Nguoc han song 1D (khi 1D da ro huong) -> VETO tieng Viet.
    if wave_1d != 0 and direction != wave_1d:
        wave_txt = "tang" if wave_1d == 1 else "giam"
        return {"verdict": "VETO",
                "reason": f"Chi thi song: cam {side} nguoc song 1D dang {wave_txt} (uu tien song ngay)",
                "updates": updates}
    # 15m lech song 1D -> khong cam han, ep probation von nho.
    if wave_1d != 0 and wave_15m != 0 and wave_15m != wave_1d:
        updates.update({"wave_misaligned": True, "probation": True, "risk_cap_pct": 0.0010})
        return {"verdict": "PASS",
                "reason": "Chi thi song: song 15m lech song 1D — chi cho tham do probation von nho 0.10%",
                "updates": updates}
    # Cung chieu (hoac 1D/15m chua ro) -> qua, ghi nhan dong thuan.
    updates["wave_aligned"] = (wave_1d == 0 or direction == wave_1d) and (wave_15m == 0 or direction == wave_15m)
    dong = "dong thuan" if updates["wave_aligned"] else "song chua ro — tham do than trong"
    return {"verdict": "PASS",
            "reason": f"Chi thi song: {side} {dong} voi song 1D/15m",
            "updates": updates}
