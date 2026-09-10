# -*- coding: utf-8 -*-
"""Dieu phoi don bay + chot lai theo dinh (peak-lock).

Van de that tu 8 lenh live #65-72: giu lai 14% dinh (thuc nhan $0.20
tren $1.43 dinh co the). Nguyen nhan kep:
1. Don bay cung 5x moi lenh: ATR thap thi phi bao mon, ATR cao thi SL gan
   thanh ly. Khong co co che tang/giam theo bien dong that.
2. Chot lai thu dong: breakeven lock +0.5R, partial +1.0R, trailing dinh
   -1.0 ATR — nhung TP goc dat xa (RR 2-3.5) nen gia quay dau truoc khi
   cham bat ky moc nao; SL goc nam yen trong khi lai dinh troi qua.

Quy tac moi (giu fail-closed, chi mo cua co kiem soat):
- Don bay = f(ATR%): ATR cang thap don bay cang cao (toi da 10x), ATR cao
  ha ve 3x. Luon giu thanh ly cach SL toi thieu 3x khoang SL.
- Chot lai 3 tang theo % dinh (khong doi TP goc):
  + Dat +0.30% (net sau phi): chot 30%, doi SL ve entry (risk-free).
  + Dat +0.50%: chot them 30% (tong 60%), doi SL len +0.25%.
  + Tut ve duoi dinh 0.25% (giveback): chot not phan con lai.
- Moi moc chi kich 1 lan; gia chua cham moc thi giu SL goc.
"""
import math
from typing import Any, Dict, Optional, Tuple

LOCK_1_PCT = 0.0030
LOCK_1_RATIO = 0.30
LOCK_2_PCT = 0.0050
LOCK_2_RATIO = 0.30
LOCK_2_SL_PCT = 0.0025
GIVEBACK_PCT = 0.0025

FEE_ROUNDTRIP_PCT = 0.0007


def leverage_for_atr(atr_pct: float, max_lev: int = 10) -> int:
    """ATR cang thap don bay cang cao (toi da max_lev)."""
    if atr_pct <= 0.0035:
        lev = 10
    elif atr_pct <= 0.0065:
        lev = 8
    elif atr_pct <= 0.0110:
        lev = 6
    elif atr_pct <= 0.0180:
        lev = 4
    else:
        lev = 3
    return max(1, min(int(max_lev or 10), lev))


def liq_distance_pct(leverage: int, mmr: float = 0.004, fee: float = 0.0005) -> float:
    lev = max(1, int(leverage or 1))
    return max(0.0, (1.0 / lev - mmr - fee)) * 100.0


def leverage_with_liq_floor(atr_pct: float, sl_pct: float, max_lev: int = 10) -> int:
    """Don bay theo ATR nhung ha dan den khi thanh ly cach SL >= 3x SL."""
    lev = leverage_for_atr(atr_pct, max_lev)
    floor_sl = max(float(sl_pct or 0.0), 0.002)
    while lev > 1:
        if liq_distance_pct(lev) >= floor_sl * 3.0 * 100.0:
            break
        lev -= 1
    return max(1, lev)


def peak_levels(entry: float, direction: int) -> Dict[str, float]:
    """Gia cac moc chot theo % dinh (khong phu thuoc TP goc)."""
    if direction not in (-1, 1):
        return {}
    return {
        "lock1": entry * (1.0 + direction * LOCK_1_PCT),
        "lock2": entry * (1.0 + direction * LOCK_2_PCT),
        "lock2_sl": entry * (1.0 + direction * LOCK_2_SL_PCT),
    }


def peak_action(pos: Dict[str, Any], price: float) -> Tuple[str, Dict[str, Any]]:
    """Tra (action, info). Action: hold|lock1|lock2|giveback_exit."""
    entry = float(pos.get("entry_price", 0.0) or 0.0)
    direction = int(pos.get("direction", 0) or 0)
    if entry <= 0 or direction not in (-1, 1) or price <= 0:
        return "hold", {}
    lv = peak_levels(entry, direction)
    prog = (price - entry) / entry * direction
    peak = float(pos.get("peak_lock_price", entry) or entry)
    if direction == 1:
        peak = max(peak, price)
    else:
        peak = min(peak, price)
    peak_prog = (peak - entry) / entry * direction
    flags = pos.get("peak_flags", {}) or {}
    eps = 1e-9
    if not flags.get("lock1") and prog + eps >= LOCK_1_PCT:
        return "lock1", {"peak": peak, "lock_sl": entry, "ratio": LOCK_1_RATIO}
    if not flags.get("lock2") and prog + eps >= LOCK_2_PCT:
        return "lock2", {"peak": peak, "lock_sl": lv["lock2_sl"], "ratio": LOCK_2_RATIO}
    if flags.get("lock1") and peak_prog >= LOCK_1_PCT:
        giveback = (peak - price) / entry * direction
        if giveback >= GIVEBACK_PCT:
            return "giveback_exit", {"peak": peak, "giveback": giveback}
    return "hold", {"peak": peak}
