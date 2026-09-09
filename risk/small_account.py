# -*- coding: utf-8 -*-
"""uat tri von nho: von $100 khong the danh theo dinh muc von lon.

Ba su that gay ket lenh live:
1. GRID doi toi thieu 4 chan x 0.001 BTC (~$316 notional o BTC $79k).
   Vi $99 khong bao gio nuoi noi -> moi chu ky deu veto o sizing.
2. Probation 0.10% x $99.79 = $0.10 risk -> quantity ~0.0003 BTC, nho hon
   buoc toi thieu san 0.001 -> lam tron ve 0 -> veto vinh vien.
3. Tang gui san (submit) cat quantity lan nua theo risk_pct ma khong giu ly
   do, nen lenh duyet pipeline xong van chet am tham.

Quy tac sua (giu fail-closed, chi mo cua toi thieu co the):
- Von < $300: khong choi GRID tu dong (GRID can toi thieu ~$105 ky quy).
- Probation bi tran ve duoi buoc toi thieu san -> nang len dung buoc toi
  thieu (0.001 BTC), gan co min_size_floor, chi khi ky quy toi thieu vua
  trong 35% vi (nguong margin headroom san).
- Veto chi khi ky quy toi thieu cung vuot han muc: luc do ly do phai noi ro
  "von qua nho", khong chet am tham.
"""
import math

MIN_QTY_STEP = 0.001
MIN_GRID_LEGS = 4
GRID_HEADROOM = 0.30
GRID_MIN_BALANCE = 300.0
MARGIN_HEADROOM = 0.35


def min_viable_margin(price: float, leverage: int, step: float = MIN_QTY_STEP) -> float:
    """Ky quy nho nhat de dat 1 lenh dung chuan san (1 buoc so luong)."""
    lev = max(1, int(leverage or 1))
    if price <= 0:
        return 0.0
    return round(step * price / lev, 2)


def grid_affordable(balance: float, price: float, leverage: int,
                    legs: int = MIN_GRID_LEGS, step: float = MIN_QTY_STEP) -> bool:
    """Von co du suc nuoi luoi toi thieu khong (ky quy luoi < 30% von)."""
    if balance <= 0 or price <= 0:
        return False
    need = legs * min_viable_margin(price, leverage, step)
    return need <= float(balance) * GRID_HEADROOM


def counter_margin(balance: float, price: float, leverage: int,
                   abs_floor: float, frac: float, abs_cap: float) -> float:
    """Margin de xuat cho Phuong an 2, co gian theo von.

    Von lon ($5000): giu dung dinh muc cu (abs_floor).
    Von nho ($100): ha ve phan tram von nhung khong thap hon ky quy toi
    thieu san (neu khong, quantity duoi buoc toi thieu -> veto vinh vien).
    """
    bal = max(0.0, float(balance or 0.0))
    frac_part = min(float(abs_floor), bal * float(frac))
    floor = max(min_viable_margin(price, leverage), frac_part)
    return round(min(float(abs_cap), floor), 0)


def probation_floor_quantity(probation_quantity: float, price: float, leverage: int,
                             remaining_margin: float, balance: float,
                             step: float = MIN_QTY_STEP):
    """Nang quantity probation bi tran ve 0 len buoc toi thieu san.

    Tra ve (quantity, min_size_floor, veto_reason). veto_reason khac rong
    nghia la ca lenh toi thieu cung vuot han muc -> veto cung voi ly do ro.
    """
    floored = math.floor((probation_quantity + 1e-12) / step) * step
    if floored > 0:
        return floored, False, ""
    need = min_viable_margin(price, leverage, step)
    if need <= 0:
        return 0.0, False, "Gia/lev khong hop le de tinh lenh toi thieu"
    if need <= max(0.0, remaining_margin) and need <= max(0.0, balance) * MARGIN_HEADROOM:
        return step, True, ""
    return 0.0, False, (
        "Von $%.2f qua nho: lenh toi thieu san can ky quy $%.2f, vuot han muc "
        "35%% von ($%.2f). Nap them von hoac doi von lon hon." % (balance, need, balance * MARGIN_HEADROOM)
    )
