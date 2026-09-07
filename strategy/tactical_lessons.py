# -*- coding: utf-8 -*-
"""
Chien luoc tac chien rut ra tu lich su 60 lenh that (bai hoc xuong mau):
1. HurstRegimeFilter: Hurst < 0.45 = mean-reverting -> CAM directional, chi cho mean-reversion/grid.
   Bang chung: 14/14 lenh thua gan day co Hurst 0.22-0.34 nhung bot van danh directional.
2. SMCtrendAlignment: SMC BEARISH_TREND cam LONG, BULLISH_TREND cam SHORT.
   Bang chung: 9/14 lenh thua LONG trong khi SMC BEARISH_TREND (martingale 14-18 mat -$55.77).
3. FeeAwareEntry: notional nho + SL chat = phi an het lai -> chan tu dau.
   Bang chung: lenh 1077674 lai gop +$0.11 nhung phi $0.11 -> net -$0.0008; fee/lai-gop 48.2%.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


@dataclass
class TacticalLessonVerdict:
    lesson: str                  # 'HURST_FILTER' | 'SMC_ALIGN' | 'FEE_AWARE'
    allowed: bool                # True = cho qua, False = chan
    direction_bias: int          # 1 = chi LONG, -1 = chi SHORT, 0 = cam ca 2 huong
    penalty: float               # 0.0 - 1.0 (giam size theo muc do vi pham)
    rationale: str


class HurstRegimeFilter:
    """Bai hoc #1: Dung danh momentum trong thi truong mean-reverting."""

    def __init__(self, mr_threshold: float = 0.45, trend_threshold: float = 0.55):
        self.mr_threshold = mr_threshold
        self.trend_threshold = trend_threshold

    def evaluate(self, hurst: float, proposed_type: str = "") -> TacticalLessonVerdict:
        hurst = float(hurst or 0.50)
        if hurst < self.mr_threshold:
            # Mean-reverting manh: cam MARKET/TRAILING (duoi trend), chi cho Maker/DCA/Grid
            if proposed_type in ("MARKET", "TRAILING_STOP", "TWAP"):
                return TacticalLessonVerdict(
                    lesson="HURST_FILTER", allowed=False, direction_bias=0, penalty=1.0,
                    rationale=f"🛡️ [HURST={hurst:.2f} MEAN-REVERTING] Cam lenh duoi trend {proposed_type}; chi cho Maker/DCA/Grid bat dao chieu.")
            return TacticalLessonVerdict(
                lesson="HURST_FILTER", allowed=True, direction_bias=0, penalty=0.0,
                rationale=f"Hurst {hurst:.2f} mean-reverting: uu tien mean-reversion, size giam 30%.")
        if hurst > self.trend_threshold:
            return TacticalLessonVerdict(
                lesson="HURST_FILTER", allowed=True, direction_bias=0, penalty=0.0,
                rationale=f"Hurst {hurst:.2f} trending: cho phep duoi trend.")
        return TacticalLessonVerdict(
            lesson="HURST_FILTER", allowed=True, direction_bias=0, penalty=0.2,
            rationale=f"Hurst {hurst:.2f} random-walk: size tham do.")


class SMCTrendAlignment:
    """Bai hoc #2 (martingale 14-18): Cam danh nguoc cau truc SMC.
    Ngoai le: sweep/bias dao chieu moi (BULLISH_REVERSAL cho LONG, BEARISH_REVERSAL cho SHORT)
    duoc cho qua vi sweep chinh la tin hieu dao chieu - structure SMA20/50 cham hon 1 nhip."""

    def evaluate(self, smc_structure: str, direction: int,
                 smc_bias: str = "", has_fresh_sweep: bool = False) -> TacticalLessonVerdict:
        smc = str(smc_structure or "RANGING").upper()
        bias = str(smc_bias or "").upper()
        if smc == "BEARISH_TREND" and direction == 1:
            if has_fresh_sweep or bias == "BULLISH_REVERSAL":
                return TacticalLessonVerdict(
                    lesson="SMC_ALIGN", allowed=True, direction_bias=0, penalty=0.3,
                    rationale="SMC BEARISH_TREND nhung co sweep dao chieu tang moi: cho LONG tham do (size -30%).")
            return TacticalLessonVerdict(
                lesson="SMC_ALIGN", allowed=False, direction_bias=-1, penalty=1.0,
                rationale="🛡️ [SMC BEARISH_TREND] Cam LONG nguoc cau truc (bai hoc martingale 14-18: -$55.77). Chi cho SHORT.")
        if smc == "BULLISH_TREND" and direction == -1:
            if has_fresh_sweep or bias == "BEARISH_REVERSAL":
                return TacticalLessonVerdict(
                    lesson="SMC_ALIGN", allowed=True, direction_bias=0, penalty=0.3,
                    rationale="SMC BULLISH_TREND nhung co sweep dao chieu giam moi: cho SHORT tham do (size -30%).")
            return TacticalLessonVerdict(
                lesson="SMC_ALIGN", allowed=False, direction_bias=1, penalty=1.0,
                rationale="🛡️ [SMC BULLISH_TREND] Cam SHORT nguoc cau truc. Chi cho LONG.")
        return TacticalLessonVerdict(
            lesson="SMC_ALIGN", allowed=True, direction_bias=0, penalty=0.0,
            rationale=f"SMC {smc}: thuan/hop le huong {direction}.")


class FeeAwareEntry:
    """Bai hoc #3 (fee/lai-gop 48.2%): Chan lenh ma phi an het lai ky vong."""

    def __init__(self, taker_fee_rate: float = 0.0005, max_fee_ratio: float = 0.35):
        self.taker_fee_rate = taker_fee_rate
        self.max_fee_ratio = max_fee_ratio

    def evaluate(self, expected_reward: float, notional: float) -> TacticalLessonVerdict:
        roundtrip = notional * self.taker_fee_rate * 2.0
        if expected_reward <= 0:
            return TacticalLessonVerdict(
                lesson="FEE_AWARE", allowed=False, direction_bias=0, penalty=1.0,
                rationale="Khong co reward ky vong duong.")
        ratio = roundtrip / expected_reward
        if ratio > self.max_fee_ratio:
            return TacticalLessonVerdict(
                lesson="FEE_AWARE", allowed=False, direction_bias=0, penalty=1.0,
                rationale=f"🛡️ [FEE {ratio*100:.0f}% > {self.max_fee_ratio*100:.0f}%] Phi roundtrip ${roundtrip:.2f} an het lai ky vong ${expected_reward:.2f} (bai hoc 1077674: +$0.11 -> -$0.0008).")
        if ratio > 0.20:
            return TacticalLessonVerdict(
                lesson="FEE_AWARE", allowed=True, direction_bias=0, penalty=0.5,
                rationale=f"Phi {ratio*100:.0f}% cao: giam 50% size.")
        return TacticalLessonVerdict(
            lesson="FEE_AWARE", allowed=True, direction_bias=0, penalty=0.0,
            rationale=f"Phi {ratio*100:.1f}% chap nhan duoc.")


class TacticalLessonsEngine:
    """Gop 3 bai hoc thanh 1 cua trong pipeline (Stage 2)."""

    def __init__(self):
        self.hurst_filter = HurstRegimeFilter()
        self.smc_align = SMCTrendAlignment()
        self.fee_aware = FeeAwareEntry()

    def evaluate(self, hurst: float, smc_structure: str, direction: int,
                 proposed_type: str, expected_reward: float,
                 notional: float, smc_bias: str = "",
                 has_fresh_sweep: bool = False) -> Dict[str, Any]:
        verdicts = [
            self.hurst_filter.evaluate(hurst, proposed_type),
            self.smc_align.evaluate(smc_structure, direction, smc_bias, has_fresh_sweep),
            self.fee_aware.evaluate(expected_reward, notional),
        ]
        blocked = [v for v in verdicts if not v.allowed]
        penalty = max([v.penalty for v in verdicts] or [0.0])
        return {
            "allowed": len(blocked) == 0,
            "penalty": penalty,
            "blocked_lessons": [v.lesson for v in blocked],
            "rationale": " | ".join(v.rationale for v in verdicts),
            "verdicts": verdicts,
        }
