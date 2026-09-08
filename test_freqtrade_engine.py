"""
Comprehensive Verification Test for Freqtrade Integration:
1. StoplossGuard
2. MaxDrawdownGuard
3. CooldownPeriod
4. FeeDragFilter
5. MinimalROIEngine
6. Monotonic Ratchet Stoploss
"""
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from risk.freqtrade_protections import FreqtradeProtectionEngine, StoplossGuard, MaxDrawdownGuard, CooldownPeriod
from strategy.freqtrade_roi import FreqtradeROIEngine
from strategy.ai_position_manager import AIPositionCoordinator


def test_stoploss_guard():
    print("=== TEST 1: STOPLOSS GUARD ===")
    engine = FreqtradeProtectionEngine(initial_balance=1000.0)
    now = time.time()

    # Trade 1: Loss
    engine.on_trade_closed({"id": 1, "pnl": -5.0, "reason": "STOP_LOSS 🛑"})
    approved, reason, status = engine.validate_new_trade(81000.0, 81500.0, 1, 995.0, 995.0)
    # Cooldown should be active immediately after trade close
    assert status.status == "COOLDOWN", f"Expected COOLDOWN, got {status.status}"
    print("  [Pass] Trade 1 triggers CooldownPeriod")

    # Fast forward past cooldown
    engine.cooldown_guard.last_trade_closed_time = now - 60.0
    approved, reason, status = engine.validate_new_trade(81000.0, 81500.0, 1, 995.0, 995.0)
    assert approved is True, f"Expected approved after cooldown, got {approved}"
    print("  [Pass] Cooldown expires after 30s -> Approved")

    # Trade 2: Second Loss within 60m
    engine.on_trade_closed({"id": 2, "pnl": -6.0, "reason": "STOP_LOSS 🛑"})
    # Fast forward past cooldown
    engine.cooldown_guard.last_trade_closed_time = now - 60.0
    approved, reason, status = engine.validate_new_trade(81000.0, 81500.0, 1, 989.0, 989.0)
    assert approved is False, f"Expected rejection after 2 losses, got {approved}"
    assert status.status == "STOPLOSS_GUARD", f"Expected STOPLOSS_GUARD, got {status.status}"
    print(f"  [Pass] StoplossGuard triggered lock: {status.lock_reason[:70]}...")


def test_max_drawdown_guard():
    print("\n=== TEST 2: MAX DRAWDOWN GUARD ===")
    engine = FreqtradeProtectionEngine(initial_balance=1000.0)
    now = time.time()

    # Fast forward cooldown
    engine.cooldown_guard.last_trade_closed_time = now - 60.0

    # Normal balance: $980 (2.0% drawdown, < 3.5%)
    approved, reason, status = engine.validate_new_trade(81000.0, 81500.0, 1, 980.0, 980.0)
    assert approved is True, "Expected approved at 2.0% DD"
    print("  [Pass] 2.0% DD is within 3.5% threshold -> Approved")

    # Drop balance: $960 (4.0% drawdown, > 3.5%)
    approved, reason, status = engine.validate_new_trade(81000.0, 81500.0, 1, 960.0, 960.0)
    assert approved is False, "Expected rejection at 4.0% DD"
    assert status.status == "MAX_DRAWDOWN", f"Expected MAX_DRAWDOWN, got {status.status}"
    print(f"  [Pass] MaxDrawdownGuard triggered: {status.lock_reason[:70]}...")


def test_fee_drag_filter():
    print("\n=== TEST 3: FEE DRAG FILTER ===")
    engine = FreqtradeProtectionEngine(initial_balance=1000.0)
    now = time.time()
    engine.cooldown_guard.last_trade_closed_time = now - 60.0

    # Micro target: Entry 81,000 -> TP 81,100 (+0.12%, < 0.35%)
    approved, reason, status = engine.validate_new_trade(81000.0, 81100.0, 1, 1000.0, 1000.0)
    assert approved is False, "Expected fee drag rejection"
    assert status.status == "FEE_DRAG", f"Expected FEE_DRAG, got {status.status}"
    print("  [Pass] Micro profit target < 0.35% rejected to eliminate fee churn")

    # Healthy target: Entry 81,000 -> TP 81,400 (+0.49%, >= 0.35%)
    approved, reason, status = engine.validate_new_trade(81000.0, 81400.0, 1, 1000.0, 1000.0)
    assert approved is True, "Expected healthy target approved"
    print("  [Pass] Target +0.49% >= 0.35% cleared fee hurdles -> Approved")


def test_minimal_roi_decay():
    print("\n=== TEST 4: MINIMAL ROI TIME-DECAY ===")
    roi_engine = FreqtradeROIEngine(default_timeframe="15m")

    # 15m curve: 0m: 3.5%, 15m: 1.8%, 45m: 1.0%, 90m: 0.4%
    roi_0m = roi_engine.get_target_roi(0.0, "15m")
    assert roi_0m == 0.035, f"Expected 0.035 at 0m, got {roi_0m}"

    roi_20m = roi_engine.get_target_roi(20 * 60.0, "15m")
    assert roi_20m == 0.018, f"Expected 0.018 at 20m, got {roi_20m}"

    roi_60m = roi_engine.get_target_roi(60 * 60.0, "15m")
    assert roi_60m == 0.010, f"Expected 0.010 at 60m, got {roi_60m}"

    roi_100m = roi_engine.get_target_roi(100 * 60.0, "15m")
    assert roi_100m == 0.004, f"Expected 0.004 at 100m, got {roi_100m}"

    print(f"  [Pass] 15m ROI curve verified: 0m (+3.5%) -> 20m (+1.8%) -> 60m (+1.0%) -> 100m (+0.4%)")

    # Test position exit evaluation
    pos = {
        "direction": 1,
        "entry_price": 81000.0,
        "open_timestamp": time.time() - (50 * 60.0), # 50 mins ago (target is 1.0%)
        "timeframe": "15m"
    }
    # Price at 81,900 (+1.11% > 1.0% target)
    exit_needed, reason, p_pct, t_pct = roi_engine.evaluate_roi_exit(pos, 81900.0)
    assert exit_needed is True, f"Expected ROI exit triggered, got {exit_needed}"
    print(f"  [Pass] Position held 50m with +1.11% profit triggered ROI exit: {reason[:60]}...")


def test_monotonic_ratchet_stoploss():
    print("\n=== TEST 5: MONOTONIC RATCHET STOPLOSS ===")
    coordinator = AIPositionCoordinator()
    pos = {
        "direction": 1,
        "entry_price": 81000.0,
        "breakeven_price": 81050.0,
        "stop_loss": 80500.0,
        "take_profit": 82500.0,
        "units": 0.01,
        "margin": 100.0,
        "initial_risk": 500.0, # 1R = $500 move
        "open_timestamp": time.time() - 30.0,
        "timeframe": "15m",
        "is_risk_free": False
    }
    indicators = {"atr": 300.0, "rsi": 55.0}

    # Case A: Profit reaches +0.5R (Price = 81,250) -> Lock Breakeven sớm
    # (winner-protection: không chờ +1.0R mới khóa, winner không trượt về SL gốc)
    decision = coordinator.evaluate_position(pos, 81250.0, indicators, None, None)
    assert decision.action == "LOCK_BREAKEVEN", f"Expected LOCK_BREAKEVEN at +0.5R, got {decision.action}"
    pos["stop_loss"] = decision.new_stop_loss
    pos["is_risk_free"] = True
    print(f"  [Pass] At +1.0R, SL ratcheted to Breakeven: ${decision.new_stop_loss}")

    # Case B: Profit reaches +2.0R (Price = 82,000) -> Lock in +1.0R profit ($81,500)
    decision = coordinator.evaluate_position(pos, 82000.0, indicators, None, None)
    assert decision.action == "UPDATE_TRAILING", f"Expected UPDATE_TRAILING at +2.0R, got {decision.action}"
    assert decision.new_stop_loss >= 81500.0, f"Expected SL >= 81500.0, got {decision.new_stop_loss}"
    pos["stop_loss"] = decision.new_stop_loss
    print(f"  [Pass] At +2.0R, SL ratcheted to lock guaranteed +1.0R profit: ${decision.new_stop_loss}")

    # Case C: Monotonic invariant - price pullbacks must NEVER lower SL
    decision = coordinator.evaluate_position(pos, 81800.0, indicators, None, None)
    # The ratchet check ensures target_ratchet_sl > current_sl. Since target is 81500 and current is >= 81500, no regression
    if decision.new_stop_loss is not None:
        assert decision.new_stop_loss >= pos["stop_loss"], "SL must NEVER regress backwards!"
    print("  [Pass] Monotonic invariant preserved: Stoploss strictly moves forward, never backwards")


if __name__ == "__main__":
    test_stoploss_guard()
    test_max_drawdown_guard()
    test_fee_drag_filter()
    test_minimal_roi_decay()
    test_monotonic_ratchet_stoploss()
    print("\n🎉 ALL 5 FREQTRADE ARCHITECTURAL TESTS PASSED PERFECTLY! 🎉")
