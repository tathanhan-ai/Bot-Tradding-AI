"""
Verification Tests for Hummingbot, Condor, Jesse AI, and LLM_trader Modules:
1. Episodic Trade Memory Bank & Anti-Hallucination Guardrails
2. Jesse Expectancy & Half-Kelly Capital Allocation Engine
3. Hummingbot Avellaneda-Stoikov Inventory Skew & Order Refresh Tolerance
4. Condor OODA Integrated Loop Test
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from strategy.memory_reasoning_engine import (
    EpisodicTradeMemoryBank,
    DeterministicValidationGuardrail
)
from risk.jesse_expectancy_engine import JesseExpectancyEngine
from strategy.hummingbot_inventory_skew import HummingbotInventorySkewEngine


def test_episodic_trade_memory():
    print("=== TEST 1: EPISODIC TRADE MEMORY (LLM_TRADER) ===")
    memory = EpisodicTradeMemoryBank()

    # Record a painful loss: Longed at Premium VWAP into Bearish Absorption
    memory.record_trade_outcome(
        trade_id=1,
        direction=1,
        entry_price=81500.0,
        indicators={"rsi": 68.0},
        of_data={"absorption_signal": "BEAR_ABSORPTION", "delta_momentum": "BEARISH"},
        smc_data={"structure": "RANGING"},
        vwap_data={"vwap_status": "PREMIUM_EXPENSIVE"},
        net_pnl=-12.50,
        exit_reason="STOP_LOSS 🛑"
    )

    # Candidate trade A: Exactly the same blunder! (Longing at Premium with Bear Absorption)
    check_a = memory.query_similarity_against_losses(
        candidate_direction=1,
        candidate_rsi=67.0,
        candidate_vwap_status="PREMIUM_EXPENSIVE",
        candidate_absorption="BEAR_ABSORPTION"
    )
    assert check_a.is_safe is False, "Expected BLOCKED on repeated mistake!"
    assert check_a.warning_level == "BLOCKED"
    assert check_a.similarity_score >= 0.70
    print(f"  [Pass] Repeated error successfully VETOED: {check_a.lesson_learned}")

    # Candidate trade B: A healthy dip buy (Longing at Discount with Bull Absorption)
    check_b = memory.query_similarity_against_losses(
        candidate_direction=1,
        candidate_rsi=28.0,
        candidate_vwap_status="DISCOUNT_CHEAP",
        candidate_absorption="BULL_ABSORPTION"
    )
    assert check_b.is_safe is True
    print(f"  [Pass] Distinct healthy setup approved: {check_b.lesson_learned}")


def test_deterministic_guardrail():
    print("\n=== TEST 2: DETERMINISTIC VALIDATION (ANTI-HALLUCINATION) ===")
    guardrail = DeterministicValidationGuardrail()

    # Hallucination test: AI says "STRONG_TRENDING_BULL" but ADX is only 14.0 (flat chop)
    result = guardrail.validate(
        claimed_regime="STRONG_TRENDING_BULL",
        claimed_confidence=90.0,
        adx=14.0,
        atr=200.0,
        rsi=48.0
    )
    assert result.corrected_regime == "RANGING_SIDEWAY", f"Expected RANGING_SIDEWAY, got {result.corrected_regime}"
    assert result.confidence_penalty > 0
    print(f"  [Pass] AI Hallucination corrected: {result.validation_notes}")


def test_jesse_expectancy_and_kelly():
    print("\n=== TEST 3: JESSE EXPECTANCY & HALF-KELLY ENGINE ===")
    jesse = JesseExpectancyEngine()

    # Simulate 5 trades: 3 wins (+15, +20, +18) and 2 losses (-10, -8)
    # Win rate = 60%, Avg win = 17.67, Avg loss = 9.0 -> Positive edge!
    jesse.record_trade(15.0)
    jesse.record_trade(-10.0)
    jesse.record_trade(20.0)
    jesse.record_trade(-8.0)
    jesse.record_trade(18.0)

    metrics = jesse.compute_metrics()
    assert metrics.win_rate_pct == 60.0
    assert metrics.expectancy_usdt > 0.0, f"Expected positive expectancy, got {metrics.expectancy_usdt}"
    assert metrics.edge_status == "POSITIVE_EDGE 🟢"
    assert metrics.kelly_fraction_pct > 0.0
    print(f"  [Pass] Jesse Metrics: WinRate={metrics.win_rate_pct}% | Expectancy=+${metrics.expectancy_usdt}/trade | Half-Kelly={metrics.kelly_fraction_pct}%")

    opt_margin = jesse.calculate_optimal_margin(current_balance=1000.0)
    assert opt_margin >= 10.0
    print(f"  [Pass] Calculated Optimal Margin for $1,000 balance: ${opt_margin:,.2f}")


def test_hummingbot_inventory_skew():
    print("\n=== TEST 4: HUMMINGBOT INVENTORY SKEW & REFRESH TOLERANCE ===")
    hb = HummingbotInventorySkewEngine(risk_aversion_gamma=0.15, refresh_tolerance_pct=0.08)

    # 1. Holding Long position -> Reservation price must shift DOWNWARD
    long_pos = {"direction": 1, "margin": 150.0}
    skew_long = hb.calculate_reservation_price(mid_price=81000.0, current_position=long_pos, atr=300.0, total_balance=1000.0)
    assert skew_long.reservation_price < 81000.0, "Long position must shift reservation price lower!"
    print(f"  [Pass] Holding Long: Mid=$81,000 -> Reservation Price=${skew_long.reservation_price:,.2f} ({skew_long.skew_direction})")

    # 2. Holding Short position -> Reservation price must shift UPWARD
    short_pos = {"direction": -1, "margin": 150.0}
    skew_short = hb.calculate_reservation_price(mid_price=81000.0, current_position=short_pos, atr=300.0, total_balance=1000.0)
    assert skew_short.reservation_price > 81000.0, "Short position must shift reservation price higher!"
    print(f"  [Pass] Holding Short: Mid=$81,000 -> Reservation Price=${skew_short.reservation_price:,.2f} ({skew_short.skew_direction})")

    # 3. Order refresh tolerance:
    # Price moves by only 0.03% (< 0.08% tolerance) -> KEEP order in book
    should_refresh, drift = hb.should_refresh_order(current_order_price=81000.0, optimal_new_price=81024.0)
    assert should_refresh is False, "Small drift within tolerance must NOT refresh order!"
    print(f"  [Pass] Drift {drift}% <= 0.08% tolerance -> Order preserved in book (Zero API spam)")

    # Price moves by 0.15% (> 0.08% tolerance) -> REFRESH order
    should_refresh, drift = hb.should_refresh_order(current_order_price=81000.0, optimal_new_price=81130.0)
    assert should_refresh is True
    print(f"  [Pass] Drift {drift}% > 0.08% tolerance -> Order refreshed successfully")


if __name__ == "__main__":
    test_episodic_trade_memory()
    test_deterministic_guardrail()
    test_jesse_expectancy_and_kelly()
    test_hummingbot_inventory_skew()
    print("\n🎉 ALL 4 QUANT ECOSYSTEM TESTS PASSED PERFECTLY! 🎉")
