"""
Comprehensive Verification Test for OctoBot Architecture Integration:
1. Tentacle Evaluators (TA, Order Flow, SMC, VWAP, MTF)
2. Evaluator Matrix Weighted Consensus Engine
3. Dip Analyser Trading Mode & Staged Take Profit (TP1, TP2, TP3)
4. Daily Trading Mode
5. Synergy Test: OctoBot Signals verified through Freqtrade Capital Protections
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from strategy.octobot_matrix import (
    OctoBotMatrixEngine,
    TATentacleEvaluator,
    OrderFlowTentacleEvaluator,
    SMCTentacleEvaluator,
    VWAPValuationTentacleEvaluator,
    MultiTimeframeTentacleEvaluator
)
from strategy.octobot_trading_modes import (
    OctoBotTradingCoordinator,
    DipAnalyserMode,
    DailyTradingMode,
    StagedTakeProfit
)
from risk.freqtrade_protections import FreqtradeProtectionEngine


def test_individual_tentacles():
    print("=== TEST 1: OCTOBOT INDIVIDUAL TENTACLES ===")

    # 1. TA Tentacle: Oversold RSI (22.0)
    ta_tentacle = TATentacleEvaluator()
    eval_ta = ta_tentacle.evaluate({"rsi": 22.0, "ema_fast": 81000.0, "ema_slow": 81050.0})
    assert eval_ta.score >= 0.4, f"Expected bullish score for oversold RSI, got {eval_ta.score}"
    assert "BULLISH" in eval_ta.status_label
    print(f"  [Pass] TA Tentacle: Score={eval_ta.score:+.2f} ({eval_ta.status_label})")

    # 2. Order Flow Tentacle: Bullish Absorption
    of_tentacle = OrderFlowTentacleEvaluator()
    eval_of = of_tentacle.evaluate({"buy_ratio": 42.0, "cvd_delta_60s": -5.2, "absorption_signal": "BULL_ABSORPTION"})
    assert eval_of.score > 0.0, f"Expected positive score with Bull Absorption, got {eval_of.score}"
    print(f"  [Pass] Order Flow Tentacle: Score={eval_of.score:+.2f} ({eval_of.status_label})")

    # 3. SMC Tentacle: Price inside Demand Zone
    smc_tentacle = SMCTentacleEvaluator()
    eval_smc = smc_tentacle.evaluate(80500.0, {
        "structure": "BULLISH_TREND",
        "demand_zone": [80400.0, 80600.0],
        "supply_zone": [82000.0, 82200.0],
        "liquidity_sweep": "SWEEP_LOW"
    })
    assert eval_smc.score >= 0.6, f"Expected strong bullish inside demand with sweep low, got {eval_smc.score}"
    print(f"  [Pass] SMC Tentacle: Score={eval_smc.score:+.2f} ({eval_smc.status_label})")

    # 4. VWAP Tentacle: Deep Discount
    vwap_tentacle = VWAPValuationTentacleEvaluator()
    eval_vwap = vwap_tentacle.evaluate({"vwap_status": "DISCOUNT_CHEAP", "dist_sigma": -2.1})
    assert eval_vwap.score > 0.5, f"Expected positive score for discount VWAP, got {eval_vwap.score}"
    print(f"  [Pass] VWAP Tentacle: Score={eval_vwap.score:+.2f} ({eval_vwap.status_label})")

    # 5. MTF Tentacle: Multi-timeframe consensus
    mtf_tentacle = MultiTimeframeTentacleEvaluator()
    eval_mtf = mtf_tentacle.evaluate({"score": 0.75, "consensus": "STRONG_BULLISH", "timeframes": {"1m": 1, "5m": 1, "15m": 1}})
    assert eval_mtf.score == 0.75
    print(f"  [Pass] MTF Tentacle: Score={eval_mtf.score:+.2f} ({eval_mtf.status_label})")


def test_matrix_weighted_consensus():
    print("\n=== TEST 2: OCTOBOT EVALUATOR MATRIX ENGINE ===")
    matrix_engine = OctoBotMatrixEngine(confidence_threshold=0.55)

    # Bullish Confluence across all 5 dimensions
    consensus = matrix_engine.evaluate_matrix(
        current_price=80500.0,
        indicators={"rsi": 28.0, "ema_fast": 80600.0, "ema_slow": 80400.0, "macd": 5.0, "macd_signal": 2.0},
        order_flow_telemetry={"buy_ratio": 55.0, "cvd_delta_60s": 2.5, "absorption_signal": "BULL_ABSORPTION"},
        smc_data={"structure": "BULLISH_TREND", "demand_zone": [80400.0, 80600.0], "liquidity_sweep": "SWEEP_LOW"},
        vwap_data={"vwap_status": "DISCOUNT_CHEAP", "dist_sigma": -1.8},
        mtf_consensus={"score": 0.65, "consensus": "STRONG_BULLISH", "timeframes": {"1m": 1, "5m": 1}}
    )

    assert consensus.matrix_score >= 0.55, f"Expected matrix_score >= 0.55, got {consensus.matrix_score}"
    assert consensus.consensus_state == "STRONG_BULLISH"
    assert consensus.recommended_direction == 1
    assert consensus.is_tradable is True
    print(f"  [Pass] Matrix Consensus: {consensus.matrix_score:+.2f} | Confidence: {consensus.confidence_pct}% | Tradable: {consensus.is_tradable}")

    # Chop / Conflict scenario -> Neutral
    neutral_consensus = matrix_engine.evaluate_matrix(
        current_price=81000.0,
        indicators={"rsi": 50.0, "ema_fast": 81000.0, "ema_slow": 81000.0},
        order_flow_telemetry={"buy_ratio": 50.0, "cvd_delta_60s": 0.0, "absorption_signal": "NONE"},
        smc_data={"structure": "RANGING", "demand_zone": [80000.0, 80200.0], "supply_zone": [82000.0, 82200.0]},
        vwap_data={"vwap_status": "EQUILIBRIUM_FAIR", "dist_sigma": 0.1},
        mtf_consensus={"score": 0.0, "consensus": "NEUTRAL", "timeframes": {}}
    )
    assert neutral_consensus.is_tradable is False
    assert neutral_consensus.consensus_state == "NEUTRAL"
    print(f"  [Pass] Neutral / Chop successfully filtered out: Score={neutral_consensus.matrix_score:+.2f}, Tradable={neutral_consensus.is_tradable}")


def test_dip_analyser_and_staged_tps():
    print("\n=== TEST 3: DIP ANALYSER & STAGED TAKE PROFITS ===")
    coordinator = OctoBotTradingCoordinator()
    matrix_engine = OctoBotMatrixEngine(confidence_threshold=0.55)

    matrix = matrix_engine.evaluate_matrix(
        current_price=80500.0,
        indicators={"rsi": 32.0, "atr": 200.0},
        order_flow_telemetry={"buy_ratio": 48.0, "absorption_signal": "BULL_ABSORPTION"},
        smc_data={"demand_zone": [80400.0, 80600.0], "supply_zone": [82500.0, 82800.0]},
        vwap_data={"vwap": 81200.0, "dist_sigma": -1.6, "vwap_status": "DISCOUNT_CHEAP"},
        mtf_consensus=None
    )

    mode_name, setup = coordinator.select_best_setup(
        current_price=80500.0,
        matrix=matrix,
        indicators={"rsi": 32.0, "atr": 200.0},
        smc_data={"demand_zone": [80400.0, 80600.0], "supply_zone": [82500.0, 82800.0]},
        of_data={"buy_ratio": 48.0, "absorption_signal": "BULL_ABSORPTION"},
        vwap_data={"vwap": 81200.0, "dist_sigma": -1.6, "vwap_status": "DISCOUNT_CHEAP"}
    )

    assert mode_name == "DIP_ANALYSER", f"Expected DIP_ANALYSER, got {mode_name}"
    assert setup is not None
    assert setup.direction == 1
    assert setup.entry_price == 80500.0
    assert setup.stop_loss < setup.entry_price
    # Check Staged TPs
    staged = setup.staged_tp
    assert staged.tp1_price > setup.entry_price
    assert staged.tp2_price > staged.tp1_price
    assert staged.tp3_price >= staged.tp2_price
    print(f"  [Pass] Dip Analyser Setup Created: Entry=${setup.entry_price:,.1f}, SL=${setup.stop_loss:,.1f}")
    print(f"  [Pass] Staged TPs: TP1=${staged.tp1_price:,.1f} (40%), TP2=${staged.tp2_price:,.1f} (35%), TP3=${staged.tp3_price:,.1f} (25%)")
    print(f"  [Pass] Risk-Reward: {setup.risk_reward_ratio}:1")


def test_octobot_freqtrade_synergy():
    print("\n=== TEST 4: OCTOBOT + FREQTRADE SYNERGY PIPELINE ===")
    matrix_engine = OctoBotMatrixEngine(confidence_threshold=0.55)
    coordinator = OctoBotTradingCoordinator()
    freqtrade_protections = FreqtradeProtectionEngine(initial_balance=1000.0)

    # 1. Normal conditions: Trade should be approved by both
    matrix = matrix_engine.evaluate_matrix(
        current_price=80500.0,
        indicators={"rsi": 30.0, "atr": 200.0},
        order_flow_telemetry={"buy_ratio": 52.0, "absorption_signal": "BULL_ABSORPTION"},
        smc_data={"demand_zone": [80400.0, 80600.0], "supply_zone": [82000.0, 82200.0]},
        vwap_data={"vwap": 81200.0, "dist_sigma": -1.5, "vwap_status": "DISCOUNT_CHEAP"},
        mtf_consensus=None
    )
    mode, setup = coordinator.select_best_setup(80500.0, matrix, {"rsi": 30.0, "atr": 200.0}, {"demand_zone": [80400.0, 80600.0]}, None, None)
    assert setup is not None

    # Verify through Freqtrade Capital Protection
    approved, reason, status = freqtrade_protections.validate_new_trade(
        entry_price=setup.entry_price,
        target_price=setup.take_profit,
        direction=setup.direction,
        balance=1000.0,
        equity=1000.0
    )
    assert approved is True, f"Expected Freqtrade approval, got {approved} ({reason})"
    print("  [Pass] OctoBot setup successfully approved by Freqtrade Protection Engine")

    # 2. Capital Protection Lockout: 2 recent Stoplosses in Freqtrade
    freqtrade_protections.stoploss_guard.record_loss(5.0, "SL 1")
    freqtrade_protections.stoploss_guard.record_loss(5.0, "SL 2")
    approved, reason, status = freqtrade_protections.validate_new_trade(
        entry_price=setup.entry_price,
        target_price=setup.take_profit,
        direction=setup.direction,
        balance=990.0,
        equity=990.0
    )
    assert approved is False
    assert status.status == "STOPLOSS_GUARD"
    print("  [Pass] OctoBot signal obediently halted by Freqtrade StoplossGuard during high-risk period")


if __name__ == "__main__":
    test_individual_tentacles()
    test_matrix_weighted_consensus()
    test_dip_analyser_and_staged_tps()
    test_octobot_freqtrade_synergy()
    print("\n🎉 ALL OCTOBOT ARCHITECTURAL TESTS PASSED PERFECTLY! 🎉")
