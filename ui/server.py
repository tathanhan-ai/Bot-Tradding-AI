"""
FastAPI Server for Binance Futures Trading Admin Dashboard
Ultra-reliable real-time market data engine with sub-second price polling and WebSocket push.
"""
import asyncio
import json
import sys
import time
import urllib.request
import uuid
import math
import threading
from copy import deepcopy
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

TF_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
    "1M": 2592000,
}

# Ensure UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn
import pandas as pd

from config.settings import DEFAULT_STRATEGY, DEFAULT_RISK, StrategyConfig, RiskConfig, DEFAULT_SYSTEM
from data.fetcher import BinanceDataFetcher
from strategy.mtf_trend_atr import MTFTrendATRStrategy
from strategy.ai_brain import AIQuantBrain, AIRegimeVerdict
from strategy.ai_position_manager import AIPositionCoordinator, AIPositionDecision
from strategy.ensemble_strategy import EnsembleCoordinator, EnsembleResult
from risk.risk_manager import FuturesRiskManager
from risk.dynamic_leverage import DynamicLeverageEngine, LeverageAdvice
from risk.order_manager import OrderQueueManager, FuturesOrder
from risk.fee_and_spread_engine import SpreadFeeEngine, BINANCE_VIP_TIERS, MEXC_VIP_TIERS
from risk.ai_order_researcher import AIOrderResearcher, AIOrderResearchResult
from risk.freqtrade_protections import FreqtradeProtectionEngine, ProtectionStatus
from strategy.freqtrade_roi import FreqtradeROIEngine
from data.binance_ws_stream import BinanceFuturesWebSocketEngine
from strategy.order_flow_cvd import OrderFlowCVDEngine, OrderFlowVerdict
from strategy.octobot_matrix import OctoBotMatrixEngine, MatrixConsensus
from strategy.octobot_trading_modes import OctoBotTradingCoordinator, OctoBotTradeSetup
from strategy.memory_reasoning_engine import EpisodicTradeMemoryBank, DeterministicValidationGuardrail, TradeContextProfile
from risk.jesse_expectancy_engine import JesseExpectancyEngine
from strategy.hummingbot_inventory_skew import HummingbotInventorySkewEngine
from data.persistent_storage import PersistentStorageManager
from data.binance_api_manager import BinanceAPIManager
from data.mexc_api_manager import MexcAPIManager
from strategy.multi_candle_patterns import MultiTimeframeCandleStrategyEngine, MTFCandleConfluenceResult
from risk.monthly_target_governor import MonthlyTargetGovernor, MonthlyGovernorStatus
from strategy.ai_model_copilot import AIModelCopilot, AICopilotVerdict
from strategy.visual_hft_microstructure import VisualHFTMicrostructureEngine, VisualHFTMetrics
from strategy.vibe_alpha_zoo import VibeAlphaZooEngine, VibeAlphaMetrics
from strategy.vibe_swarm_council import VibeSwarmCouncil, SwarmCouncilVerdict
from strategy.shadow_account import ShadowAccountAnalyzer, BehavioralBiasReport
from trading.pipeline import CandidateOrder, MarketSnapshot, SevenStagePipeline, StageOutcome, apply_closed_kline, candle_end
from dataclasses import asdict
from trading.execution import ExecutionLifecycle

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Binance Futures Quant Desk")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"

connected_clients: Set[WebSocket] = set()


def serialized_action(fn):
    @wraps(fn)
    def run(*args, **kwargs):
        # ponytail: single-account lock; shard only for a multi-account service.
        with state.decision_lock:
            return fn(*args, **kwargs)
    return run


class LiveTradingState:
    def __init__(self, symbol: str = "BTCUSDT", balance: float = 5000.0, storage=None):
        self.symbol = symbol
        self.initial_balance = balance
        self.current_balance = balance
        self.is_running = True
        self.decision_lock = threading.RLock()
        self.market_lock = threading.RLock()
        self.last_history_repair_at = 0.0
        self.live_price = 0.0
        self.price_history: List[float] = []

        self.strategy_config = StrategyConfig(symbol=symbol)
        self.risk_config = RiskConfig(initial_balance=balance)
        self.strategy = MTFTrendATRStrategy(self.strategy_config)
        self.risk_manager = FuturesRiskManager(self.risk_config)
        self.fetcher = BinanceDataFetcher()

        self.ai_brain = AIQuantBrain()
        self.ai_verdict: Optional[AIRegimeVerdict] = None
        self.is_grid_active = False

        # Order Queue Manager (Market & Limit Orders)
        self.order_manager = OrderQueueManager()

        # Dynamic Leverage Strategy
        self.leverage_engine = DynamicLeverageEngine()
        self.leverage_mode = "AI_AUTO"  # 'AI_AUTO' or 'MANUAL'
        self.manual_leverage = 3
        self.leverage_advice: Optional[LeverageAdvice] = None

        self.storage = storage or PersistentStorageManager()
        saved_settings = self.storage.get_all_settings()
        api_key = str(saved_settings.get("api_key", "")).strip()
        api_secret = str(saved_settings.get("api_secret", "")).strip()

        def _as_bool(val, d=False):
            if val is None:
                return d
            if isinstance(val, bool):
                return val
            if isinstance(val, str):
                return val.strip().lower() in ("true", "1", "yes", "on")
            return bool(val)

        is_testnet = _as_bool(saved_settings.get("is_testnet"), False)
        is_live = _as_bool(saved_settings.get("is_live_enabled"), False)
        vip_tier = saved_settings.get("vip_tier", "VIP_0")
        use_bnb = _as_bool(saved_settings.get("use_bnb_discount"), False)
        custom_maker = saved_settings.get("custom_maker_fee")
        custom_taker = saved_settings.get("custom_taker_fee")

        # Multi-Exchange Support (Binance Futures & MEXC Futures)
        self.active_exchange = str(saved_settings.get("active_exchange", "binance")).strip().lower()
        if self.active_exchange not in ("binance", "mexc"):
            self.active_exchange = "binance"

        # Bid-Ask Spread & Fee Engine (Binance / MEXC Standard VIP Tiers & Fee Discounts)
        self.fee_engine = SpreadFeeEngine(vip_tier=vip_tier, use_bnb_discount=use_bnb, exchange=self.active_exchange)
        if custom_maker is not None and custom_taker is not None:
            self.fee_engine.set_custom_rates(float(custom_maker), float(custom_taker))

        mexc_key = str(saved_settings.get("mexc_api_key", "")).strip()
        mexc_secret = str(saved_settings.get("mexc_api_secret", "")).strip()
        mexc_base_url = str(saved_settings.get("mexc_base_url", "https://contract.mexc.co")).strip()
        if "contract.mexc.com" in mexc_base_url or "api.mexc.com" in mexc_base_url:
            mexc_base_url = "https://contract.mexc.co"
        mexc_proxy_url = str(saved_settings.get("mexc_proxy_url", "")).strip()

        # Live Binance Futures API Manager (Mainnet & Testnet)
        self.binance_api = BinanceAPIManager(
            api_key=api_key,
            api_secret=api_secret,
            is_testnet=is_testnet,
            is_live_enabled=is_live
        )

        # Live MEXC Contract (Futures) API Manager
        self.mexc_api = MexcAPIManager(
            api_key=mexc_key,
            api_secret=mexc_secret,
            base_url=mexc_base_url,
            proxy_url=mexc_proxy_url,
            is_live_enabled=is_live
        )

        # AI Position Coordinator (Dynamic Exits, Breakeven Lock, Early TP/SL, Minimal ROI)
        self.ai_coordinator = AIPositionCoordinator()

        # Freqtrade-Grade Capital Protection Framework (StoplossGuard, MaxDrawdown, Cooldown)
        self.freqtrade_protections = FreqtradeProtectionEngine(initial_balance=balance)

        # OctoBot Tentacle Matrix & Trading Modes Framework
        self.octobot_matrix = OctoBotMatrixEngine(confidence_threshold=0.55)
        self.octobot_coordinator = OctoBotTradingCoordinator()
        self.octobot_consensus: Optional[MatrixConsensus] = None
        self.octobot_setup: Optional[OctoBotTradeSetup] = None
        self.active_trading_mode: str = "STAND_ASIDE"

        # Condor OODA & LLM_trader Memory Reasoning Engine
        self.trade_memory = EpisodicTradeMemoryBank(max_records=40)
        self.deterministic_guardrail = DeterministicValidationGuardrail()
        # Jesse AI Expectancy & Kelly Capital Allocation
        self.jesse_engine = JesseExpectancyEngine(default_margin_pct=10.0)
        # Hummingbot Inventory Skew & Reservation Price
        self.hummingbot_skew = HummingbotInventorySkewEngine(risk_aversion_gamma=0.15, refresh_tolerance_pct=0.08)

        # AI Multi-Strategy Ensemble Hub (Trend, Mean Reversion, Liquidity Sweep, Grid, Multi-Candle MTF)
        self.ensemble_coordinator = EnsembleCoordinator()
        self.ensemble_result: Optional[EnsembleResult] = None
        self.multi_candle_engine = MultiTimeframeCandleStrategyEngine()
        self.candle_confluence: Optional[MTFCandleConfluenceResult] = None

        # Real-time AI Order Execution Researcher
        self.ai_order_researcher = AIOrderResearcher()
        self.order_research: Optional[AIOrderResearchResult] = None

        # AI Model Direct Cognitive Co-pilot via 9Router (Port 8039)
        saved_ai_model = self.storage.get_setting("ai_copilot_model", "ag/gemini-3.8-flash-high")
        if saved_ai_model in ("gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "deepseek-r1", ""):
            saved_ai_model = "ag/gemini-3.8-flash-high"
        self.ai_copilot = AIModelCopilot(
            gateway_url="http://127.0.0.1:8039/v1",
            default_model=saved_ai_model
        )
        saved_instruction = str(self.storage.get_setting("ai_copilot_user_instruction", "")).strip()
        if saved_instruction:
            self.ai_copilot.set_user_instruction(saved_instruction)
        self.ai_copilot_verdict: Optional[AICopilotVerdict] = None
        self.ai_copilot_last_response_at = 0.0
        self.available_ai_models = [saved_ai_model]
        self.auto_grid_rotation: bool = True

        # Monthly Target Governor & Adaptive Capital Allocation
        target_pct = float(saved_settings.get("monthly_target_pct", 10.0))
        target_enabled = _as_bool(saved_settings.get("monthly_target_enabled"), True)
        compensate_def = _as_bool(saved_settings.get("monthly_compensate_deficit"), True)
        self.monthly_governor = MonthlyTargetGovernor(
            base_target_pct=target_pct,
            enabled=target_enabled,
            auto_compensate_deficit=compensate_def,
            storage=self.storage
        )

        self.active_timeframe = "15m"
        self.data_map: Dict[str, pd.DataFrame] = {}
        self.current_position: Optional[dict] = None
        self.trades: List[dict] = []
        self.total_fees = 0.0
        self.tick_count = 0
        self.last_auto_order_tick = 0
        self.last_auto_order_time = 0.0
        self.last_trade_closed_time = 0.0

        # Load persisted account state if available
        saved_state = self.storage.load_account_state()
        if saved_state:
            self.symbol = saved_state.get("symbol", symbol)
            self.initial_balance = saved_state.get("initial_balance", balance)
            self.current_balance = saved_state.get("current_balance", balance)
            self.total_fees = saved_state.get("total_fees", 0.0)
            self.is_running = saved_state.get("is_running", True)
            self.active_timeframe = saved_state.get("active_timeframe", "15m")
            self.leverage_mode = saved_state.get("leverage_mode", "AI_AUTO")
            self.manual_leverage = saved_state.get("manual_leverage", 3)
            self.current_position = saved_state.get("current_position")
            saved_peak = saved_state.get("peak_balance", self.current_balance)
            self.freqtrade_protections.max_drawdown_guard.peak_balance = saved_peak
            self.risk_manager.update_balance(self.current_balance)

        # Load persisted trades
        self.trades = self.storage.load_trades(limit=200)
        for t in self.trades:
            self.jesse_engine.record_trade(t["pnl"])
            self.risk_manager.ai_cro.record_trade_result(t["pnl"])

        # Load persisted trade memory records
        saved_memories = self.storage.load_trade_memory_records(limit=40)
        for m in saved_memories:
            rec = TradeContextProfile(
                trade_id=m["trade_id"],
                direction=m["direction"],
                entry_price=m["entry_price"],
                rsi=m["rsi"],
                cvd_momentum=m.get("delta_momentum", "BALANCED"),
                smc_structure=m.get("smc_structure", "RANGING"),
                vwap_status=m.get("vwap_status", "EQUILIBRIUM_FAIR"),
                net_pnl=m["net_pnl"],
                outcome="WIN" if m["net_pnl"] > 0 else "LOSS",
                absorption_signal=m.get("absorption_signal", "NONE"),
                failure_reason=m.get("lesson_learned", "")
            )
            self.trade_memory.memory_records.append(rec)

        print(f"📦 [SQLITE KHÔI PHỤC] Đã nạp {len(self.trades)} lệnh lịch sử, {len(self.trade_memory.memory_records)} bài học kinh nghiệm, số dư: ${self.current_balance:,.2f}", flush=True)

        self.funding_rate = 0.0001
        self.next_funding_time = 0
        self.last_funding_fetch_time = 0.0

        self.indicators = {
            "ema": None,
            "rsi": None,
            "atr": None,
            "adx": None,
        }

        # Real-time Binance Futures WebSocket Stream & Order Flow CVD
        self.ws_engine: Optional[BinanceFuturesWebSocketEngine] = None
        self.visual_hft = VisualHFTMicrostructureEngine(bucket_size_btc=10.0, num_buckets=30)
        self.order_flow_engine = OrderFlowCVDEngine(max_ticks=3000, window_seconds=60)
        self.order_flow_verdict: Optional[OrderFlowVerdict] = None
        self.ws_latency_ms: float = 0.0
        self.market_source_times: Dict[str, float] = {}
        self.latest_l2_bids: List[List[float]] = []
        self.latest_l2_asks: List[List[float]] = []
        self.last_decision_trace = None
        self.last_candidate: Optional[CandidateOrder] = None
        self.deterministic_validation = None
 
        # HKUDS Vibe-Trading Quantitative Framework (Alpha Zoo + Swarm Council + Shadow Account)
        self.vibe_alpha_zoo = VibeAlphaZooEngine()
        vibe_cfg = self.storage.get_vibe_config()
        self.vibe_swarm = VibeSwarmCouncil(
            gateway_url="http://127.0.0.1:8039/v1",
            default_model="ag/gemini-3.8-flash-high",
            min_votes_required=int(vibe_cfg.get("min_votes", 3)),
            enabled=bool(vibe_cfg.get("enabled", True))
        )
        self.vibe_swarm.update_config(
            macro_model=vibe_cfg.get("macro_model", "ag/gemini-3.8-flash-high"),
            quant_model=vibe_cfg.get("quant_model", "ag/gemini-3.8-flash-high"),
            risk_model=vibe_cfg.get("risk_model", "ag/gemini-3.8-flash-high"),
            exec_model=vibe_cfg.get("exec_model", "ag/gemini-3.8-flash-high")
        )
        self.shadow_account = ShadowAccountAnalyzer()
        self.execution_blocker = ""
        self.execution = ExecutionLifecycle(self)
        self.execution.restore()
        if self.trades:
            self.shadow_account.analyze_trade_history(self.trades, self.initial_balance)

    @property
    def active_exchange_api(self):
        return self.mexc_api if self.active_exchange == "mexc" else self.binance_api

    def init_ws_engine(self):
        """Initializes direct Binance Futures WebSocket stream callbacks"""
        def atomic_event(callback):
            def receive(*args):
                with self.market_lock:
                    return callback(*args)
            return receive

        @atomic_event
        def on_book_ticker(bid: float, ask: float, mid: float):
            self.fee_engine.update_book(bid, ask)
            self.market_source_times["book_ticker"] = time.time()
            self.on_tick(mid)

        @atomic_event
        def on_depth(bids: List[List[float]], asks: List[List[float]], event_time: int):
            self.latest_l2_bids = bids
            self.latest_l2_asks = asks
            self.fee_engine.update_book(bids[0][0], asks[0][0])
            self.on_tick((bids[0][0] + asks[0][0]) / 2)
            self.market_source_times["depth"] = (event_time - (self.ws_engine.clock_offset_ms or 0)) / 1000.0
            self.visual_hft.update_order_book(bids, asks)

        @atomic_event
        def on_agg_trade(price: float, qty: float, is_buyer_maker: bool, trade_time: int):
            local_time_ms = trade_time - (self.ws_engine.clock_offset_ms or 0)
            if not self.order_flow_engine.add_trade(price, qty, is_buyer_maker, local_time_ms):
                return
            t_sec = local_time_ms / 1000.0
            self.market_source_times["agg_trade"] = t_sec
            self.visual_hft.update_trade(price, qty, is_buyer_maker, timestamp=t_sec)

        @atomic_event
        def on_kline(k: dict):
            timeframe = k.get("timeframe", "1m")
            if apply_closed_kline(self.data_map, timeframe, k):
                self.market_source_times[f"kline_{timeframe}"] = time.time()
                print(f"✅ [BINANCE {timeframe.upper()} KLINE ĐÃ ĐÓNG] O:${k['open']:,.1f} H:${k['high']:,.1f} L:${k['low']:,.1f} C:${k['close']:,.1f} | Vol: {k['volume']:.2f} BTC", flush=True)

        def on_latency(latency_ms: float):
            self.ws_latency_ms = latency_ms

        self.ws_engine = BinanceFuturesWebSocketEngine(
            symbol=self.symbol,
            on_book_ticker=on_book_ticker,
            on_depth=on_depth,
            on_agg_trade=on_agg_trade,
            on_kline=on_kline,
            on_latency_update=on_latency,
            is_testnet=self.binance_api.is_live_enabled and self.binance_api.is_testnet,
        )

    def fetch_binance_funding(self):
        now = time.time()
        if now - self.last_funding_fetch_time < 30:
            return
        self.last_funding_fetch_time = now
        try:
            import urllib.request
            import json
            url = f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={self.symbol}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=2.5) as resp:
                data = json.loads(resp.read().decode())
                self.funding_rate = float(data.get("lastFundingRate", 0.0001))
                self.next_funding_time = int(data.get("nextFundingTime", 0))
        except Exception:
            pass

    def get_structure_df(self) -> Optional[pd.DataFrame]:
        df = self.data_map.get(self.active_timeframe)
        if df is None or df.empty:
            df = self.data_map.get("15m")
        return df

    def get_macro_df(self) -> Optional[pd.DataFrame]:
        macro_hierarchy = {
            "1m": "15m",
            "3m": "15m",
            "5m": "1h",
            "15m": "1h",
            "1h": "1w",
            "4h": "1w",
            "1d": "1w",
            "1w": "1M",
            "1M": "1M"
        }
        target_tf = macro_hierarchy.get(self.active_timeframe, "1h")
        df = self.data_map.get(target_tf)
        if df is None or df.empty:
            df = self.data_map.get("1h")
            if df is None or df.empty:
                df = self.data_map.get("15m")
        return df

    def update_indicators(self):
        """Continuously update indicators from the active timeframe dataframe"""
        df = self.get_structure_df()
        if df is not None and not df.empty:
            inds = self.ai_brain.calculate_indicators(df)
            self.indicators = {
                "ema": float(inds.get("ema50", 0.0)),
                "rsi": float(inds.get("rsi", 50.0)),
                "atr": float(inds.get("atr", self.live_price * 0.006)),
                "adx": float(inds.get("adx", 25.0)),
            }

    def initialize_history(self):
        # Fetch real, closed OHLCV only. Alpha is evaluated inside Stage 2.
        from concurrent.futures import ThreadPoolExecutor
        self.fetcher.base_url = "https://demo-fapi.binance.com" if self.binance_api.is_live_enabled and self.binance_api.is_testnet else "https://fapi.binance.com"
        intervals = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M")
        def fetch(tf):
            return tf, self.fetcher.fetch_klines(self.symbol, tf, 250, use_cache=False)
        with ThreadPoolExecutor(max_workers=4) as pool:
            for tf, frame in pool.map(fetch, intervals):
                self.data_map[tf] = frame
                self.market_source_times[f"kline_{tf}"] = time.time()
        self.live_price = float(self.data_map["1m"]["close"].iloc[-1])
        self.price_history = self.data_map["1m"]["close"].tail(50).tolist()
        self.update_indicators()

    def repair_closed_history(self, now=None):
        """Backfill missed closes outside the event loop without replacing newer WS bars."""
        fixed_now = now
        now = time.time() if now is None else now
        if now - self.last_history_repair_at < 15.0:
            return []
        snapshot = self.build_market_snapshot()
        issues = snapshot.freshness_issues(now)

        def frame_invalid(tf, problems):
            return any(problem in problems for problem in (
                f"insufficient closed {tf} history", f"forming/future {tf} candle",
                f"stale {tf} history", f"gap in {tf} closed candles", f"invalid {tf} OHLCV"))

        needed = [tf for tf in snapshot.context["required_frames"] if frame_invalid(tf, issues)]
        if any(issue.startswith(("missing kline_1m", "kline_1m age")) for issue in issues) and "1m" not in needed:
            needed.append("1m")
        if not needed:
            return []
        self.last_history_repair_at = now
        fetcher, symbol, stream = self.fetcher, self.symbol, self.ws_engine
        base_url = fetcher.base_url

        def closed_rows(frame, tf, cutoff):
            columns = ["open", "high", "low", "close", "volume"]
            result = frame.copy(deep=True)
            result.index = pd.to_datetime(result.index, utc=True).tz_localize(None)
            result[columns] = result[columns].apply(pd.to_numeric, errors="coerce")
            valid = result[columns].apply(lambda col: col.map(lambda value: math.isfinite(value))).all(axis=1)
            valid &= (result[["open", "high", "low", "close"]] > 0).all(axis=1) & (result["volume"] >= 0)
            valid &= result["high"] >= result[["open", "close", "low"]].max(axis=1)
            valid &= result["low"] <= result[["open", "close", "high"]].min(axis=1)
            valid &= pd.Series([candle_end(ts, tf) <= pd.Timestamp(cutoff, unit="s") for ts in result.index], index=result.index)
            return result.loc[valid]

        def fetch(tf):
            try:
                # Never use cached or cross-environment history for a repair.
                return tf, fetcher.fetch_klines(symbol, tf, 250, use_cache=False), None
            except Exception as exc:
                return tf, None, str(exc)

        from concurrent.futures import ThreadPoolExecutor
        repaired = []
        with ThreadPoolExecutor(max_workers=min(4, len(needed))) as pool:
            for tf, fetched, error in pool.map(fetch, needed):
                if error:
                    print(f"[DATA REPAIR] {tf}: {error}; snapshot veto remains active", flush=True)
                    continue
                try:
                    cutoff = time.time() if fixed_now is None else fixed_now
                    fetched = closed_rows(fetched, tf, cutoff)
                    with self.market_lock:
                        if self.symbol != symbol or self.fetcher.base_url != base_url or self.ws_engine is not stream:
                            continue  # A mode/symbol change invalidates this REST response.
                        live = self.data_map.get(tf)
                        if live is not None:
                            live = closed_rows(live, tf, cutoff)
                            fetched = pd.concat([fetched, live])
                        # Live rows win overlap; a kline received during REST is retained.
                        merged = fetched.loc[~fetched.index.duplicated(keep="last")].sort_index().iloc[-500:]
                        self.data_map[tf] = merged
                        snapshot.frames[tf] = merged
                        if not frame_invalid(tf, snapshot.freshness_issues(cutoff)):
                            self.market_source_times[f"kline_{tf}"] = cutoff
                            repaired.append(tf)
                except (KeyError, TypeError, ValueError, OverflowError) as exc:
                    print(f"[DATA REPAIR] {tf}: invalid closed OHLCV ({exc}); snapshot veto remains active", flush=True)
        return repaired

    def update_leverage_advice(self):
        atr_val = self.indicators["atr"] or (self.live_price * 0.008)
        regime = self.ai_verdict.regime if self.ai_verdict else "RANGING_SIDEWAY"
        conf = self.ai_verdict.confidence if self.ai_verdict else 75
        self.leverage_advice = self.leverage_engine.calculate_optimal_leverage(
            current_price=self.live_price,
            atr=atr_val,
            regime=regime,
            confidence=conf
        )

    def get_effective_leverage(self) -> int:
        if self.leverage_mode == "AI_AUTO" and self.leverage_advice:
            lev = self.leverage_advice.leverage
        else:
            lev = self.manual_leverage

        if hasattr(self, 'monthly_governor') and self.monthly_governor.enabled:
            gov = self.monthly_governor.evaluate(self.current_balance, self.trades)
            if gov.enabled and lev > gov.max_leverage_cap:
                lev = gov.max_leverage_cap

        return max(1, lev)

    def build_market_snapshot(self) -> MarketSnapshot:
        with self.market_lock:
            now = time.time()
            bids, asks = deepcopy(self.latest_l2_bids), deepcopy(self.latest_l2_asks)
            price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else self.live_price
            flow = self.order_flow_engine.evaluate(price)
            hft = self.visual_hft.get_metrics()
            return MarketSnapshot(
                snapshot_id=f"{self.symbol}-{int(now * 1000)}",
                symbol=self.symbol,
                exchange="binance",
                captured_at=now,
                price=price,
                source_times={
                    "depth": self.market_source_times.get("depth", 0.0),
                    "agg_trade": self.market_source_times.get("agg_trade", 0.0),
                    "kline_1m": self.market_source_times.get("kline_1m", 0.0),
                },
                bids=bids,
                asks=asks,
                frames={tf: frame.copy(deep=True) for tf, frame in self.data_map.items()},
                environment="testnet" if self.ws_engine and self.ws_engine.is_testnet else "paper",
                context={
                    "hft": hft,
                    "flow": flow,
                    "active_timeframe": self.active_timeframe,
                    "required_frames": list(dict.fromkeys(("1m", "5m", "15m", "1h", self.active_timeframe, *self.data_map))),
                },
                cvd=flow.current_cvd,
                vpin=hft.vpin,
            )

    def build_candidate_order(self, direction: int, order_type: str, entry_price: float, stop_loss: float, take_profit: float, source: str, confidence: float = 0.0) -> CandidateOrder:
        return CandidateOrder(
            order_id=f"{source[:8]}-{int(time.time() * 1000):x}-{uuid.uuid4().hex[:8]}",
            symbol=self.symbol,
            direction=direction,
            order_type=order_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self.get_effective_leverage(),
            source=source,
            confidence=confidence,
            regime=self.ai_verdict.regime if self.ai_verdict else "UNKNOWN",
        )

    def analyze_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Stage 2 computations use only the captured, closed-bar market view."""
        frames, price = snapshot.frames, snapshot.price
        df_struct = frames.get(snapshot.context["active_timeframe"])
        if df_struct is None or len(df_struct) < 50:
            raise ValueError("insufficient closed structure candles")
        self.order_flow_verdict = snapshot.context["flow"]
        self.indicators = self.ai_brain.calculate_indicators(df_struct)
        close = df_struct["close"]
        macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
        self.indicators.update(ema=self.indicators["ema50"], ema_fast=self.indicators["ema20"], ema_slow=self.indicators["ema50"], macd=float(macd.iloc[-1]), macd_signal=float(macd.ewm(span=9, adjust=False).mean().iloc[-1]))
        self.ai_verdict = self.ai_brain.analyze(
            frames, price, spread=(snapshot.asks[0][0] - snapshot.bids[0][0]), active_timeframe=snapshot.context["active_timeframe"]
        )
        self.deterministic_validation = self.deterministic_guardrail.validate(
            claimed_regime=self.ai_verdict.regime,
            claimed_confidence=self.ai_verdict.confidence,
            adx=self.indicators.get("adx") or 0.0,
            atr=self.indicators.get("atr") or 0.0,
            rsi=self.indicators.get("rsi") or 50.0,
        )
        self.ai_verdict.regime = self.deterministic_validation.corrected_regime
        self.ai_verdict.confidence = max(0, int(self.ai_verdict.confidence - self.deterministic_validation.confidence_penalty))
        self.update_leverage_advice()

        regime = self.ai_verdict.regime if self.ai_verdict else "RANGING_SIDEWAY"
        self.ensemble_result = self.ensemble_coordinator.evaluate_ensemble(
            frames, price, regime
        )
        self.candle_confluence = self.ensemble_coordinator.last_candle_confluence or self.multi_candle_engine.evaluate(frames, price)

        atr_val = self.indicators.get("atr") or (price * 0.008)
        atr_pct = (atr_val / price * 100.0) if price > 0 else 0.8
        conf = self.ai_verdict.confidence if self.ai_verdict else 80
        self.risk_manager.ai_cro.evaluate_risk_profile(
            current_balance=self.current_balance,
            market_regime=regime,
            confidence=conf,
            atr_pct=atr_pct
        )

        # -------------------------------------------------------------
        # OctoBot Tentacle Matrix & Trading Modes Evaluation
        # -------------------------------------------------------------
        smc_dict = {
            "structure": self.ai_verdict.smc_structure if self.ai_verdict else "RANGING",
            "demand_zone": list(self.ai_verdict.demand_zone) if (self.ai_verdict and self.ai_verdict.demand_zone) else None,
            "supply_zone": list(self.ai_verdict.supply_zone) if (self.ai_verdict and self.ai_verdict.supply_zone) else None,
            "liquidity_sweep": self.ai_verdict.last_sweep_info if self.ai_verdict else "NONE"
        } if self.ai_verdict else None

        vwap_status_str = getattr(self.ai_verdict, "vwap_status", "") if self.ai_verdict else ""
        vwap_dict = {
            "vwap": self.ai_verdict.vwap_fair_price if self.ai_verdict else price,
            "vwap_status": self.ai_verdict.vwap_status if self.ai_verdict else "EQUILIBRIUM_FAIR",
            "dist_sigma": -1.8 if ("DISCOUNT" in vwap_status_str) else (1.8 if ("PREMIUM" in vwap_status_str) else 0.0)
        } if self.ai_verdict else None

        of_dict = {
            "buy_ratio": self.order_flow_verdict.buy_ratio_pct if self.order_flow_verdict else 50.0,
            "cvd_delta_60s": self.order_flow_verdict.cvd_delta_60s if self.order_flow_verdict else 0.0,
            "absorption_signal": self.order_flow_verdict.absorption_signal if self.order_flow_verdict else "NONE"
        } if self.order_flow_verdict else None

        mtf_dict = {
            "score": (self.ensemble_result.consensus_score / 100.0) if self.ensemble_result else 0.0,
            "consensus": self.ensemble_result.consensus_verdict if self.ensemble_result else "NEUTRAL",
            "timeframes": self.ai_verdict.mtf_radar if (self.ai_verdict and self.ai_verdict.mtf_radar) else {}
        }

        self.octobot_consensus = self.octobot_matrix.evaluate_matrix(
            current_price=price,
            indicators=self.indicators,
            order_flow_telemetry=of_dict,
            smc_data=smc_dict,
            vwap_data=vwap_dict,
            mtf_consensus=mtf_dict
        )

        mode_name, trade_setup = self.octobot_coordinator.select_best_setup(
            current_price=price,
            matrix=self.octobot_consensus,
            indicators=self.indicators,
            smc_data=smc_dict,
            of_data=of_dict,
            vwap_data=vwap_dict
        )
        self.active_trading_mode = mode_name
        self.octobot_setup = trade_setup

        # Evaluate HKUDS Vibe Alpha Zoo (12 Quantitative Factors)
        if df_struct is not None and not df_struct.empty:
            self.vibe_alpha_zoo.evaluate(
                df=df_struct,
                current_price=price,
                best_bid=snapshot.bids[0][0],
                best_ask=snapshot.asks[0][0],
                spread=(snapshot.asks[0][0] - snapshot.bids[0][0])
            )

        # Research is deliberately last: it consumes the corrected regime,
        # current CRO profile, OctoBot tradability and fresh Alpha Zoo factors.
        self.order_research = self.ai_order_researcher.research(
            current_price=price,
            best_bid=snapshot.bids[0][0],
            best_ask=snapshot.asks[0][0],
            spread=(snapshot.asks[0][0] - snapshot.bids[0][0]),
            indicators=self.indicators,
            ai_verdict=self.ai_verdict,
            ensemble_result=self.ensemble_result,
            ai_cro=self.risk_manager.ai_cro,
            current_balance=self.current_balance,
            df_structure=df_struct,
            df_macro=frames.get("1h"),
            active_timeframe=snapshot.context["active_timeframe"],
            candle_confluence=self.candle_confluence,
            order_flow_verdict=self.order_flow_verdict,
            effective_leverage=self.get_effective_leverage(),
            current_position=self.current_position,
            inventory_skew=self.hummingbot_skew.calculate_reservation_price(
                price, self.current_position, self.indicators.get("atr", 200.0), self.current_balance
            ) if self.current_position else None,
            visual_hft_metrics=snapshot.context["hft"],
            jesse_metrics=self.jesse_engine.compute_metrics(),
            octobot_consensus=self.octobot_consensus
        )


    def evaluate_candidate_pipeline(self, candidate: CandidateOrder, research: Optional[AIOrderResearchResult] = None, snapshot=None, trace=None):
        snapshot = snapshot or self.build_market_snapshot()
        research = research or self.order_research

        def freqtrade_gate(order: CandidateOrder, _snapshot: MarketSnapshot) -> StageOutcome:
            if self.active_exchange != "binance":
                return StageOutcome.veto("MEXC is paper-only pending market-data parity; Binance data cannot approve MEXC orders")
            if self.binance_api.is_live_enabled and (not self.binance_api.is_testnet or _snapshot.environment != "testnet"):
                return StageOutcome.veto("execution/data environment mismatch or mainnet disabled")
            if getattr(self, "execution_blocker", ""):
                return StageOutcome.veto(self.execution_blocker)
            equity = self.current_balance + (self.current_position["unrealized_pnl"] if self.current_position else 0.0)
            allowed, reason, _status = self.freqtrade_protections.validate_new_trade(
                entry_price=order.entry_price,
                target_price=order.take_profit,
                direction=order.direction,
                balance=self.current_balance,
                equity=equity,
            )
            return StageOutcome.pass_(reason) if allowed else StageOutcome.veto(reason)

        def visual_hft_gate(order: CandidateOrder, _snapshot: MarketSnapshot) -> StageOutcome:
            metrics = _snapshot.context["hft"]
            if not metrics.vpin_ready or not metrics.depth_ready:
                return StageOutcome.veto(f"Microstructure warmup: {metrics.completed_bucket_count}/5 measured VPIN buckets; depth_ready={metrics.depth_ready}")
            if metrics.is_toxic_flow:
                return StageOutcome.veto(f"toxic flow VPIN={metrics.vpin:.2f}")
            if metrics.liquidity_drought_warning:
                return StageOutcome.veto(f"liquidity drought resilience={metrics.market_resilience_pct:.1f}%")
            return StageOutcome.pass_(f"L2 healthy; VPIN={metrics.vpin:.2f}")

        def jesse_gate(order: CandidateOrder, _snapshot: MarketSnapshot) -> StageOutcome:
            metrics = self.jesse_engine.compute_metrics()
            if metrics.current_consecutive_losses >= 3 or self.risk_manager.circuit_breaker_active:
                return StageOutcome.veto("Jesse loss-streak / daily circuit breaker")
            if metrics.total_trades < 30:
                return StageOutcome.pass_("probation: insufficient expectancy sample", probation=True, risk_cap_pct=0.0025)
            if metrics.expectancy_usdt <= 0 or metrics.current_consecutive_losses >= 3:
                return StageOutcome.veto(f"expectancy={metrics.expectancy_usdt:.2f}, loss streak={metrics.current_consecutive_losses}")
            return StageOutcome.pass_(f"positive expectancy={metrics.expectancy_usdt:.2f}")

        def deterministic_gate(order: CandidateOrder, _snapshot: MarketSnapshot) -> StageOutcome:
            nonlocal research
            self.analyze_snapshot(_snapshot)
            research = self.order_research
            if order.source == "auto" and not order.metadata.get("copilot_revalidated"):
                if research.recommended_side not in ("BUY", "SELL"):
                    return StageOutcome.veto("No directional research setup")
                order.direction = 1 if research.recommended_side == "BUY" else -1
                order.order_type = research.recommended_type
                order.entry_price = research.optimal_price
                order.stop_loss = research.structural_sl
                order.take_profit = research.structural_tp
            validation = self.deterministic_validation
            return StageOutcome.pass_(validation.validation_notes, regime=self.ai_verdict.regime, confidence=self.ai_verdict.confidence)

        def alpha_regime_gate(order: CandidateOrder, _snapshot: MarketSnapshot) -> StageOutcome:
            matrix = self.octobot_consensus
            if not matrix or not matrix.is_tradable:
                reason = matrix.summary_reason if matrix else "OctoBot matrix unavailable"
                return StageOutcome.veto(reason)
            if matrix.recommended_direction and matrix.recommended_direction != order.direction:
                return StageOutcome.veto("OctoBot direction conflicts with candidate")
            setup = self.octobot_setup
            if setup and setup.direction != order.direction:
                return StageOutcome.veto("OctoBot trade setup conflicts with candidate")
            alpha = self.vibe_alpha_zoo.get_latest_metrics()
            if alpha.composite_alpha_score * order.direction <= -15.0:
                return StageOutcome.veto(f"Alpha Zoo conflicts ({alpha.composite_alpha_score:+.1f})")
            fee_check = freqtrade_gate(order, _snapshot)
            if fee_check.verdict == "VETO":
                return fee_check
            staged_tp = {}
            if setup:
                staged_tp = {
                    "tp1": setup.staged_tp.tp1_price,
                    "tp1_ratio": setup.staged_tp.tp1_ratio,
                    "tp2": setup.staged_tp.tp2_price,
                    "tp2_ratio": setup.staged_tp.tp2_ratio,
                    "tp3": setup.staged_tp.tp3_price,
                    "tp3_ratio": setup.staged_tp.tp3_ratio,
                }
                if order.source == "auto":
                    order.take_profit = setup.staged_tp.tp3_price
                # Manual TP remains a hard final exit; do not advertise later stages.
                for stage in ("tp1", "tp2"):
                    if not 0 < (staged_tp[stage] - order.entry_price) * order.direction < (order.take_profit - order.entry_price) * order.direction:
                        staged_tp[f"{stage}_ratio"] = 0.0
            return StageOutcome.pass_(
                f"OctoBot {matrix.consensus_state}; Alpha Zoo {alpha.composite_alpha_score:+.1f}",
                alpha_score=alpha.composite_alpha_score,
                staged_take_profits=staged_tp,
                inventory_skew_applied=bool(self.current_position),
            )

        def council_gate(order: CandidateOrder, _snapshot: MarketSnapshot) -> StageOutcome:
            verdict = self.vibe_swarm.evaluate_council(
                current_price=_snapshot.price, indicators=self.indicators,
                ai_verdict=self.ai_verdict, ensemble_result=self.ensemble_result,
                order_research=research, alpha_zoo_metrics=self.vibe_alpha_zoo.get_latest_metrics(),
                visual_hft_metrics=_snapshot.context["hft"], jesse_metrics=self.jesse_engine.compute_metrics(),
                octobot_metrics=self.octobot_consensus, current_position=self.current_position,
                candidate=order, snapshot=_snapshot,
            )
            details = {"council_votes": [asdict(vote) for vote in verdict.votes], "order_id": order.order_id, "snapshot_id": _snapshot.snapshot_id}
            # A received REJECT remains a veto even if another reviewer is offline.
            if not verdict.available:
                if any(vote.vote == "REJECT" for vote in verdict.votes):
                    return StageOutcome.veto("Available reviewer rejected during partial 9Router outage", **details)
                return StageOutcome.unavailable(verdict.availability_reason, **details)
            if not verdict.approved:
                return StageOutcome.veto(verdict.council_rationale, **details)
            return StageOutcome("PASS", verdict.council_rationale, details=details)

        def sizing_gate(order: CandidateOrder, _snapshot: MarketSnapshot) -> StageOutcome:
            if not research:
                return StageOutcome.veto("Carver research is unavailable")
            leverage = min(order.leverage, self.get_effective_leverage())
            governor = self.monthly_governor.evaluate(self.current_balance, self.trades)
            leverage = min(leverage, governor.max_leverage_cap)
            position = self.current_position or {}
            if position and position["direction"] != order.direction and order.source != "auto":
                return StageOutcome.veto("Close opposite exposure before opening a new direction")
            pending = [item for item in self.order_manager.pending_orders if item.order_id != order.metadata.get("replace_order_id")]
            held = position.get("units", 0.0) * position.get("direction", 0) + sum(max(0, item.units - item.exchange_executed_quantity) * item.direction for item in pending)
            quant = research.carver_output or {}
            raw_signal = float(quant.get("raw_forecast", quant.get("capped_forecast", 0.0)))
            # Explicit pre-tuning blend: 75% strategy forecast + 25% normalized Zoo.
            blended_signal = 0.75 * raw_signal + 0.25 * order.alpha_score / 5.0
            daily_vol = float(quant.get("daily_price_vol_pct", 0.0))
            if not math.isfinite(daily_vol) or daily_vol <= 0:
                return StageOutcome.veto("No measured daily volatility for Carver")
            peak = max(self.current_balance, self.freqtrade_protections.max_drawdown_guard.peak_balance)
            carver = self.ai_order_researcher.carver_engine.compute_systematic_position(
                current_price=order.entry_price, capital_usdt=self.current_balance,
                raw_signal=blended_signal, daily_vol_pct=daily_vol,
                current_position_contracts=held,
                current_drawdown_pct=max(0.0, (peak - self.current_balance) / peak),
                effective_leverage=leverage,
            )
            contracts = carver.contracts_to_execute
            if carver.rebalance_action == "HOLD" or abs(contracts) <= 0:
                return StageOutcome.veto("Carver inertia buffer says HOLD", carver=asdict(carver))
            if position and order.source == "auto" and contracts * position["direction"] < 0:
                quantity = min(position["units"], math.floor(abs(contracts) / .001) * .001)
                if quantity <= 0:
                    return StageOutcome.veto("Reduction below contract step")
                return StageOutcome.pass_("Carver target reduces existing inventory; never reverses in one step",
                    direction=-position["direction"], order_type="MARKET", quantity=quantity, margin=0.0,
                    reduce_only=True, carver=asdict(carver), raw_carver_contracts=contracts)
            if contracts * order.direction <= 0:
                return StageOutcome.veto("Carver target conflicts with candidate", carver=asdict(carver))
            proposal = self.risk_manager.evaluate_order(
                symbol=order.symbol,
                direction=order.direction,
                entry_price=order.entry_price,
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                leverage=leverage,
            )
            if not proposal.approved:
                return StageOutcome.veto(proposal.rejection_reason or "risk manager rejected order")
            quantity = min(abs(contracts) * governor.size_multiplier, proposal.units)
            stop_distance = abs(order.entry_price - order.stop_loss)
            risk_per_unit = stop_distance + order.entry_price * (2 * self.fee_engine.taker_fee_rate + 0.0004)
            quantity = min(quantity, proposal.units * stop_distance / risk_per_unit)
            used_risk = sum(max(0, item.units - item.exchange_executed_quantity) * abs(item.price - item.stop_loss) for item in pending)
            used_risk += position.get("units", 0.0) * max(0.0, (position.get("entry_price", 0) - position.get("stop_loss", 0)) * position.get("direction", 0))
            used_margin = position.get("margin", 0.0) + sum(max(0, item.units - item.exchange_executed_quantity) * item.price / item.leverage for item in pending)
            remaining_risk = max(0.0, self.current_balance * self.risk_config.max_account_risk_pct - used_risk)
            remaining_margin = max(0.0, self.current_balance * 0.35 - used_margin)
            quantity = min(quantity, remaining_risk / stop_distance, remaining_margin * proposal.leverage / order.entry_price)
            for key in ("requested_quantity", "requested_margin"):
                if key in order.metadata:
                    cap = float(order.metadata[key])
                    if not math.isfinite(cap) or cap <= 0:
                        return StageOutcome.veto("Invalid requested size cap")
                    quantity = min(quantity, cap if key == "requested_quantity" else cap * proposal.leverage / order.entry_price)
            if position:
                skew = self.hummingbot_skew.calculate_reservation_price(_snapshot.price, position, self.indicators.get("atr"), self.current_balance)
                quantity *= max(0.0, 1.0 - abs(skew.inventory_ratio_q))
            if abs(order.take_profit - order.entry_price) / stop_distance < 1.0:
                return StageOutcome.veto("Risk/reward below 1:1")
            adjusted_margin_cap = order.metadata.get("copilot_max_margin")
            if adjusted_margin_cap is not None:
                try:
                    margin_cap = float(adjusted_margin_cap)
                    if margin_cap > 0:
                        quantity = min(quantity, margin_cap * proposal.leverage / order.entry_price)
                except (TypeError, ValueError):
                    return StageOutcome.veto("invalid Copilot margin adjustment")
            if order.metadata.get("probation"):
                stop_distance = abs(order.entry_price - order.stop_loss)
                probation_quantity = (self.current_balance * order.metadata["risk_cap_pct"]) / risk_per_unit if stop_distance > 0 else 0.0
                quantity = min(quantity, probation_quantity)
            quantity = math.floor((quantity + 1e-12) / 0.001) * 0.001
            if quantity <= 0:
                return StageOutcome.veto("final risk clamp below BTCUSDT minimum step")
            margin = quantity * order.entry_price / proposal.leverage
            return StageOutcome.pass_(
                f"Carver delta {contracts:+.4f}; CRO/monthly clamp applied",
                quantity=round(quantity, 4),
                margin=round(margin, 2),
                leverage=proposal.leverage,
                raw_carver_contracts=contracts,
                carver=asdict(carver),
                hard_quantity_cap=proposal.units,
                monthly_multiplier=governor.size_multiplier,
            )

        def memory_gate(order: CandidateOrder, _snapshot: MarketSnapshot) -> StageOutcome:
            if order.metadata.get("reduce_only"):
                return StageOutcome.pass_("Risk-reducing inventory exit; no new entry memory exposure")
            absorption = self.order_flow_verdict.absorption_signal if self.order_flow_verdict else "NONE"
            vwap = getattr(self.ai_verdict, "vwap_status", "EQUILIBRIUM_FAIR")
            check = self.trade_memory.query_similarity_against_losses(order.direction, self.indicators.get("rsi", 50.0), vwap, absorption)
            if not check.is_safe:
                return StageOutcome.veto(check.lesson_learned)
            return StageOutcome.pass_(check.lesson_learned, entry_context={
                "indicators": deepcopy(self.indicators), "of_data": {"absorption_signal": absorption, "delta_momentum": self.order_flow_verdict.delta_momentum},
                "vwap_data": {"vwap_status": vwap}, "smc_data": {"structure": self.ai_verdict.smc_structure},
                "snapshot_id": _snapshot.snapshot_id, "captured_at": _snapshot.captured_at,
            })

        decision = SevenStagePipeline(
            defense_gates=[("Freqtrade", freqtrade_gate), ("VisualHFT", visual_hft_gate), ("Jesse", jesse_gate)],
            stage2_gates=[("Deterministic Guard", deterministic_gate), ("OctoBot + Alpha Zoo", alpha_regime_gate)],
            council=council_gate,
            sizing=sizing_gate,
            memory=memory_gate,
        ).decide(candidate, snapshot, trace=trace)
        if decision.approved and self.ai_copilot.user_instruction and not candidate.metadata.get("copilot_revalidated"):
            # An explicit Copilot instruction reviews this exact sized order.
            copilot = self.ai_copilot._query_9router({"candidate": asdict(candidate), "snapshot_id": snapshot.snapshot_id, "instruction": self.ai_copilot.user_instruction})
            copilot.order_id = candidate.order_id
            copilot.snapshot_id = snapshot.snapshot_id
            self.ai_copilot_verdict = copilot
            self.ai_copilot_last_response_at = time.time()
            if copilot.gateway_connected and copilot.decision == "VETO":
                decision.trace.add("Copilot", StageOutcome.veto(copilot.thought_process))
            elif not copilot.gateway_connected:
                decision.trace.add("Copilot", StageOutcome.unavailable("No connected candidate review"))
        if decision.approved and self._apply_copilot_adjustment(decision.candidate):
            # The adjusted order must traverse every hard gate again.
            decision.trace.add("Copilot / Adjust", StageOutcome.pass_("Risk-reducing adjustment; Stage 1-5 revalidation required", **candidate.metadata["copilot_adjustment"]))
            return self.evaluate_candidate_pipeline(decision.candidate, research, trace=decision.trace)
        self.last_decision_trace = decision.trace
        self.last_candidate = decision.candidate
        return decision

    def _apply_copilot_adjustment(self, candidate: CandidateOrder) -> bool:
        """Accept only a fresh, connected Copilot reduction, then force a full recheck."""
        verdict = self.ai_copilot_verdict
        if (
            not verdict
            or candidate.metadata.get("copilot_revalidated")
            or getattr(verdict, "order_id", "") != candidate.order_id
            or verdict.decision != "ADJUST_ORDER"
            or not verdict.gateway_connected
            or (time.time() - self.ai_copilot_last_response_at) > 10.0
        ):
            return False

        changes: Dict[str, float] = {}
        if verdict.adjusted_margin is not None and 0 < verdict.adjusted_margin < candidate.margin:
            changes["max_margin"] = float(verdict.adjusted_margin)
            candidate.metadata["copilot_max_margin"] = float(verdict.adjusted_margin)
        if verdict.adjusted_sl is not None:
            new_sl = float(verdict.adjusted_sl)
            safer_sl = (candidate.direction == 1 and candidate.stop_loss <= new_sl < candidate.entry_price) or (
                candidate.direction == -1 and candidate.entry_price < new_sl <= candidate.stop_loss
            )
            if safer_sl:
                candidate.stop_loss = new_sl
                changes["stop_loss"] = new_sl
        if verdict.adjusted_tp is not None:
            new_tp = float(verdict.adjusted_tp)
            valid_tp = (candidate.direction == 1 and new_tp > candidate.entry_price) or (
                candidate.direction == -1 and 0 < new_tp < candidate.entry_price
            )
            if valid_tp:
                candidate.take_profit = new_tp
                changes["take_profit"] = new_tp
        if not changes:
            return False
        candidate.metadata["copilot_adjustment"] = changes
        candidate.metadata["copilot_revalidated"] = True
        return True

    def submit_candidate(self, candidate):
        decision = self.evaluate_candidate_pipeline(candidate)
        if not decision.approved:
            return {"status": "rejected", "reason": decision.trace.entries[-1].reason, "trace": [asdict(e) for e in decision.trace.entries]}
        result = self.queue_approved_candidate(candidate, self.order_research, candidate.metadata.get("execution_group", ""))
        return {**result, "candidate_id": candidate.order_id, "trace": [asdict(e) for e in decision.trace.entries]}

    def replace_pending(self, order_id, execute_now=False, **changes):
        target = next((o for o in self.order_manager.pending_orders if o.order_id == order_id), None)
        if not target or not target.candidate_payload:
            return {"status": "rejected", "reason": "Pending candidate not found"}
        candidate = CandidateOrder(**deepcopy(target.candidate_payload))
        candidate.order_id = f"replace-{uuid.uuid4().hex[:24]}"
        candidate.source = "manual-replace"
        candidate.entry_price = self.live_price if execute_now else changes.get("price") or target.price
        candidate.order_type = "MARKET" if execute_now else changes.get("order_type") or target.order_type
        if candidate.order_type == "TWAP_SLICE":
            candidate.order_type = "MARKET"
        for key in ("stop_loss", "take_profit", "leverage"):
            if changes.get(key) is not None:
                setattr(candidate, key, changes[key])
        if changes.get("side") is not None:
            if changes["side"] not in ("BUY", "SELL"):
                return {"status": "rejected", "reason": "Invalid side"}
            candidate.direction = 1 if changes["side"] == "BUY" else -1
        candidate.metadata.update(replace_order_id=order_id, requested_quantity=changes.get("units") or max(0, target.units - target.exchange_executed_quantity))
        if changes.get("margin") is not None:
            candidate.metadata["requested_margin"] = changes["margin"]
        for key in ("trigger_price", "callback_pct"):
            if changes.get(key) is not None:
                candidate.metadata[key] = changes[key]
        decision = self.evaluate_candidate_pipeline(candidate)
        if not decision.approved:
            return {"status": "rejected", "reason": decision.trace.entries[-1].reason}
        if not self.execution.cancel(target):
            return {"status": "cancel_pending", "reason": "No replacement until original cancel is confirmed"}
        remaining = max(0, target.units - target.exchange_executed_quantity)
        if remaining <= 1e-10:
            return {"status": "filled_during_cancel", "reason": "Original completed; no replacement required"}
        candidate.metadata["requested_quantity"] = min(candidate.metadata["requested_quantity"], remaining)
        candidate.metadata.update(execution_group=target.execution_group, group_type=target.group_type)
        return self.submit_candidate(candidate)

    def queue_approved_candidate(self, candidate, research=None, execution_group=""):
        if candidate.metadata.get("reduce_only"):
            position = self.current_position
            if not position or candidate.direction != -position["direction"] or candidate.quantity > position["units"]:
                return {"status": "rejected", "reason": "Reduction no longer matches inventory"}
            done = self.execution.request_close(candidate.quantity / position["units"], "CARVER_REBALANCE_REDUCE_ONLY")
            self.execution.trace(candidate.order_id, "Execution / Carver reduction", "PASS" if done else "PENDING", "Reduce-only exit", quantity=candidate.quantity)
            return {"status": "filled" if done else "exit_pending"}
        return self.execution.queue(candidate, research, execution_group)

    def reconcile_binance_testnet_orders(self):
        self.execution.tick()

    def on_tick(self, price: float):
        """Called on every live price tick"""
        self.live_price = price
        self.tick_count += 1

        if not self.price_history or abs(self.price_history[-1] - price) >= 0.01:
            self.price_history.append(price)
            if len(self.price_history) > 60:
                self.price_history.pop(0)

    def process_execution_tick(self, price: float):
        # Update position, floating PnL, and AI Continuous Coordination
        if self.current_position:
            pos = self.current_position
            direction = pos["direction"]
            units = pos["units"]
            unrealized = (price - pos["entry_price"]) * units * direction
            pos["unrealized_pnl"] = round(unrealized, 2)
            pos["roe_pct"] = round((unrealized / pos["margin"] * 100.0), 2) if pos["margin"] > 0 else 0.0

            # Backwards compatibility: Wrap existing position into orders if missing
            if "orders" not in pos or not pos["orders"]:
                pos["orders"] = [{
                    "slice_id": "ORD-1",
                    "order_id": "POS-ORIG",
                    "order_type": "MARKET",
                    "timeframe": pos.get("timeframe", self.active_timeframe),
                    "side": "LONG" if direction == 1 else "SHORT",
                    "direction": direction,
                    "entry_price": pos["entry_price"],
                    "units": pos["units"],
                    "margin": pos["margin"],
                    "notional": pos.get("notional", pos["margin"] * 10),
                    "entry_fee": pos.get("entry_fee", 0.0),
                    "fee_tier": pos.get("fee_tier", "TAKER (0.05%)"),
                    "entry_time": pos.get("entry_time", ""),
                    "unrealized_pnl": pos["unrealized_pnl"],
                    "roe_pct": pos["roe_pct"],
                }]

            # Update live PnL and ROE% for each constituent order slice
            for ord_item in pos.get("orders", []):
                ord_dir = ord_item.get("direction", direction)
                ord_units = ord_item.get("units", 0.0)
                ord_entry = ord_item.get("entry_price", pos["entry_price"])
                ord_margin = ord_item.get("margin", 0.0)
                slice_pnl = (price - ord_entry) * ord_units * ord_dir
                ord_item["unrealized_pnl"] = round(slice_pnl, 2)
                ord_item["roe_pct"] = round((slice_pnl / ord_margin * 100.0), 2) if ord_margin > 0 else 0.0

            # Update peak price
            if direction == 1 and price > pos.get("peak_price", pos["entry_price"]):
                pos["peak_price"] = price
            elif direction == -1 and price < pos.get("peak_price", pos["entry_price"]):
                pos["peak_price"] = price

            # Protective exits run even when entry gates veto or the bot is paused.
            if not self.execution.live:
                if (price - pos["stop_loss"]) * direction <= 0:
                    self.execution.request_close(1.0, "STOP_LOSS")
                    return
                if (price - pos["take_profit"]) * direction >= 0:
                    self.execution.request_close(1.0, "TAKE_PROFIT")
                    return
                staged = pos.get("staged_take_profits") or {}
                for stage in ("tp1", "tp2"):
                    remaining = self.execution.staged_tp_remaining(pos, stage)
                    if remaining > 0 and staged.get(stage) and (price - staged[stage]) * direction >= 0:
                        self.execution.request_close(min(1.0, remaining / pos["units"]), f"{stage.upper()} staged exit", tp_stage=stage)
                        return
            self.update_indicators()
            decision = self.ai_coordinator.evaluate_position(pos=pos, current_price=price, indicators=self.indicators, ai_verdict=self.ai_verdict, fee_engine=self.fee_engine)
            pos["ai_action_status"] = decision.status_display
            if decision.action in ("LOCK_BREAKEVEN", "EXPAND_TAKE_PROFIT", "UPDATE_TRAILING"):
                self.execution.replace_protection(sl=decision.new_stop_loss or None, tp=decision.new_take_profit or None)
            elif decision.action == "PARTIAL_TAKE_PROFIT":
                self.execution.request_close(0.5, decision.reason, after_stop_loss=decision.new_stop_loss)
            elif decision.action in ("AI_TAKE_PROFIT", "AI_CUT_LOSS"):
                self.execution.request_close(1.0, decision.reason)

        # AI Continuous Multi-Strategy Ensemble Execution & Coordination on every tick (4 spaces)

    def run_decision_cycle(self):
        with self.decision_lock:
            self.risk_manager.sync_day()
            if self.live_price <= 0:
                return
            self.reconcile_binance_testnet_orders()
            self.process_execution_tick(self.live_price)
            self.repair_closed_history()
            self.evaluate_ensemble_automated_decision(self.live_price)

    def evaluate_ensemble_automated_decision(self, current_price: float):
        if not self.is_running or self.order_manager.pending_orders:
            return
        now = time.time()
        if now - self.last_auto_order_time < 15.0:
            return
        self.last_auto_order_time = now
        candidate = self.build_candidate_order(0, "AUTO", current_price, 0.0, 0.0, "auto")
        decision = self.evaluate_candidate_pipeline(candidate)
        if decision.approved:
            self.queue_approved_candidate(decision.candidate, self.order_research)

    def persist_current_state(self):
        # Runtime is authoritative after restart; legacy account row is a UI/export view.
        if hasattr(self, "execution"):
            self.execution.persist()
        self.storage.save_account_state(
            symbol=self.symbol, initial_balance=self.initial_balance, current_balance=self.current_balance,
            peak_balance=self.freqtrade_protections.max_drawdown_guard.peak_balance,
            total_fees=self.total_fees, is_running=self.is_running, active_timeframe=self.active_timeframe,
            leverage_mode=self.leverage_mode, manual_leverage=self.manual_leverage, current_position=self.current_position,
        )

    def close_partial_position(self, ratio=0.5, reason="PARTIAL_TAKE_PROFIT", is_maker=False):
        return self.execution.request_close(ratio, reason)

    def close_position(self, exit_price, reason, is_maker=False, exchange_confirmed=False):
        return self.execution.request_close(1.0, reason)

    def lock_breakeven_now(self):
        if not self.current_position:
            return False
        return self.execution.replace_protection(sl=self.current_position["breakeven_price"])[0]

    def set_manual_tp_sl(self, sl, tp, lock_manual=True):
        return self.execution.replace_protection(sl, tp, manual=lock_manual)

    def close_order_slice(self, slice_id, exit_price=None, reason="ĐÓNG LỆNH LẺ THỦ CÔNG", is_maker=False):
        ok = self.execution.request_close(1.0, reason, slice_id=slice_id)
        return ok, "Exit filled" if ok else "Exit pending or rejected; inspect exchange status"

    def update_order_slice(self, slice_id, **kwargs):
        return False, "Confirmed fill price/quantity is immutable. Use a new candidate or reduce-only close."

    def get_state_dict(self):
        flow = self.order_flow_engine.evaluate(self.live_price)
        jesse = self.jesse_engine.compute_metrics()
        hb_skew = self.hummingbot_skew.calculate_reservation_price(
            self.live_price, self.current_position, self.indicators.get("atr", 200.0), self.current_balance
        )
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]
        total = len(self.trades)
        win_rate = (len(wins) / total * 100.0) if total > 0 else 0.0

        # Floating Equity & Real-time Net PnL (Closed PnL + Open Position Floating PnL)
        unrealized = self.current_position["unrealized_pnl"] if self.current_position else 0.0
        equity = self.current_balance + unrealized
        total_net_pnl = equity - self.initial_balance
        total_net_pnl_pct = (total_net_pnl / self.initial_balance) * 100.0
        closed_pnl = self.current_balance - self.initial_balance
        now = time.time()

        return {
            "symbol": self.symbol,
            "balance": round(self.current_balance, 2),
            "equity": round(equity, 2),
            "unrealized_pnl": round(unrealized, 2),
            "initial_balance": self.initial_balance,
            "net_pnl": round(total_net_pnl, 2),
            "net_pnl_pct": round(total_net_pnl_pct, 2),
            "closed_pnl": round(closed_pnl, 2),
            "is_running": self.is_running,
            "live_price": self.live_price,
            "price_history": self.price_history,
            "position": self.current_position,
            "trades": self.trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "total_trades": total,
            "total_fees": self.total_fees,
            "tick_count": self.tick_count,
            "indicators": self.indicators,
            "active_timeframe": self.active_timeframe,
            "funding_rate": self.funding_rate,
            "funding_rate_pct": round(self.funding_rate * 100, 4),
            "next_funding_time": self.next_funding_time,
            "mtf_radar": self.ai_verdict.mtf_radar if (self.ai_verdict and self.ai_verdict.mtf_radar) else {},
            "ai_brain": {
                "regime": self.ai_verdict.regime if self.ai_verdict else "ANALYZING",
                "regime_display": self.ai_verdict.regime_display if self.ai_verdict else "Đang phân tích...",
                "recommended_strategy": self.ai_verdict.recommended_strategy if self.ai_verdict else "MTF_TREND",
                "strategy_display": self.ai_verdict.strategy_display if self.ai_verdict else "Đang nhận diện...",
                "bull_score": self.ai_verdict.bull_score if self.ai_verdict else 50,
                "bear_score": self.ai_verdict.bear_score if self.ai_verdict else 50,
                "confidence": self.ai_verdict.confidence if self.ai_verdict else 80,
                "rationale": self.ai_verdict.rationale if self.ai_verdict else "Bộ não AI đang nạp dữ liệu nến để chẩn đoán trạng thái thị trường...",
                "risk_level": self.ai_verdict.risk_level if self.ai_verdict else "TRUNG BÌNH",
                "support_price": self.ai_verdict.support_price if self.ai_verdict else 0.0,
                "resistance_price": self.ai_verdict.resistance_price if self.ai_verdict else 0.0,
                "suggested_limit_buy": self.ai_verdict.suggested_limit_buy if self.ai_verdict else 0.0,
                "suggested_limit_sell": self.ai_verdict.suggested_limit_sell if self.ai_verdict else 0.0,
                "updated_time": self.ai_verdict.updated_time if self.ai_verdict else "",
                "smc_structure": self.ai_verdict.smc_structure if self.ai_verdict else "RANGING",
                "smc_bias": self.ai_verdict.smc_bias if self.ai_verdict else "NEUTRAL",
                "smc_rationale": self.ai_verdict.smc_rationale if self.ai_verdict else "",
                "vwap_fair_price": self.ai_verdict.vwap_fair_price if self.ai_verdict else 0.0,
                "vwap_status": self.ai_verdict.vwap_status if self.ai_verdict else "EQUILIBRIUM_FAIR",
                "demand_zone": list(self.ai_verdict.demand_zone) if (self.ai_verdict and self.ai_verdict.demand_zone) else None,
                "supply_zone": list(self.ai_verdict.supply_zone) if (self.ai_verdict and self.ai_verdict.supply_zone) else None,
                "active_fvgs": self.ai_verdict.active_fvgs if self.ai_verdict else [],
                "last_sweep_info": self.ai_verdict.last_sweep_info if self.ai_verdict else "",
                "hurst_exponent": getattr(self.ai_verdict, "hurst_exponent", 0.50) if self.ai_verdict else 0.50,
                "hurst_regime": getattr(self.ai_verdict, "hurst_regime", "RANDOM_WALK") if self.ai_verdict else "RANDOM_WALK",
                "garman_klass_vol": getattr(self.ai_verdict, "garman_klass_vol", 0.0) if self.ai_verdict else 0.0,
                "volatility_regime": getattr(self.ai_verdict, "volatility_regime", "NORMAL") if self.ai_verdict else "NORMAL"
            },
            "ai_risk": {
                "risk_per_trade_pct": self.risk_manager.ai_cro.last_verdict.risk_per_trade_pct if self.risk_manager.ai_cro.last_verdict else 1.5,
                "max_risk_amount_usdt": self.risk_manager.ai_cro.last_verdict.max_risk_amount_usdt if self.risk_manager.ai_cro.last_verdict else 15.0,
                "max_margin_utilization_pct": self.risk_manager.ai_cro.last_verdict.max_margin_utilization_pct if self.risk_manager.ai_cro.last_verdict else 40.0,
                "protection_mode": self.risk_manager.ai_cro.last_verdict.protection_mode if self.risk_manager.ai_cro.last_verdict else "BALANCED",
                "consecutive_loss_count": self.risk_manager.ai_cro.last_verdict.consecutive_loss_count if self.risk_manager.ai_cro.last_verdict else 0,
                "circuit_breaker_active": self.risk_manager.ai_cro.last_verdict.circuit_breaker_active if self.risk_manager.ai_cro.last_verdict else False,
                "risk_rationale": self.risk_manager.ai_cro.last_verdict.risk_rationale if self.risk_manager.ai_cro.last_verdict else "Antigravity AI Chief Risk Officer đang giám sát rủi ro vốn...",
                "defense_status": self.risk_manager.ai_cro.last_verdict.defense_status if self.risk_manager.ai_cro.last_verdict else "NORMAL",
                "updated_at": self.risk_manager.ai_cro.last_verdict.updated_at if self.risk_manager.ai_cro.last_verdict else ""
            },
            "pending_orders": self.order_manager.get_pending_orders(),
            "is_grid_active": self.is_grid_active,
            "auto_grid_rotation": self.auto_grid_rotation,
            "grid_total_profit": sum(t["pnl"] for t in self.trades if t.get("group_type") == "GRID"),
            "grid_levels": [
                {"id": g.order_id, "buy": g.price if g.direction == 1 else None, "sell": g.price if g.direction == -1 else None, "status": g.status} for g in self.order_manager.orders if g.group_type == "GRID"
            ],
            "leverage_info": {
                "mode": self.leverage_mode,
                "effective_leverage": self.get_effective_leverage(),
                "manual_leverage": self.manual_leverage,
                "ai_leverage": self.leverage_advice.leverage if self.leverage_advice else 3,
                "rationale": self.leverage_advice.rationale if self.leverage_advice else "Đang phân tích đòn bẩy an toàn...",
                "liq_distance_pct": self.leverage_advice.est_liq_distance_pct if self.leverage_advice else 30.0,
                "safety_rating": self.leverage_advice.safety_rating if self.leverage_advice else "AN TOÀN"
            },
            "book_ticker": self.fee_engine.get_state_dict(),
            "ensemble": {
                "consensus_score": self.ensemble_result.consensus_score if self.ensemble_result else 0.0,
                "consensus_direction": self.ensemble_result.consensus_direction if self.ensemble_result else 0,
                "consensus_verdict": self.ensemble_result.consensus_verdict if self.ensemble_result else "SIDEWAY_GRID",
                "confidence": self.ensemble_result.confidence if self.ensemble_result else 80,
                "active_mode": self.ensemble_result.active_mode if self.ensemble_result else "BALANCED",
                "rationale": self.ensemble_result.rationale if self.ensemble_result else "Đang tổng hợp dữ liệu đa chiến lược...",
                "strategies": [
                    {
                        "name": v.name,
                        "direction": v.direction,
                        "score": v.score,
                        "weight_pct": round(v.weight * 100, 1),
                        "rationale": v.rationale
                    }
                    for v in (self.ensemble_result.votes if self.ensemble_result else [])
                ]
            },
            "candle_radar": {
                "confluence_score": round(self.candle_confluence.confluence_score, 1) if self.candle_confluence else 0.0,
                "confluence_verdict": self.candle_confluence.confluence_verdict if self.candle_confluence else "NEUTRAL_MIXED",
                "recommended_action": self.candle_confluence.recommended_action if self.candle_confluence else "STAND_ASIDE",
                "aligned_timeframes_count": self.candle_confluence.aligned_timeframes_count if self.candle_confluence else 0,
                "macro_alignment": self.candle_confluence.macro_alignment if self.candle_confluence else False,
                "summary_rationale": self.candle_confluence.summary_rationale if self.candle_confluence else "Đang quét các mẫu hình nến đa khung...",
                "patterns": {
                    tf: {
                        "pattern_name": p.pattern_name,
                        "pattern_display": p.pattern_display,
                        "bias": p.bias,
                        "strength": round(p.strength, 1),
                        "candle_count": p.candle_count,
                        "description": p.description,
                        "close": p.last_candle_close
                    }
                    for tf, p in (self.candle_confluence.timeframe_patterns.items() if self.candle_confluence else {}.items())
                }
            },
            "order_research": {
                "ready": self.order_research is not None,
                "recommended_type": self.order_research.recommended_type if self.order_research else "WAIT",
                "recommended_side": self.order_research.recommended_side if self.order_research else "NONE",
                "optimal_price": self.order_research.optimal_price if self.order_research else self.live_price,
                "optimal_margin": self.order_research.optimal_margin if self.order_research else 0.0,
                "optimal_leverage": self.order_research.optimal_leverage if self.order_research else 3,
                "optimal_trigger_price": self.order_research.optimal_trigger_price if self.order_research else 0.0,
                "optimal_trigger_cond": self.order_research.optimal_trigger_cond if self.order_research else "ABOVE",
                "optimal_callback_pct": self.order_research.optimal_callback_pct if self.order_research else 0.8,
                "optimal_twap_slices": self.order_research.optimal_twap_slices if self.order_research else 5,
                "structural_sl": self.order_research.structural_sl if self.order_research else 0.0,
                "structural_tp": self.order_research.structural_tp if self.order_research else 0.0,
                "structural_tp_macro": self.order_research.structural_tp_macro if self.order_research else 0.0,
                "rr_ratio": self.order_research.rr_ratio if self.order_research else 1.5,
                "fee_tier": self.order_research.fee_tier if self.order_research else "MAKER (0.02%)",
                "estimated_fee_saved_usdt": self.order_research.estimated_fee_saved_usdt if self.order_research else 0.0,
                "win_probability": self.order_research.win_probability if self.order_research else 0,
                "research_rationale": self.order_research.research_rationale if self.order_research else "Chưa qua Defense Gates; chưa chạy Alpha/Research.",
                "dual_buy_price": self.order_research.dual_buy_price if self.order_research else 0.0,
                "dual_sell_price": self.order_research.dual_sell_price if self.order_research else 0.0,
                "execution_horizon": getattr(self.order_research, "execution_horizon", "IMMEDIATE"),
                "carver_contracts": getattr(self.order_research, "carver_contracts", 0.0),
                "carver_action": getattr(self.order_research, "carver_action", "HOLD"),
                "vpin": getattr(self.order_research, "vpin", 0.0),
                "toxicity_regime": getattr(self.order_research, "toxicity_regime", "WARMUP"),
                "market_resilience_pct": getattr(self.order_research, "market_resilience_pct", 0.0),
                "lob_imbalance_20": getattr(self.order_research, "lob_imbalance_20", 0.0),
                "kelly_multiplier": getattr(self.order_research, "kelly_multiplier", 1.0),
                "octobot_tradable": getattr(self.order_research, "octobot_tradable", False)
            },
            "carver_systematic": (
                self.order_research.carver_output if (self.order_research and self.order_research.carver_output) else {
                    "scaled_forecast": 0.0,
                    "capped_forecast": 0.0,
                    "daily_cash_vol_target": round((self.current_balance * 0.25) / (365 ** 0.5), 2),
                    "instrument_value_vol": round(self.live_price * 0.015, 2),
                    "rule_diversification_mult": 1.35,
                    "risk_overlay_multiplier": 1.0,
                    "drawdown_multiplier": 1.0,
                    "vol_shock_multiplier": 1.0,
                    "optimal_contracts": 0.0,
                    "rebalance_action": "HOLD",
                    "contracts_to_execute": 0.0,
                    "buffer_lower_bound": 0.0,
                    "buffer_upper_bound": 0.0,
                    "fee_saved_usdt": 0.0,
                    "summary_rationale": "Rob Carver Systematic Framework đang khởi tạo..."
                }
            ),
            "ai_copilot": {
                "decision": self.ai_copilot_verdict.decision if self.ai_copilot_verdict else "ABSTAIN",
                "confidence": self.ai_copilot_verdict.confidence if self.ai_copilot_verdict else 0,
                "market_regime_sentiment": self.ai_copilot_verdict.market_regime_sentiment if self.ai_copilot_verdict else "CHÂN TRỜI TÍCH LŨY",
                "shark_trap_warning": self.ai_copilot_verdict.shark_trap_warning if self.ai_copilot_verdict else "Chưa được AI đánh giá",
                "thought_process": self.ai_copilot_verdict.thought_process if self.ai_copilot_verdict else "Chưa có phản hồi 9Router; deterministic pipeline đang chịu trách nhiệm.",
                "strategic_advice": self.ai_copilot_verdict.strategic_advice if self.ai_copilot_verdict else "Chưa có đề xuất AI; xem Decision Trace.",
                "user_instruction_feedback": self.ai_copilot_verdict.user_instruction_feedback if self.ai_copilot_verdict else (self.ai_copilot.user_instruction or "Chưa có chỉ thị riêng từ bạn."),
                "model_used": self.ai_copilot_verdict.model_used if self.ai_copilot_verdict else f"9router/{self.ai_copilot.default_model}",
                "active_model": self.ai_copilot.default_model,
                "available_models": self.available_ai_models,
                "gateway_connected": bool(self.ai_copilot_verdict and self.ai_copilot_verdict.gateway_connected),
                "gateway_url": self.ai_copilot.gateway_url,
                "user_instruction": self.ai_copilot.user_instruction,
                "active_intel_tab": self.storage.get_setting("active_intel_tab", "intel-copilot"),
                "timestamp": self.ai_copilot_verdict.timestamp if self.ai_copilot_verdict else datetime.now().strftime("%H:%M:%S")
            },
            "order_flow": {
                "current_cvd": flow.current_cvd,
                "buy_volume_1m": flow.buy_volume_1m,
                "sell_volume_1m": flow.sell_volume_1m,
                "buy_ratio_pct": flow.buy_ratio_pct,
                "delta_momentum": flow.delta_momentum,
                "absorption_divergence": flow.absorption_divergence,
                "flow_rationale": flow.flow_rationale,
                "latency_ms": self.ws_latency_ms
            },
            "visual_hft": asdict(self.visual_hft.get_metrics()),
            "freqtrade": {
                "status": self.freqtrade_protections.get_status(self.current_balance, equity).status,
                "is_locked": self.freqtrade_protections.get_status(self.current_balance, equity).is_locked,
                "lock_reason": self.freqtrade_protections.get_status(self.current_balance, equity).lock_reason,
                "remaining_seconds": self.freqtrade_protections.get_status(self.current_balance, equity).remaining_seconds,
                "stoploss_count_60m": self.freqtrade_protections.stoploss_guard.get_recent_count(),
                "drawdown_pct": self.freqtrade_protections.max_drawdown_guard.get_drawdown_pct(self.current_balance),
                "peak_balance": round(self.freqtrade_protections.max_drawdown_guard.peak_balance, 2),
                "minimal_roi_target_pct": round(self.ai_coordinator.roi_engine.get_target_roi(
                    time.time() - self.current_position["open_timestamp"] if self.current_position and "open_timestamp" in self.current_position else 0.0,
                    self.active_timeframe
                ) * 100.0, 2),
                "trade_duration_sec": int(time.time() - self.current_position["open_timestamp"]) if self.current_position and "open_timestamp" in self.current_position else 0
            },
            "octobot": {
                "matrix_score": self.octobot_consensus.matrix_score if self.octobot_consensus else 0.0,
                "confidence_pct": self.octobot_consensus.confidence_pct if self.octobot_consensus else 0.0,
                "consensus_state": self.octobot_consensus.consensus_state if self.octobot_consensus else "NEUTRAL",
                "recommended_direction": self.octobot_consensus.recommended_direction if self.octobot_consensus else 0,
                "is_tradable": self.octobot_consensus.is_tradable if self.octobot_consensus else False,
                "summary_reason": self.octobot_consensus.summary_reason if self.octobot_consensus else "OctoBot Matrix đang nạp dữ liệu xúc tu...",
                "tentacles": self.octobot_consensus.tentacles if self.octobot_consensus else {},
                "active_trading_mode": (self.octobot_setup.mode_name if self.octobot_setup else "STAND_ASIDE"),
                "staged_tp": {
                    "tp1": self.octobot_setup.staged_tp.tp1_price if self.octobot_setup else 0.0,
                    "tp2": self.octobot_setup.staged_tp.tp2_price if self.octobot_setup else 0.0,
                    "tp3": self.octobot_setup.staged_tp.tp3_price if self.octobot_setup else 0.0
                } if self.octobot_setup else None
            },
            "jesse": {
                "expectancy_usdt": self.jesse_engine.compute_metrics().expectancy_usdt,
                "win_rate_pct": self.jesse_engine.compute_metrics().win_rate_pct,
                "profit_factor": jesse.profit_factor if math.isfinite(jesse.profit_factor) else None,
                "kelly_fraction_pct": self.jesse_engine.compute_metrics().kelly_fraction_pct,
                "edge_status": self.jesse_engine.compute_metrics().edge_status,
                "avg_win": self.jesse_engine.compute_metrics().avg_win_usdt,
                "avg_loss": self.jesse_engine.compute_metrics().avg_loss_usdt,
                "consecutive_losses": self.jesse_engine.compute_metrics().current_consecutive_losses
            },
            "trade_memory": {
                "lessons": self.trade_memory.get_recent_lessons(limit=10),
                "total_records": len(self.trade_memory.memory_records),
                "records": [
                    {
                        "trade_id": m.trade_id,
                        "direction": "LONG" if m.direction == 1 else "SHORT",
                        "entry_price": m.entry_price,
                        "rsi": round(m.rsi, 1),
                        "cvd_momentum": m.cvd_momentum,
                        "smc_structure": m.smc_structure,
                        "vwap_status": m.vwap_status,
                        "net_pnl": round(m.net_pnl, 2),
                        "outcome": m.outcome,
                        "failure_reason": m.failure_reason
                    }
                    for m in reversed(self.trade_memory.memory_records[-30:])
                ]
            },
            "pipeline": {
                "execution_blocker": self.execution_blocker,
                "execution_intents": [asdict(o) for o in self.order_manager.orders[-30:]],
                "protective_orders": deepcopy(self.execution.protective[-30:]),
                "execution_traces": deepcopy(dict(list(self.execution.traces.items())[-10:])),
                "data_environment": "testnet" if self.ws_engine and self.ws_engine.is_testnet else "paper",
                "stream_errors": dict(self.ws_engine.last_error) if self.ws_engine else {},
                "clock_offset_ms": self.ws_engine.clock_offset_ms if self.ws_engine else None,
                "source_age_seconds": {
                    source: (round(max(0.0, now - timestamp), 2) if timestamp else None)
                    for source, timestamp in self.market_source_times.items()
                },
                "l2_levels": {"bids": len(self.latest_l2_bids), "asks": len(self.latest_l2_asks)},
                "last_trace": {
                    "order_id": self.last_decision_trace.order_id,
                    "snapshot_id": self.last_decision_trace.snapshot_id,
                    "entries": [asdict(entry) for entry in self.last_decision_trace.entries],
                } if self.last_decision_trace else None,
                "candidate": asdict(self.last_candidate) if self.last_candidate else None,
            },
            "storage": self.storage.get_storage_telemetry(),
            "inventory_skew": {
                "side": hb_skew.current_position_side,
                "reservation_price": hb_skew.reservation_price,
                "skew_direction": hb_skew.skew_direction,
                "price_skew_offset": hb_skew.price_skew_offset,
                "inventory_ratio_q": hb_skew.inventory_ratio_q,
                "volatility_sigma": hb_skew.volatility_sigma,
                "skew_code": "DISCOURAGE_BUY" if hb_skew.current_position_side == "LONG" else ("DISCOURAGE_SELL" if hb_skew.current_position_side == "SHORT" else "NEUTRAL")
            },
            "settings": {
                "active_exchange": self.active_exchange,
                "exchange_name": "MEXC Futures" if self.active_exchange == "mexc" else "Binance Futures",
                **self.binance_api.get_masked_credentials(),
                **self.mexc_api.get_masked_credentials(),
                "binance": self.binance_api.get_masked_credentials(),
                "mexc": self.mexc_api.get_masked_credentials(),
                "vip_tier": self.fee_engine.vip_tier,
                "vip_name": self.fee_engine.current_tiers_dict.get(self.fee_engine.vip_tier, {}).get("name", self.fee_engine.vip_tier),
                "use_bnb_discount": self.fee_engine.use_bnb_discount,
                "maker_fee_pct": round(self.fee_engine.maker_fee_rate * 100, 4),
                "taker_fee_pct": round(self.fee_engine.taker_fee_rate * 100, 4),
                "is_custom_rate": self.fee_engine.is_custom_rate,
                "db_trades_count": len(self.trades),
                "db_memory_count": len(self.trade_memory.memory_records)
            },
            "vibe_trading": {
                "enabled": self.vibe_swarm.enabled,
                "min_votes": self.vibe_swarm.min_votes_required,
                "latest_council": (
                    {
                        **asdict(self.vibe_swarm.latest_verdict),
                        "approved": self.vibe_swarm.latest_verdict.approved,
                        "approved_votes": self.vibe_swarm.latest_verdict.approved_votes,
                        "total_votes": self.vibe_swarm.latest_verdict.total_votes,
                    }
                    if self.vibe_swarm.latest_verdict else None
                ),
                "alpha_zoo": asdict(self.vibe_alpha_zoo.get_latest_metrics()),
                "shadow_account": asdict(self.shadow_account.get_latest_analysis()),
                "models": self.vibe_swarm.agent_models
            },
            "monthly_target": asdict(self.monthly_governor.evaluate(self.current_balance, self.trades))
        }


state = LiveTradingState(symbol="BTCUSDT", balance=5000.0)


async def broadcast_state():
    """Broadcast state to all connected browser WebSockets"""
    if not connected_clients:
        return
    state_json = json.dumps({"type": "tick", "data": state.get_state_dict()})
    to_remove = set()
    for ws in connected_clients:
        try:
            await ws.send_text(state_json)
        except Exception:
            to_remove.add(ws)
    for ws in to_remove:
        connected_clients.discard(ws)


async def state_broadcast_loop():
    """Publish current telemetry at 4 FPS without network calls per client."""
    while True:
        try:
            if connected_clients and state.live_price > 0:
                await broadcast_state()
        except Exception:
            pass
        await asyncio.sleep(0.25)


async def ai_quant_background_loop():
    while True:
        try:
            await asyncio.to_thread(state.run_decision_cycle)
        except Exception as exc:
            import traceback
            print(f"[decision cycle] {exc}\n{traceback.format_exc()}", flush=True)
        await asyncio.sleep(1.0)


@app.on_event("startup")
async def startup_event():
    app.state.event_loop = asyncio.get_running_loop()
    await asyncio.to_thread(state.initialize_history)
    state.init_ws_engine()
    await state.ws_engine.start()
    asyncio.create_task(state_broadcast_loop())
    asyncio.create_task(ai_quant_background_loop())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        await websocket.send_text(json.dumps({"type": "full_state", "data": state.get_state_dict()}))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        connected_clients.discard(websocket)


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/state")
async def get_state():
    return state.get_state_dict()


@app.get("/api/storage/telemetry")
async def get_storage_telemetry():
    return state.storage.get_storage_telemetry()


@app.get("/api/klines")
def get_klines(symbol: Optional[str] = None, interval: str = "15m", limit: int = 120):
    """
    Returns candlestick OHLCV data for TradingView chart across timeframes:
    1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d
    """
    sym = (symbol or state.symbol).upper().strip()
    if sym != state.symbol or interval not in ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M") or not 1 <= limit <= 1500:
        raise HTTPException(status_code=400, detail="Invalid symbol, interval or limit")
    base = "https://demo-fapi.binance.com" if state.execution.live else "https://fapi.binance.com"
    url = f"{base}/fapi/v1/klines?symbol={sym}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "BinanceFuturesQuant/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candles = []
            for item in data:
                candles.append({
                    "time": int(item[0] / 1000),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5])
                })
            return {"symbol": sym, "interval": interval, "candles": candles}
    except Exception as e:
        return {"error": str(e), "candles": []}


@app.post("/api/action/toggle")
@serialized_action
def toggle_bot():
    state.is_running = not state.is_running
    return {"status": "ok", "is_running": state.is_running}


@app.post("/api/action/close_all")
@serialized_action
def close_all():
    if state.current_position and state.live_price > 0:
        ok = state.close_position(state.live_price, "THỦ CÔNG 🛑")
        return {"status": "closed" if ok else "exit_pending"}
    return {"status": "no_position"}


@app.post("/api/action/set_symbol")
async def set_symbol(symbol: str = "BTCUSDT"):
    if symbol.upper().strip() != "BTCUSDT" or state.current_position or state.order_manager.pending_orders:
        return {"status": "rejected", "reason": "BTCUSDT-only scope; cannot change symbol with exposure"}
    sym = symbol.upper().strip()
    if sym != state.symbol:
        state.symbol = sym
        state.current_position = None
        state.initialize_history()
        if state.ws_engine:
            await state.ws_engine.stop()
            state.init_ws_engine()
            await state.ws_engine.start()
        await broadcast_state()
    return {"status": "ok", "symbol": state.symbol}


@app.post("/api/action/reset_balance")
@serialized_action
def reset_balance(amount: float = 1000.0):
    if state.execution.live or state.current_position or state.order_manager.pending_orders:
        return {"status": "rejected", "reason": "Close/cancel exposure before resetting paper state"}
    if not math.isfinite(amount) or amount <= 0:
        return {"status": "rejected", "reason": "Invalid balance"}
    state.storage.reset_database(initial_balance=amount, symbol=state.symbol)
    state.initial_balance = amount
    state.current_balance = amount
    state.trades = []
    state.trade_memory.memory_records.clear()
    state.current_position = None
    state.total_fees = 0.0
    state.tick_count = 0
    state.last_auto_order_tick = 0
    state.last_auto_order_time = 0.0
    state.last_trade_closed_time = 0.0
    state.order_manager.orders.clear()
    state.freqtrade_protections = FreqtradeProtectionEngine(initial_balance=amount)
    state.risk_manager.reset_daily_stats(amount)
    state.persist_current_state()
    return {"status": "ok", "balance": state.current_balance}


@app.post("/api/action/manual_order")
@serialized_action
def manual_order(direction: int = 1):
    if direction not in (-1, 1):
        return {"status": "rejected", "reason": "Direction must be -1 or 1"}
    return place_custom_order(side="BUY" if direction == 1 else "SELL", margin=state.current_balance * 0.35)


@app.post("/api/action/set_leverage")
@serialized_action
def set_leverage(mode: str = "AI_AUTO", val: int = 3):
    state.leverage_mode = mode
    if mode == "MANUAL":
        state.manual_leverage = max(1, min(val, 20))
    state.update_leverage_advice()
    return {"status": "ok", "mode": state.leverage_mode, "effective_leverage": state.get_effective_leverage()}


@app.post("/api/action/toggle_grid")
@serialized_action
def toggle_grid():
    if state.is_grid_active:
        for order in list(state.order_manager.pending_orders):
            if order.group_type == "GRID":
                state.execution.cancel(order)
        state.is_grid_active = False
        return {"status": "stopped"}
    outcomes = []
    atr = state.indicators.get("atr") or state.live_price * 0.008
    group = f"grid-{uuid.uuid4().hex[:16]}"
    # Levels are real candidates, never pre-bought inventory or synthetic grid PnL.
    for direction in (1, -1):
        for level in range(1, 4):
            entry = state.live_price - direction * level * atr * 0.5
            candidate = state.build_candidate_order(direction, "LIMIT", entry, entry - direction * atr * 1.5, entry + direction * atr * 2, "manual-grid")
            candidate.metadata.update(group_type="GRID", requested_margin=state.current_balance * 0.035)
            decision = state.evaluate_candidate_pipeline(candidate)
            if decision.approved:
                outcomes.append(state.queue_approved_candidate(candidate, state.order_research, group))
            else:
                outcomes.append({"status": "rejected", "reason": decision.trace.entries[-1].reason})
    state.is_grid_active = any(o.group_type == "GRID" for o in state.order_manager.pending_orders)
    return {"status": "evaluated", "levels": outcomes, "is_grid_active": state.is_grid_active}


@app.post("/api/action/toggle_auto_grid")
async def toggle_auto_grid():
    state.auto_grid_rotation = not state.auto_grid_rotation
    print(f"🔄 [AI AUTO-ROTATION] Chế độ xoay tua tự động Grid/Trend đã chuyển thành: {state.auto_grid_rotation}", flush=True)
    await broadcast_state()
    return {"status": "ok", "auto_grid_rotation": state.auto_grid_rotation}


@app.post("/api/action/ai_copilot_reason")
@serialized_action
def ai_copilot_reason(instruction: Optional[str] = None):
    if instruction:
        state.ai_copilot.set_user_instruction(instruction)
    
    carver_dict = getattr(state.order_research, "carver_output", None) if state.order_research else None
    verdict = state.ai_copilot.evaluate_market(
        current_price=state.live_price,
        indicators=state.indicators,
        ai_verdict=state.ai_verdict,
        ensemble_result=state.ensemble_result,
        order_research=state.order_research,
        carver_metrics=carver_dict,
        trade_memories=state.trade_memory.memory_records[-3:] if state.trade_memory.memory_records else [],
        current_position=state.current_position
    )
    state.ai_copilot_verdict = verdict
    state.ai_copilot_last_response_at = time.time()
    return {"status": "ok", "verdict": asdict(verdict)}


@app.post("/api/action/set_ai_copilot_instruction")
async def set_ai_copilot_instruction(instruction: str = ""):
    state.ai_copilot.set_user_instruction(instruction)
    state.storage.save_setting("ai_copilot_user_instruction", instruction)
    print(f"💬 [USER INSTRUCTION] Tiếp nhận chỉ thị từ người dùng & lưu SQLite: '{instruction}'", flush=True)
    await broadcast_state()
    return {"status": "ok", "instruction": instruction}


@app.post("/api/action/set_active_tab")
async def set_active_tab(tab: str = "intel-copilot"):
    state.storage.save_setting("active_intel_tab", tab)
    return {"status": "ok", "active_tab": tab}


@app.get("/api/action/get_ai_models")
async def get_ai_models():
    models = await asyncio.to_thread(state.ai_copilot.get_available_models)
    state.available_ai_models = models
    return {"status": "ok", "models": models, "active_model": state.ai_copilot.default_model}


@app.post("/api/action/set_ai_model")
async def set_ai_model(model: str = "ag/gemini-3.8-flash-high"):
    state.ai_copilot.set_model(model)
    state.storage.save_setting("ai_copilot_model", state.ai_copilot.default_model)
    print(f"🤖 [AI COPILOT] Đã chuyển sang mô hình AI: {state.ai_copilot.default_model} và lưu vĩnh viễn vào SQLite", flush=True)
    await broadcast_state()
    return {"status": "ok", "active_model": state.ai_copilot.default_model}



@app.post("/api/action/place_custom_order")
@serialized_action
def place_custom_order(
    side: str = "BUY",
    order_type: str = "MARKET",
    price: float = 0.0,
    margin: float = 100.0,
    leverage: Optional[int] = None,
    trigger_price: float = 0.0,
    trigger_condition: str = "ABOVE",
    callback_pct: float = 0.8,
    twap_slices: int = 5,
    twap_interval_ticks: int = 6
):
    """
    Supports all 7 professional order types:
    MARKET, LIMIT, POST_ONLY, CONDITIONAL, TRAILING_STOP, TWAP, SCALE_RATIO
    """
    sym = state.symbol
    current_p = state.live_price
    target_price = price if price > 0 else current_p
    lev = leverage if (leverage and leverage > 0) else state.get_effective_leverage()
    clean_type = order_type.upper()
    if clean_type not in ("MARKET", "LIMIT", "POST_ONLY", "CONDITIONAL", "TRAILING_STOP", "TWAP", "SCALE_RATIO"):
        return {"status": "rejected", "reason": "Loại lệnh không được hỗ trợ."}
    direction = 1 if side.upper() == "BUY" else -1
    if side.upper() not in ("BUY", "SELL"):
        return {"status": "rejected", "reason": "side chỉ được BUY hoặc SELL."}
    setup = state.ai_order_researcher.structural_calculator.compute_setup(
        side=side.upper(), entry_price=target_price, df_structure=state.get_structure_df(), timeframe=state.active_timeframe
    )
    candidate = state.build_candidate_order(
        direction=direction,
        order_type=clean_type,
        entry_price=target_price,
        stop_loss=setup.stop_loss,
        take_profit=setup.take_profit,
        source="manual",
        confidence=state.ai_verdict.confidence if state.ai_verdict else 0.0,
    )
    candidate.leverage = min(candidate.leverage, int(lev))
    candidate.metadata.update(requested_margin=margin, trigger_price=trigger_price, trigger_condition=trigger_condition, callback_pct=callback_pct, twap_slices=twap_slices, twap_interval_seconds=twap_interval_ticks * 0.5)
    return state.submit_candidate(candidate)


@app.post("/api/action/cancel_order")
@serialized_action
def cancel_order(order_id: int):
    target = next((o for o in state.order_manager.pending_orders if o.order_id == order_id), None)
    if not target:
        return {"status": "not_found"}
    return {"status": "cancelled" if state.execution.cancel(target) else "cancel_pending", "order_id": order_id}


@app.post("/api/action/update_pending_order")
@serialized_action
def update_pending_order(
    order_id: int,
    price: Optional[float] = None,
    units: Optional[float] = None,
    margin: Optional[float] = None,
    timeframe: Optional[str] = None,
    order_type: Optional[str] = None,
    side: Optional[str] = None,
    trigger_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    leverage: Optional[int] = None,
    callback_pct: Optional[float] = None
):
    return state.replace_pending(order_id, price=price, units=units, margin=margin, timeframe=timeframe, order_type=order_type, side=side, trigger_price=trigger_price, stop_loss=stop_loss, take_profit=take_profit, leverage=leverage, callback_pct=callback_pct)


@app.post("/api/action/execute_pending_order")
@serialized_action
def execute_pending_order(order_id: int):
    return state.replace_pending(order_id, execute_now=True)


class UpdateTpSlRequest(BaseModel):
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    lock_manual: bool = True


@app.post("/api/action/update_position_tp_sl")
@serialized_action
def update_position_tp_sl(req: UpdateTpSlRequest):
    if not state.current_position:
        raise HTTPException(status_code=400, detail="Không có vị thế nào đang mở!")

    success, msg = state.set_manual_tp_sl(
        sl=req.stop_loss,
        tp=req.take_profit,
        lock_manual=req.lock_manual
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {
        "status": "ok",
        "message": msg,
        "stop_loss": state.current_position.get("stop_loss"),
        "take_profit": state.current_position.get("take_profit"),
        "is_manual_tpsl": state.current_position.get("is_manual_tpsl")
    }


@app.post("/api/action/close_order_slice")
@serialized_action
def close_order_slice(slice_id: str):
    if not state.current_position:
        raise HTTPException(status_code=400, detail="Không có vị thế nào đang mở!")

    success, msg = state.close_order_slice(slice_id=slice_id)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {"status": "ok", "message": msg, "slice_id": slice_id}


@app.post("/api/action/update_order_slice")
@serialized_action
def update_order_slice(
    slice_id: str,
    entry_price: Optional[float] = None,
    units: Optional[float] = None,
    margin: Optional[float] = None,
    timeframe: Optional[str] = None,
    order_type: Optional[str] = None
):
    if not state.current_position:
        raise HTTPException(status_code=400, detail="Không có vị thế nào đang mở!")

    success, msg = state.update_order_slice(
        slice_id=slice_id,
        entry_price=entry_price,
        units=units,
        margin=margin,
        timeframe=timeframe,
        order_type=order_type
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    return {"status": "ok", "message": msg, "slice_id": slice_id, "position": state.current_position}


@app.post("/api/action/set_timeframe")
@serialized_action
def set_timeframe(timeframe: str = "15m"):
    if timeframe not in state.data_map:
        return {"status": "error", "message": "No closed history for timeframe"}
    state.active_timeframe = timeframe
    state.last_auto_order_time = 0
    state.persist_current_state()
    return {"status": "ok", "active_timeframe": timeframe}


@app.post("/api/action/place_dual_bracket")
@serialized_action
def place_dual_bracket():
    res = state.order_research
    if not res:
        return {"status": "error", "message": "Nghiên cứu AI chưa sẵn sàng"}
    sub_margin = round(res.optimal_margin * 0.5, 2)
    atr = state.indicators.get("atr") or (state.live_price * 0.008)
    group = f"dual-{int(time.time() * 1000):x}-{uuid.uuid4().hex[:6]}"
    candidates = [
        state.build_candidate_order(1, "POST_ONLY", res.dual_buy_price, round(res.dual_buy_price - 1.2 * atr, 2), res.dual_sell_price, "manual-dual", getattr(res, "win_probability", 0.0)),
        state.build_candidate_order(-1, "POST_ONLY", res.dual_sell_price, round(res.dual_sell_price + 1.2 * atr, 2), res.dual_buy_price, "manual-dual", getattr(res, "win_probability", 0.0)),
    ]
    for candidate in candidates:
        candidate.metadata.update(group_type="OCO", requested_margin=sub_margin)
    decisions = [state.evaluate_candidate_pipeline(candidate, res) for candidate in candidates]
    rejected = [decision.trace.entries[-1].reason for decision in decisions if not decision.approved]
    if rejected:
        return {"status": "rejected", "reason": "; ".join(rejected), "trace": [[asdict(entry) for entry in decision.trace.entries] for decision in decisions]}
    for decision in decisions:
        state.last_decision_trace = decision.trace
        state.queue_approved_candidate(decision.candidate, res, execution_group=group)
    return {
        "status": "ok",
        "message": f"Đã rải 2 đầu: Mua ${res.dual_buy_price} & Bán ${res.dual_sell_price}",
        "execution_group": group,
        "trace": [[asdict(entry) for entry in decision.trace.entries] for decision in decisions],
    }


@app.post("/api/action/partial_close")
@serialized_action
def partial_close(ratio: float = 0.5):
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        return {"status": "rejected", "reason": "Ratio must be in (0, 1]"}
    if not state.current_position:
        return {"status": "error", "message": "Không có vị thế nào đang mở"}
    rec = state.close_partial_position(ratio=ratio, reason=f"THỦ CÔNG: CHỐT {int(ratio*100)}% VỊ THẾ 💰", is_maker=False)
    return {"status": "closed" if rec else "exit_pending", "record": rec}


@app.post("/api/action/lock_breakeven")
@serialized_action
def lock_breakeven():
    if not state.current_position:
        return {"status": "error", "message": "Không có vị thế nào đang mở"}
    ok = state.lock_breakeven_now()
    return {"status": "ok" if ok else "error", "message": "Đã dời Stop Loss về điểm hòa vốn (Entry + Phí Sàn)!"}


@app.post("/api/action/set_risk_pct")
@serialized_action
def set_risk_pct(risk_pct: float = 1.5):
    clamped = max(0.2, min(5.0, risk_pct))
    state.risk_manager.ai_cro.user_risk_pct = clamped
    regime = state.ai_verdict.regime if state.ai_verdict else "BALANCED"
    confidence = state.ai_verdict.confidence if state.ai_verdict else 80
    atr = state.indicators.get("atr") or (state.live_price * 0.008)
    atr_pct = (atr / state.live_price * 100.0) if state.live_price > 0 else 0.8
    state.risk_manager.ai_cro.last_verdict = state.risk_manager.ai_cro.evaluate_risk_profile(
        current_balance=state.current_balance,
        market_regime=regime,
        confidence=confidence,
        atr_pct=atr_pct
    )
    return {"status": "ok", "risk_pct": clamped}


# =========================================================================
# BINANCE API SETTINGS, PERSISTENT STORAGE & VIP FEE MANAGEMENT ROUTES
# =========================================================================

cached_public_ip = "1.52.182.16"

@app.get("/api/settings")
async def get_settings():
    global cached_public_ip
    try:
        import urllib.request
        req = urllib.request.Request("https://api.ipify.org", headers={"User-Agent": "curl/7.68.0"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            cached_public_ip = resp.read().decode().strip()
    except Exception:
        pass

    return {
        "status": "ok",
        "server_public_ip": cached_public_ip,
        "active_exchange": state.active_exchange,
        "binance": state.binance_api.get_masked_credentials(),
        "mexc": state.mexc_api.get_masked_credentials(),
        "settings": {
            "active_exchange": state.active_exchange,
            "exchange_name": "MEXC Futures" if state.active_exchange == "mexc" else "Binance Futures",
            **state.binance_api.get_masked_credentials(),
            **state.mexc_api.get_masked_credentials(),
            "server_public_ip": cached_public_ip,
            "vip_tier": state.fee_engine.vip_tier,
            "vip_name": BINANCE_VIP_TIERS.get(state.fee_engine.vip_tier, {}).get("name", state.fee_engine.vip_tier),
            "use_bnb_discount": state.fee_engine.use_bnb_discount,
            "maker_fee_pct": round(state.fee_engine.maker_fee_rate * 100, 4),
            "taker_fee_pct": round(state.fee_engine.taker_fee_rate * 100, 4),
            "is_custom_rate": state.fee_engine.is_custom_rate,
            "db_trades_count": len(state.trades),
            "db_memory_count": len(state.trade_memory.memory_records)
        },
        "monthly_target": asdict(state.monthly_governor.evaluate(state.current_balance, state.trades)),
        "available_vip_tiers": BINANCE_VIP_TIERS
    }


@app.post("/api/settings/update_monthly_target")
async def update_monthly_target(payload: dict):
    target_pct = float(payload.get("target_pct", 10.0))
    enabled = bool(payload.get("enabled", True))
    auto_compensate = bool(payload.get("auto_compensate_deficit", True))
    state.monthly_governor.update_config(target_pct, enabled, auto_compensate)
    await broadcast_state()
    gov_status = state.monthly_governor.evaluate(state.current_balance, state.trades)
    return {
        "status": "ok",
        "message": f"Đã lưu cấu hình mục tiêu tháng: {target_pct:.1f}% (Tự động điều tiết an toàn: BẬT, Tự động bù thiếu hụt: {'BẬT' if auto_compensate else 'TẮT'})",
        "monthly_target": asdict(gov_status)
    }


@app.post("/api/settings/save_api_keys")
@serialized_action
def save_api_keys(payload: dict):
    if state.execution.live or state.current_position or state.order_manager.pending_orders:
        return {"status": "rejected", "reason": "Close/cancel exposure before replacing data or credentials"}
    exchange = str(payload.get("exchange", "binance")).strip().lower()
    set_active = bool(payload.get("set_active", False))

    if exchange == "mexc":
        mexc_key = str(payload.get("api_key", payload.get("mexc_api_key", ""))).strip()
        mexc_secret = str(payload.get("api_secret", payload.get("mexc_api_secret", ""))).strip()
        mexc_base = str(payload.get("base_url", payload.get("mexc_base_url", ""))).strip()
        mexc_proxy = str(payload.get("proxy_url", payload.get("mexc_proxy_url", ""))).strip()

        if mexc_key and "*" not in mexc_key:
            state.storage.save_setting("mexc_api_key", mexc_key)
            state.mexc_api.api_key = mexc_key
        if mexc_secret and "*" not in mexc_secret:
            state.storage.save_setting("mexc_api_secret", mexc_secret)
            state.mexc_api.api_secret = mexc_secret
        if mexc_base:
            state.storage.save_setting("mexc_base_url", mexc_base)
            state.mexc_api.base_url = mexc_base
        if mexc_proxy is not None:
            state.storage.save_setting("mexc_proxy_url", mexc_proxy)
            state.mexc_api.proxy_url = mexc_proxy

        if set_active:
            state.active_exchange = "mexc"
            state.storage.save_setting("active_exchange", "mexc")

        return {"status": "ok", "message": "Đã lưu thông tin API MEXC Contract an toàn vào SQLite!"}

    else:
        api_key = str(payload.get("api_key", "")).strip()
        api_secret = str(payload.get("api_secret", "")).strip()
        raw_tn = payload.get("is_testnet", False)
        is_testnet = (raw_tn is True) or str(raw_tn).strip().lower() in ("true", "1", "yes", "on")

        if api_key and "*" not in api_key:
            state.storage.save_setting("api_key", api_key)
            state.binance_api.api_key = api_key
        if api_secret and "*" not in api_secret:
            state.storage.save_setting("api_secret", api_secret)
            state.binance_api.api_secret = api_secret
        state.storage.save_setting("is_testnet", is_testnet)
        state.binance_api.is_testnet = is_testnet

        if set_active:
            state.active_exchange = "binance"
            state.storage.save_setting("active_exchange", "binance")

        return {"status": "ok", "message": "Đã lưu thông tin API Binance an toàn vào cơ sở dữ liệu SQLite!"}


@app.post("/api/settings/test_connection")
@serialized_action
def test_binance_connection(payload: Optional[dict] = None):
    if state.execution.live or state.current_position or state.order_manager.pending_orders:
        payload = None  # Active account credentials are immutable during exposure.
    exchange = "binance"
    if payload:
        exchange = str(payload.get("exchange", "binance")).strip().lower()

    if exchange == "mexc":
        if payload:
            mexc_key = str(payload.get("api_key", payload.get("mexc_api_key", ""))).strip()
            mexc_secret = str(payload.get("api_secret", payload.get("mexc_api_secret", ""))).strip()
            mexc_base = str(payload.get("base_url", payload.get("mexc_base_url", ""))).strip()
            mexc_proxy = str(payload.get("proxy_url", payload.get("mexc_proxy_url", ""))).strip()

            if mexc_key and "*" not in mexc_key:
                state.mexc_api.api_key = mexc_key
                state.storage.save_setting("mexc_api_key", mexc_key)
            if mexc_secret and "*" not in mexc_secret:
                state.mexc_api.api_secret = mexc_secret
                state.storage.save_setting("mexc_api_secret", mexc_secret)
            if mexc_base:
                state.mexc_api.base_url = mexc_base
                state.storage.save_setting("mexc_base_url", mexc_base)
            if mexc_proxy is not None:
                state.mexc_api.proxy_url = mexc_proxy
                state.storage.save_setting("mexc_proxy_url", mexc_proxy)

        return state.mexc_api.test_connection()

    else:
        if payload:
            api_key = str(payload.get("api_key", "")).strip()
            api_secret = str(payload.get("api_secret", "")).strip()
            raw_tn = payload.get("is_testnet", False)
            is_testnet = (raw_tn is True) or str(raw_tn).strip().lower() in ("true", "1", "yes", "on")
            if api_key and "*" not in api_key:
                state.binance_api.api_key = api_key
                state.storage.save_setting("api_key", api_key)
            if api_secret and "*" not in api_secret:
                state.binance_api.api_secret = api_secret
                state.storage.save_setting("api_secret", api_secret)
            state.binance_api.is_testnet = is_testnet
            state.storage.save_setting("is_testnet", is_testnet)
        return state.binance_api.test_connection()


@app.post("/api/settings/switch_exchange")
async def switch_exchange(payload: dict):
    if state.current_position or state.order_manager.pending_orders or state.execution.live:
        return {"status": "rejected", "reason": "Close/cancel exposure before switching exchange"}
    ex = str(payload.get("exchange", "binance")).strip().lower()
    if ex not in ("binance", "mexc"):
        return {"status": "error", "message": "Sàn giao dịch không hợp lệ. Chọn 'binance' hoặc 'mexc'."}
    state.active_exchange = ex
    state.storage.save_setting("active_exchange", ex)
    state.fee_engine.set_exchange(ex)

    # Automatically adapt fees to new active exchange
    if ex == "mexc":
        comm = state.mexc_api.fetch_commission_rate(state.symbol)
        if comm.get("success"):
            m = comm["maker_commission"]
            t = comm["taker_commission"]
            state.fee_engine.set_custom_rates(m, t)
        else:
            state.fee_engine.set_vip_tier("MEXC_STANDARD")
    else:
        saved_tier = state.storage.get_setting("vip_tier", "VIP_0")
        use_bnb = state.storage.get_setting("use_bnb_discount", False)
        state.fee_engine.set_vip_tier(saved_tier, use_bnb_discount=use_bnb)

    await broadcast_state()
    ex_name = "MEXC Futures" if ex == "mexc" else "Binance Futures"
    return {"status": "ok", "active_exchange": ex, "message": f"Đã chuyển sang sàn hoạt động: {ex_name}!"}


@app.post("/api/settings/toggle_mode")
@serialized_action
def toggle_trading_mode(payload: dict):
    if state.current_position or state.order_manager.pending_orders or state.execution.entry_exit_barrier:
        return {"status": "rejected", "reason": "Close/cancel exposure before switching execution environment"}
    live_enabled = payload.get("live_enabled", False)
    if not isinstance(live_enabled, bool):
        return {"status": "rejected", "reason": "live_enabled must be boolean"}
    if live_enabled:
        if state.active_exchange != "binance":
            return {"status": "error", "message": "MEXC chỉ chạy paper-only cho đến khi có market-data và lifecycle parity."}
        if not state.binance_api.is_testnet:
            return {"status": "error", "message": "Scope hiện tại chỉ cho phép Binance Futures Testnet, không bật Mainnet."}
        conn_res = state.active_exchange_api.test_connection()
        if not conn_res.get("success", False):
            return {
                "status": "error",
                "message": f"Không thể kích hoạt Giao Dịch Thật ({state.active_exchange.upper()}): {conn_res.get('message', 'Lỗi kết nối')}"
            }
        if not conn_res.get("can_trade", False):
            return {
                "status": "error",
                "message": f"Tài khoản {state.active_exchange.upper()} chưa được cấp quyền Giao Dịch Futures (canTrade = False)!"
            }

    state.binance_api.is_live_enabled = live_enabled
    state.mexc_api.is_live_enabled = False
    state.storage.save_setting("is_live_enabled", live_enabled)
    if state.ws_engine:
        asyncio.run_coroutine_threadsafe(state.ws_engine.stop(), app.state.event_loop).result(timeout=20)
    state.market_source_times.clear()
    state.data_map.clear()
    state.visual_hft = VisualHFTMicrostructureEngine(bucket_size_btc=10.0, num_buckets=30)
    state.order_flow_engine = OrderFlowCVDEngine(max_ticks=3000, window_seconds=60)
    state.order_flow_verdict = None
    state.latest_l2_bids = []
    state.latest_l2_asks = []
    state.execution.startup_reconciled = False
    state.initialize_history()
    state.init_ws_engine()
    asyncio.run_coroutine_threadsafe(state.ws_engine.start(), app.state.event_loop).result(timeout=20)
    mode_text = "BINANCE TESTNET" if live_enabled else "MÔ PHỎNG (PAPER TRADING) 🧪"
    return {"status": "ok", "live_enabled": live_enabled, "message": f"Đã chuyển sang chế độ {mode_text}"}


@app.post("/api/settings/update_fees")
async def update_fees(payload: dict):
    default_tier = "MEXC_STANDARD" if state.active_exchange == "mexc" else "VIP_0"
    vip_tier = str(payload.get("vip_tier", default_tier))
    use_bnb = bool(payload.get("use_bnb_discount", False))
    state.fee_engine.set_vip_tier(vip_tier, use_bnb_discount=use_bnb)
    state.storage.save_setting("vip_tier", vip_tier)
    state.storage.save_setting("use_bnb_discount", use_bnb)
    state.storage.save_setting("custom_maker_fee", None)
    state.storage.save_setting("custom_taker_fee", None)
    await broadcast_state()
    disc_name = "MX Token" if state.active_exchange == "mexc" else "BNB"
    return {
        "status": "ok",
        "exchange": state.active_exchange,
        "vip_tier": state.fee_engine.vip_tier,
        "maker_pct": state.fee_engine.maker_fee_rate * 100,
        "taker_pct": state.fee_engine.taker_fee_rate * 100,
        "message": f"Đã cập nhật biểu phí [{state.active_exchange.upper()}] {vip_tier} ({disc_name} discount -10%: {'BẬT' if use_bnb else 'TẮT'})"
    }


@app.post("/api/settings/sync_api_fees")
async def sync_api_fees():
    api_manager = state.active_exchange_api
    res = api_manager.fetch_commission_rate(state.symbol)
    if res.get("success", False):
        m = res["maker_commission"]
        t = res["taker_commission"]
        state.fee_engine.set_custom_rates(m, t)
        state.storage.save_setting("custom_maker_fee", m)
        state.storage.save_setting("custom_taker_fee", t)
        await broadcast_state()
    return res


@app.post("/api/settings/reset_data")
@serialized_action
def reset_data(amount: float = 1000.0):
    if state.execution.live or state.current_position or state.order_manager.pending_orders:
        return {"status": "rejected", "reason": "Close/cancel exposure before resetting paper state"}
    if not math.isfinite(amount) or amount <= 0:
        return {"status": "rejected", "reason": "Invalid balance"}
    state.storage.reset_database(initial_balance=amount, symbol=state.symbol)
    state.initial_balance = amount
    state.current_balance = amount
    state.trades = []
    state.trade_memory.memory_records.clear()
    state.current_position = None
    state.total_fees = 0.0
    state.tick_count = 0
    state.last_auto_order_tick = 0
    state.last_auto_order_time = 0.0
    state.last_trade_closed_time = 0.0
    state.order_manager.orders.clear()
    state.freqtrade_protections = FreqtradeProtectionEngine(initial_balance=amount)
    state.risk_manager.reset_daily_stats(amount)
    state.persist_current_state()
    return {"status": "ok", "message": f"Đã xóa trắng lịch sử SQLite và đặt lại số dư ban đầu ${amount:,.2f}"}


@app.get("/api/data/export")
async def export_data(
    download: bool = True,
    trades: bool = True,
    memory: bool = True,
    settings: bool = True
):
    pkg = state.storage.export_full_package(
        include_trades=trades,
        include_memory=memory,
        include_settings=settings
    )
    pkg["account_state"]["current_balance"] = state.current_balance
    pkg["account_state"]["initial_balance"] = state.initial_balance
    pkg["account_state"]["total_fees"] = state.total_fees

    if download:
        content_str = json.dumps(pkg, indent=2, ensure_ascii=False)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"antigravity_backup_{ts}.json"
        return Response(
            content=content_str,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "application/json; charset=utf-8"
            }
        )
    return pkg


@app.post("/api/data/import")
@serialized_action
def import_data(payload: dict):
    if state.execution.live or state.current_position or state.order_manager.pending_orders:
        return {"status": "rejected", "reason": "Close/cancel exposure before replacing data or credentials"}
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Dữ liệu nhập không hợp lệ (cần định dạng JSON object)!")

    if "trades" not in payload and "trade_memory_records" not in payload and "settings" not in payload and "ai_lessons" not in payload:
        raise HTTPException(status_code=400, detail="Tệp sao lưu không chứa các trường dữ liệu hợp lệ (trades, settings, trade_memory_records)!")

    overwrite_balance = bool(payload.get("overwrite_balance", False))
    mode = str(payload.get("mode", "merge")).lower()
    is_overwrite = (mode in ("overwrite", "replace"))

    # If overwrite mode, purge in-memory history before restoring
    if is_overwrite:
        if "trades" in payload:
            state.trades = []
        if "trade_memory_records" in payload or "ai_lessons" in payload:
            state.trade_memory.memory_records = []

    res = state.storage.import_full_package(payload, overwrite_account_state=overwrite_balance, mode=mode)

    # Refresh in-memory state
    state.trades = state.storage.load_trades(limit=500)

    # Reload memory records
    loaded_mems = state.storage.load_trade_memory_records(limit=200)
    state.trade_memory.memory_records = []
    for m in loaded_mems:
        rec = TradeContextProfile(
            trade_id=m["trade_id"],
            direction=m["direction"],
            entry_price=m["entry_price"],
            rsi=m.get("rsi", 50.0),
            cvd_momentum=m.get("delta_momentum", "BALANCED"),
            smc_structure=m.get("smc_structure", "RANGING"),
            vwap_status=m.get("vwap_status", "EQUILIBRIUM_FAIR"),
            net_pnl=m["net_pnl"],
            outcome="WIN" if m["net_pnl"] > 0 else "LOSS",
            failure_reason=m.get("lesson_learned", "")
        )
        state.trade_memory.memory_records.append(rec)

    # Reload settings if provided
    settings = payload.get("settings") or payload.get("user_settings") or {}
    if settings:
        if "api_key" in settings:
            state.binance_api.api_key = str(settings["api_key"])
        if "api_secret" in settings:
            state.binance_api.api_secret = str(settings["api_secret"])
        if "is_testnet" in settings:
            state.binance_api.is_testnet = bool(settings["is_testnet"])
        if "mexc_api_key" in settings:
            state.mexc_api.api_key = str(settings["mexc_api_key"])
        if "mexc_api_secret" in settings:
            state.mexc_api.api_secret = str(settings["mexc_api_secret"])
        if "active_exchange" in settings:
            state.active_exchange = str(settings["active_exchange"])
            state.fee_engine.set_exchange(str(settings["active_exchange"]))
        if "vip_tier" in settings:
            state.fee_engine.set_vip_tier(str(settings["vip_tier"]), use_bnb_discount=bool(settings.get("use_bnb_discount", False)))

    if overwrite_balance and "account_state" in payload and payload["account_state"]:
        acc = payload["account_state"]
        if "current_balance" in acc:
            state.current_balance = float(acc["current_balance"])
        if "initial_balance" in acc:
            state.initial_balance = float(acc["initial_balance"])
        if "total_fees" in acc:
            state.total_fees = float(acc["total_fees"])

    state.persist_current_state()
    return res


# -------------------------------------------------------------
# HKUDS VIBE-TRADING & SWARM COUNCIL (9ROUTER) SETTINGS API
# -------------------------------------------------------------
@app.get("/api/settings/vibe")
@app.get("/api/action/get_vibe_config")
async def get_vibe_config():
    return {
        "status": "ok",
        "config": state.storage.get_vibe_config(),
        "available_models": state.ai_copilot.get_available_models(),
        "active_council": asdict(state.vibe_swarm.latest_verdict) if state.vibe_swarm.latest_verdict else None
    }


@app.post("/api/settings/vibe")
@app.post("/api/action/save_vibe_config")
@serialized_action
def save_vibe_config(payload: dict):
    enabled = bool(payload.get("enabled", True))
    min_votes = int(payload.get("min_votes", 3))
    macro_model = str(payload.get("macro_model", "ag/gemini-3.8-flash-high")).strip()
    quant_model = str(payload.get("quant_model", "ag/gemini-3.8-flash-high")).strip()
    risk_model = str(payload.get("risk_model", "ag/gemini-3.8-flash-high")).strip()
    exec_model = str(payload.get("exec_model", "ag/gemini-3.8-flash-high")).strip()

    cfg = state.storage.save_vibe_config(
        enabled=enabled,
        min_votes=min_votes,
        macro_model=macro_model,
        quant_model=quant_model,
        risk_model=risk_model,
        exec_model=exec_model
    )
    state.vibe_swarm.update_config(
        enabled=enabled,
        min_votes=min_votes,
        macro_model=macro_model,
        quant_model=quant_model,
        risk_model=risk_model,
        exec_model=exec_model
    )
    return {
        "status": "ok",
        "message": f"Đã lưu cấu hình Vibe AI Swarm Council (Bật: {enabled}, Ngưỡng: {min_votes}/4 phiếu)",
        "config": cfg
    }


@app.post("/api/action/run_vibe_swarm_debate")
@serialized_action
def run_vibe_swarm_debate():
    verdict = state.vibe_swarm.evaluate_council(
        current_price=state.live_price,
        indicators=state.indicators,
        ai_verdict=state.ai_verdict,
        ensemble_result=state.ensemble_result,
        order_research=state.order_research,
        alpha_zoo_metrics=state.vibe_alpha_zoo.get_latest_metrics(),
        visual_hft_metrics=state.visual_hft.get_metrics(),
        jesse_metrics=state.jesse_engine.compute_metrics(),
        octobot_metrics=state.octobot_consensus,
        current_position=state.current_position,
        user_instruction=state.ai_copilot.user_instruction or ""
    )
    return {"status": "ok", "verdict": asdict(verdict) if verdict else None}


if __name__ == "__main__":
    uvicorn.run("ui.server:app", host="127.0.0.1", port=8000, reload=False)
