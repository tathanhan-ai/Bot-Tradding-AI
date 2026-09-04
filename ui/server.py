"""
FastAPI Server for Binance Futures Trading Admin Dashboard
Ultra-reliable real-time market data engine with sub-second price polling and WebSocket push.
"""
import asyncio
import json
import sys
import time
import urllib.request
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
from strategy.grid_bot import FuturesGridBot
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
from dataclasses import asdict

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


class LiveTradingState:
    def __init__(self, symbol: str = "BTCUSDT", balance: float = 5000.0):
        self.symbol = symbol
        self.initial_balance = balance
        self.current_balance = balance
        self.is_running = True
        self.live_price = 0.0
        self.price_history: List[float] = []

        self.strategy_config = StrategyConfig(symbol=symbol)
        self.risk_config = RiskConfig(initial_balance=balance)
        self.strategy = MTFTrendATRStrategy(self.strategy_config)
        self.risk_manager = FuturesRiskManager(self.risk_config)
        self.fetcher = BinanceDataFetcher()

        self.ai_brain = AIQuantBrain()
        self.ai_verdict: Optional[AIRegimeVerdict] = None
        self.grid_bot = FuturesGridBot(symbol=symbol, total_balance=balance, leverage=3, num_grids=10)
        self.is_grid_active = False

        # Order Queue Manager (Market & Limit Orders)
        self.order_manager = OrderQueueManager()

        # Dynamic Leverage Strategy
        self.leverage_engine = DynamicLeverageEngine()
        self.leverage_mode = "AI_AUTO"  # 'AI_AUTO' or 'MANUAL'
        self.manual_leverage = 3
        self.leverage_advice: Optional[LeverageAdvice] = None

        self.storage = PersistentStorageManager()
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
        self.ws_latency_ms: float = 15.0
 
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
        if self.trades:
            self.shadow_account.analyze_trade_history(self.trades, self.initial_balance)

    @property
    def active_exchange_api(self):
        return self.mexc_api if self.active_exchange == "mexc" else self.binance_api

    def init_ws_engine(self):
        """Initializes direct Binance Futures WebSocket stream callbacks"""
        def on_book_ticker(bid: float, ask: float, mid: float):
            self.fee_engine.update_book(bid, ask)
            bids = [[bid - i * 0.5, 1.0 + i * 0.15] for i in range(20)]
            asks = [[ask + i * 0.5, 1.0 + i * 0.15] for i in range(20)]
            self.visual_hft.update_order_book(bids, asks)
            self.on_tick(mid)

        def on_agg_trade(price: float, qty: float, is_buyer_maker: bool, trade_time: int):
            self.order_flow_engine.add_trade(price, qty, is_buyer_maker, trade_time)
            t_sec = trade_time / 1000.0 if trade_time > 1e11 else float(trade_time)
            self.visual_hft.update_trade(price, qty, is_buyer_maker, timestamp=t_sec)

        def on_kline(k: dict):
            df_1m = self.data_map.get("1m")
            if df_1m is not None and not df_1m.empty:
                k_dt = pd.to_datetime(k["time"], unit="s")
                if k.get("is_closed", False):
                    if k_dt in df_1m.index:
                        df_1m.at[k_dt, "open"] = k["open"]
                        df_1m.at[k_dt, "high"] = k["high"]
                        df_1m.at[k_dt, "low"] = k["low"]
                        df_1m.at[k_dt, "close"] = k["close"]
                        df_1m.at[k_dt, "volume"] = k["volume"]
                    print(f"✅ [BINANCE 1M KLINE ĐÃ ĐÓNG] O:${k['open']:,.1f} H:${k['high']:,.1f} L:${k['low']:,.1f} C:${k['close']:,.1f} | Vol: {k['volume']:.2f} BTC", flush=True)
                else:
                    last_idx = df_1m.index[-1]
                    df_1m.at[last_idx, "close"] = k["close"]
                    df_1m.at[last_idx, "high"] = max(df_1m.at[last_idx, "high"], k["high"])
                    df_1m.at[last_idx, "low"] = min(df_1m.at[last_idx, "low"], k["low"])
                    df_1m.at[last_idx, "volume"] = k["volume"]

        def on_latency(latency_ms: float):
            self.ws_latency_ms = latency_ms

        self.ws_engine = BinanceFuturesWebSocketEngine(
            symbol=self.symbol,
            on_book_ticker=on_book_ticker,
            on_agg_trade=on_agg_trade,
            on_kline=on_kline,
            on_latency_update=on_latency
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
            df = self.data_map.get("1h") or self.data_map.get("15m")
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
        print(f"[UI Server] Nạp lịch sử đa khung nến (1m, 5m, 15m, 1h, 1w, 1M) cho {self.symbol}...", flush=True)
        try:
            self.data_map["1M"] = self.fetcher.fetch_klines(self.symbol, "1M", 50)
            self.data_map["1w"] = self.fetcher.fetch_klines(self.symbol, "1w", 100)
            self.data_map["1h"] = self.fetcher.fetch_klines(self.symbol, "1h", 300)
            self.data_map["15m"] = self.fetcher.fetch_klines(self.symbol, "15m", 500)
            self.data_map["5m"] = self.fetcher.fetch_klines(self.symbol, "5m", 500)
            self.data_map["1m"] = self.fetcher.fetch_klines(self.symbol, "1m", 500)
            if not self.data_map["15m"].empty:
                self.live_price = float(self.data_map["15m"]["close"].iloc[-1])
                self.price_history = [float(p) for p in self.data_map["15m"]["close"].iloc[-50:]]
                df_sig = self.strategy.generate_signals(self.data_map)
                last_row = df_sig.iloc[-1]
                self.indicators = {
                    "ema": float(last_row.get("ema_trend_macro", 0.0)),
                    "rsi": float(last_row.get("rsi", 50.0)),
                    "atr": float(last_row.get("atr", 0.0)),
                    "adx": float(last_row.get("adx", 0.0)),
                }
                self.ai_verdict = self.ai_brain.analyze(self.data_map, self.live_price, active_timeframe=self.active_timeframe)
                self.candle_confluence = self.multi_candle_engine.evaluate(self.data_map, self.live_price)
                self.update_leverage_advice()
                df_struct = self.get_structure_df()
                df_macro = self.get_macro_df()
                self.order_research = self.ai_order_researcher.research(
                    current_price=self.live_price,
                    best_bid=self.live_price - 0.5,
                    best_ask=self.live_price + 0.5,
                    spread=1.0,
                    indicators=self.indicators,
                    ai_verdict=self.ai_verdict,
                    ensemble_result=self.ensemble_result,
                    ai_cro=self.risk_manager.ai_cro,
                    current_balance=self.current_balance,
                    df_structure=df_struct,
                    df_macro=df_macro,
                    active_timeframe=self.active_timeframe,
                    candle_confluence=self.candle_confluence,
                    order_flow_verdict=self.order_flow_verdict,
                    effective_leverage=self.get_effective_leverage(),
                    current_position=self.current_position,
                    inventory_skew=self.hummingbot_skew.calculate_reservation_price(
                        self.live_price, self.current_position, self.indicators.get("atr", 200.0), self.current_balance
                    ),
                    visual_hft_metrics=self.visual_hft.get_metrics(),
                    jesse_metrics=self.jesse_engine.compute_metrics(),
                    octobot_consensus=self.octobot_consensus
                )
            print(f"[UI Server] Đã nạp dữ liệu xong. Giá ban đầu {self.symbol}: ${self.live_price:,.2f}", flush=True)
        except Exception as e:
            print(f"[UI Server] Lỗi nạp dữ liệu: {e}", flush=True)

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

    def on_tick(self, price: float):
        """Called on every live price tick"""
        self.live_price = price
        self.tick_count += 1
        self.fetch_binance_funding()

        if not self.price_history or abs(self.price_history[-1] - price) >= 0.01:
            self.price_history.append(price)
            if len(self.price_history) > 60:
                self.price_history.pop(0)

        # Update latest candle and rollover per timeframe boundaries
        now_epoch = int(datetime.now(timezone.utc).timestamp())
        for tf, tf_df in list(self.data_map.items()):
            if tf_df is not None and not tf_df.empty:
                sec = TF_SECONDS.get(tf, 60)
                candle_open_epoch = (now_epoch // sec) * sec
                candle_open_dt = pd.to_datetime(candle_open_epoch, unit='s')

                if candle_open_dt > tf_df.index[-1]:
                    # 🕯️ CÂY NẾN TRƯỚC ĐÃ ĐÓNG (NGẮT NẾN THEO KHUNG) -> MỞ NẾN MỚI
                    prev_candle = tf_df.iloc[-1]
                    print(f"🕯️ [NGẮT NẾN {tf.upper()}] Đóng nến {tf} tại ${prev_candle['close']:,.2f} | Bắt đầu nến mới tại ${price:,.2f}", flush=True)
                    new_row = pd.DataFrame([{
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 0.0,
                        "quote_volume": 0.0
                    }], index=[candle_open_dt])
                    self.data_map[tf] = pd.concat([tf_df, new_row])
                    if len(self.data_map[tf]) > 550:
                        self.data_map[tf] = self.data_map[tf].iloc[-500:]
                else:
                    # Cập nhật giá nến đang hình thành
                    last_idx = tf_df.index[-1]
                    tf_df.at[last_idx, "close"] = price
                    if price > tf_df.at[last_idx, "high"]:
                        tf_df.at[last_idx, "high"] = price
                    if price < tf_df.at[last_idx, "low"]:
                        tf_df.at[last_idx, "low"] = price

        # Match pending orders across all 7 types on live price tick
        filled_orders = self.order_manager.match_orders(
            price,
            best_bid=self.fee_engine.bid_price,
            best_ask=self.fee_engine.ask_price
        )
        for fo in filled_orders:
            is_maker_order = fo.order_type in ("LIMIT", "POST_ONLY", "SCALE_RATIO")

            if not self.current_position:
                fo_tf = getattr(fo, "timeframe", self.active_timeframe)
                struct_calc = self.ai_order_researcher.structural_calculator
                if fo.stop_loss and ((fo.direction == 1 and 0 < fo.stop_loss < price) or (fo.direction == -1 and fo.stop_loss > price)):
                    struct_sl = fo.stop_loss
                    struct_tp = fo.take_profit
                else:
                    setup = struct_calc.compute_setup(
                        side="BUY" if fo.direction == 1 else "SELL",
                        entry_price=price,
                        df_structure=self.get_structure_df(),
                        timeframe=fo_tf
                    )
                    struct_sl = setup.stop_loss
                    struct_tp = setup.take_profit
                notional = fo.margin * fo.leverage
                liq_buffer = (0.98 / fo.leverage) * price
                liq = (price - liq_buffer) if fo.direction == 1 else (price + liq_buffer)
                self.open_position(
                    direction=fo.direction,
                    price=price,
                    units=fo.units,
                    notional=notional,
                    margin=fo.margin,
                    sl=struct_sl,
                    tp=struct_tp,
                    liq=liq,
                    dt=datetime.now(),
                    is_maker=is_maker_order,
                    timeframe=getattr(fo, "timeframe", self.active_timeframe),
                    order_id=getattr(fo, "order_id", None),
                    order_type=getattr(fo, "order_type", "MARKET")
                )
                print(f"🎯 [{fo.order_type} KHỚP LỆNH] #{fo.order_id} {fo.side} {fo.symbol} [{getattr(fo, 'timeframe', self.active_timeframe)}] tại ${price:,.2f} | SL: ${struct_sl:,.1f} | TP: ${struct_tp:,.1f} | Ký quỹ: ${fo.margin:,.2f}", flush=True)

            elif self.current_position and self.current_position["direction"] == fo.direction:
                # Scale-in / Ladder DCA / TWAP Slice into existing position
                pos = self.current_position
                add_margin = fo.margin
                add_notional = fo.margin * fo.leverage
                add_units = fo.units
                total_units = pos["units"] + add_units
                total_notional = pos["notional"] + add_notional
                new_avg_entry = total_notional / total_units if total_units > 0 else pos["entry_price"]
                
                pos["entry_price"] = round(new_avg_entry, 2)
                pos["units"] = round(total_units, 4)
                pos["margin"] += add_margin
                pos["notional"] = total_notional
                pos["breakeven_price"] = self.fee_engine.calculate_breakeven_price(new_avg_entry, fo.direction)
                
                entry_fee = self.fee_engine.calculate_fee(add_notional, is_maker=is_maker_order)
                self.current_balance -= entry_fee
                self.total_fees += entry_fee
                pos["entry_fee"] += entry_fee

                # Track slice in pos["orders"]
                order_slice = {
                    "slice_id": f"ORD-{fo.order_id}",
                    "order_id": fo.order_id,
                    "order_type": getattr(fo, "order_type", "LIMIT DCA"),
                    "timeframe": getattr(fo, "timeframe", self.active_timeframe),
                    "side": fo.side,
                    "direction": fo.direction,
                    "entry_price": round(price, 2),
                    "units": round(add_units, 4),
                    "margin": round(add_margin, 2),
                    "notional": round(add_notional, 2),
                    "entry_fee": round(entry_fee, 4),
                    "fee_tier": "MAKER (0.02%)" if is_maker_order else "TAKER (0.05%)",
                    "entry_time": datetime.now().strftime("%m-%d %H:%M:%S"),
                    "unrealized_pnl": 0.0,
                    "roe_pct": 0.0,
                }
                if "orders" not in pos or not isinstance(pos["orders"], list):
                    pos["orders"] = []
                pos["orders"].append(order_slice)

                print(f"📊 [BỒI VỊ THẾ / DCA] #{fo.order_id} ({fo.order_type}): Giá vào bình quân mới ${new_avg_entry:,.2f} | Tổng Ký quỹ: ${pos['margin']:,.2f} | {fo.note}", flush=True)

        # Handle Grid Bot Execution if active
        if self.is_grid_active:
            grid_event = self.grid_bot.on_tick(price)
            if grid_event and grid_event["action"] == "GRID_PROFIT_TAKEN":
                profit = grid_event["profit"]
                self.current_balance += profit
                self.trades.append({
                    "id": len(self.trades) + 1,
                    "symbol": self.symbol,
                    "direction": "GRID ⚡",
                    "entry_time": datetime.now().strftime("%m-%d %H:%M:%S"),
                    "entry_price": grid_event["sell_price"] - (price * 0.003),
                    "exit_time": datetime.now().strftime("%m-%d %H:%M:%S"),
                    "exit_price": grid_event["sell_price"],
                    "reason": f"GRID_TP (Tầng {grid_event['level']})",
                    "pnl": profit,
                    "return_pct": round((profit / (self.current_balance * 0.07)) * 100, 2)
                })
                print(f"⚡ [GRID PROFIT] Chốt lời tầng {grid_event['level']} | +${profit:.2f}", flush=True)

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

            # AI Continuous Exit Coordination (Smart TP, Early Cut-Loss, Breakeven Lock, Trailing)
            decision = self.ai_coordinator.evaluate_position(
                pos=pos,
                current_price=price,
                indicators=self.indicators,
                ai_verdict=self.ai_verdict,
                fee_engine=self.fee_engine
            )
            pos["ai_action_status"] = decision.status_display

            if decision.action == "LOCK_BREAKEVEN":
                pos["stop_loss"] = decision.new_stop_loss
                pos["trailing_status"] = "Đã khóa Hòa Vốn (+ Phí Sàn) 🛡️"
                print(f"🛡️ [AI BREAKEVEN LOCK] {self.symbol}: Dời SL lên ${decision.new_stop_loss:,.2f} để đưa lệnh về trạng thái Risk-Free!", flush=True)

            elif decision.action == "EXPAND_TAKE_PROFIT":
                pos["stop_loss"] = decision.new_stop_loss
                pos["take_profit"] = decision.new_take_profit
                pos["trailing_status"] = f"Nới TP gồng lãi: ${decision.new_take_profit:,.1f} 🚀"
                print(f"🚀 [AI TP EXPANSION] {self.symbol}: Dời SL khóa lãi lên ${decision.new_stop_loss:,.2f} & Nới TP lên ${decision.new_take_profit:,.2f} để gồng sóng dài!", flush=True)

            elif decision.action == "UPDATE_TRAILING":
                pos["stop_loss"] = decision.new_stop_loss
                pos["trailing_status"] = f"AI Trailing: ${decision.new_stop_loss:,.1f} 🚀"
                print(f"🚀 [AI TRAILING UPDATE] {self.symbol}: Nâng mốc Trailing Stop lên ${decision.new_stop_loss:,.2f}", flush=True)

            elif decision.action == "PARTIAL_TAKE_PROFIT":
                self.close_partial_position(ratio=0.5, reason=decision.reason, is_maker=False)

            elif decision.action in ("AI_TAKE_PROFIT", "AI_CUT_LOSS"):
                self.close_position(price, decision.reason, is_maker=False)

        # AI Continuous Multi-Strategy Ensemble Execution & Coordination on every tick (4 spaces)
        self.evaluate_ensemble_automated_decision(price)

    def evaluate_ensemble_automated_decision(self, current_price: float):
        """
        AI Continuous Multi-Strategy Ensemble Execution & Coordination:
        1. When in position: checks if Ensemble Consensus flipped strongly against position -> Early Exit to lock gains.
        2. When no position: checks if Ensemble Consensus reached high conviction (|Score| >= 50.0) -> Auto Entry.
        3. When market is ranging: automatically coordinates Futures Grid Bot.
        """
        if not self.is_running:
            return

        ens = self.ensemble_result
        if not ens:
            return

        # Case 1: Open Position Protection via Ensemble Reversal
        if self.current_position:
            pos = self.current_position
            direction = pos["direction"]
            # If Long, but Ensemble flipped to Strong Short (<= -55.0)
            if direction == 1 and ens.consensus_score <= -55.0:
                self.close_position(
                    current_price,
                    f"AI ENSEMBLE ĐẢO CHIỀU 🔄 (Đồng thuận Short {ens.consensus_score:.1f} điểm, đóng Long bảo vệ vốn)"
                )
            # If Short, but Ensemble flipped to Strong Long (>= 55.0)
            elif direction == -1 and ens.consensus_score >= 55.0:
                self.close_position(
                    current_price,
                    f"AI ENSEMBLE ĐẢO CHIỀU 🔄 (Đồng thuận Long +{ens.consensus_score:.1f} điểm, đóng Short bảo vệ vốn)"
                )
            return

        # Case 2: Autonomous AI Research Execution when no position and no pending orders
        if not self.current_position and len(self.order_manager.pending_orders) == 0:
            res = self.order_research
            if not res:
                return

            now = time.time()
            direction = 1 if res.recommended_side == "BUY" else -1
            entry_ref = res.optimal_price if res.optimal_price > 0 else current_price
            target_ref = res.structural_tp if res.structural_tp > 0 else (entry_ref + 0.01 * entry_ref * direction)

            # 1. Freqtrade Protections (StoplossGuard, MaxDrawdown, Cooldown, FeeDrag)
            equity = self.current_balance + (self.current_position["unrealized_pnl"] if self.current_position else 0.0)
            is_allowed, prot_reason, prot_status = self.freqtrade_protections.validate_new_trade(
                entry_price=entry_ref,
                target_price=target_ref,
                direction=direction,
                balance=self.current_balance,
                equity=equity
            )
            if not is_allowed:
                return

            # 1.5. VisualHFT Microstructure Toxic Flow & Resilience Guard
            hft_m = self.visual_hft.get_metrics()
            if hft_m.is_toxic_flow:
                print(f"🛑 [VISUALHFT VETO] Từ chối mở lệnh {res.recommended_side}: Dòng tiền độc hại VPIN={hft_m.vpin:.2f} (> {self.visual_hft.toxic_vpin_threshold})! Cá mập đang quét thanh khoản.", flush=True)
                return
            if hft_m.liquidity_drought_warning and res.recommended_type in ("MARKET", "TAKER"):
                print(f"🛑 [VISUALHFT VETO] Từ chối lệnh Market: Thanh khoản cạn kiệt (Resilience={hft_m.market_resilience_pct:.0f}% < 35%).", flush=True)
                return

            # 1.6. OctoBot Matrix Consensus Tradability Guard
            if self.octobot_consensus and not self.octobot_consensus.is_tradable:
                print(f"🛑 [OCTOBOT MATRIX VETO] Từ chối vào lệnh: Trạng thái {self.octobot_consensus.consensus_state} ({self.octobot_consensus.summary_reason})", flush=True)
                return

            # 1.7. Jesse Expectancy Engine Edge Guard
            jesse_m = self.jesse_engine.compute_metrics()
            if jesse_m.expectancy_usdt <= -15.0 and jesse_m.current_consecutive_losses >= 3:
                print(f"🛑 [JESSE EXPECTANCY VETO] Tạm dừng vào lệnh: Kỳ vọng âm (${jesse_m.expectancy_usdt:.2f}) sau {jesse_m.current_consecutive_losses} lệnh thua liên tiếp.", flush=True)
                return

            # 1.8. Rob Carver Systematic Buffer Inertia Guard
            if res.carver_action == "HOLD" and abs(ens.consensus_score if ens else 0.0) < 65.0:
                print(f"⏸️ [ROB CARVER BUFFER] Nằm trong dải đệm trơ [{res.carver_buffer_bands[0]:.3f}, {res.carver_buffer_bands[1]:.3f}], giữ vị thế tránh phí rác.", flush=True)
                return

            # 1.9. Episodic Trade Memory Veto Check (LLM_trader)
            absorption_sig = self.order_flow_verdict.absorption_divergence if self.order_flow_verdict else "NONE"
            vwap_stat = getattr(self.ai_verdict, "vwap_status", "EQUILIBRIUM_FAIR") if self.ai_verdict else "EQUILIBRIUM_FAIR"
            curr_rsi = self.indicators.get("rsi", 50.0)
            mem_check = self.trade_memory.query_similarity_against_losses(
                candidate_direction=direction,
                candidate_rsi=curr_rsi,
                candidate_vwap_status=vwap_stat,
                candidate_absorption=absorption_sig
            )
            if not mem_check.is_safe:
                print(f"🛑 [TRADE MEMORY VETO] Từ chối mở lệnh {res.recommended_side}: {mem_check.lesson_learned}", flush=True)
                return

            # 1.10. AI Model Copilot (9Router Port 8039) Supreme Cognitive Oversight & Veto
            if self.ai_copilot.is_active:
                carver_dict = getattr(res, "carver_output", None)
                copilot_verdict = self.ai_copilot.evaluate_market(
                    current_price=current_price,
                    indicators=self.indicators,
                    ai_verdict=self.ai_verdict,
                    ensemble_result=ens,
                    order_research=res,
                    carver_metrics=carver_dict,
                    trade_memories=self.trade_memory.memory_records[-3:] if self.trade_memory.memory_records else [],
                    current_position=self.current_position,
                    visual_hft_metrics=hft_m,
                    octobot_metrics=self.octobot_consensus,
                    jesse_metrics=jesse_m
                )
                self.ai_copilot_verdict = copilot_verdict
                if copilot_verdict.decision == "VETO":
                    print(f"🛑 [9ROUTER AI COPILOT VETO] {copilot_verdict.thought_process} | Cảnh báo: {copilot_verdict.shark_trap_warning}", flush=True)
                    return
                elif copilot_verdict.decision == "ROTATE_GRID" and self.auto_grid_rotation and not self.is_grid_active:
                    self.is_grid_active = True
                    self.grid_bot.total_balance = self.current_balance
                    self.grid_bot.generate_grid(current_price)
                    print(f"🔄 [9ROUTER AI COPILOT] Khuyến nghị xoay sang Lưới Grid 10 tầng: {copilot_verdict.thought_process}", flush=True)
                    return

            # 1.11. HKUDS Vibe-Trading Swarm Council (Macro, Quant, Risk, Execution via 9Router)
            if self.vibe_swarm.enabled:
                alpha_zoo_m = self.vibe_alpha_zoo.get_latest_metrics()
                swarm_verdict = self.vibe_swarm.evaluate_council(
                    current_price=current_price,
                    indicators=self.indicators,
                    ai_verdict=self.ai_verdict,
                    ensemble_result=ens,
                    order_research=res,
                    alpha_zoo_metrics=alpha_zoo_m,
                    visual_hft_metrics=hft_m,
                    jesse_metrics=jesse_m,
                    octobot_metrics=self.octobot_consensus,
                    current_position=self.current_position,
                    user_instruction=self.ai_copilot.user_instruction or ""
                )
                if not swarm_verdict.approved:
                    print(f"🛑 [VIBE SWARM VETO] Hội đồng 4 Đặc Vụ Bác Lệnh ({swarm_verdict.approved_votes}/{self.vibe_swarm.min_votes_required} phiếu): {swarm_verdict.council_rationale}", flush=True)
                    return

            # Auto-Adaptive Grid & Trend Rotation
            if self.auto_grid_rotation and self.ai_verdict:
                h = getattr(self.ai_verdict, "hurst_exponent", 0.50)
                adx = self.indicators.get("adx", 20.0)
                regime = self.ai_verdict.regime
                
                # If Market Resilience is dangerously low (< 30%), pause grid to avoid filling during crash
                if hft_m.market_resilience_pct < 30.0 and self.is_grid_active:
                    self.is_grid_active = False
                    print(f"⚠️ [VISUALHFT GRID PAUSE] Thanh khoản bị rút mạnh (Resilience {hft_m.market_resilience_pct:.0f}%) -> Tạm dừng Lưới Grid để chống bắt dao rơi!", flush=True)
                elif (h < 0.45 and adx < 22.0) or regime == "RANGING_SIDEWAY":
                    if not self.is_grid_active and not self.current_position and hft_m.market_resilience_pct >= 35.0:
                        self.is_grid_active = True
                        self.grid_bot.total_balance = self.current_balance
                        self.grid_bot.generate_grid(current_price)
                        print(f"🔄 [AI AUTO-ROTATION] Thị trường Sideway (Hurst: {h:.2f}, ADX: {adx:.1f}) -> Tự động kích hoạt Futures Grid Bot 10 tầng quanh ${current_price:,.2f}!", flush=True)
                elif (h > 0.55 or adx > 26.0) and regime in ("TRENDING_BULL", "TRENDING_BEAR"):
                    if self.is_grid_active:
                        self.is_grid_active = False
                        print(f"🏄 [AI AUTO-ROTATION] Bùng nổ xu hướng {regime} (Hurst: {h:.2f}, ADX: {adx:.1f}) -> Tắt Lưới Grid, xoay sang bám Trend Carver!", flush=True)

            if (now - self.last_auto_order_time) < 15.0:
                return

            # 2. Conviction & Ensemble Alignment Filter (Triệt tiêu nhiễu giằng co)
            if res.win_probability < 68:
                return

            consensus = ens.consensus_score if ens else 0.0
            if res.recommended_side == "BUY" and consensus < 15.0:
                return
            elif res.recommended_side == "SELL" and consensus > -15.0:
                return

            # 3. Structural Risk-Reward & SL Validity Check
            if res.rr_ratio < 1.15:
                return

            direction = 1 if res.recommended_side == "BUY" else -1
            if direction == 1 and res.structural_sl >= entry_ref:
                return
            if direction == -1 and (res.structural_sl > 0 and res.structural_sl <= entry_ref):
                return

            self.last_auto_order_tick = self.tick_count
            self.last_auto_order_time = now
            atr = self.indicators.get("atr") or (current_price * 0.008)

            if res.recommended_type in ("POST_ONLY", "LIMIT"):
                self.order_manager.place_order(
                    order_type=res.recommended_type,
                    symbol=self.symbol,
                    side=res.recommended_side,
                    price=res.optimal_price,
                    margin=res.optimal_margin,
                    leverage=res.optimal_leverage,
                    stop_loss=res.structural_sl,
                    take_profit=res.structural_tp,
                    timeframe=self.active_timeframe,
                    note=f"AI Cấu Trúc [{self.active_timeframe}] (SL:${res.structural_sl:.0f} TP:${res.structural_tp:.0f})"
                )
                print(f"🤖 [AI TỰ ĐỘNG VÀO LỆNH] {res.recommended_type} {res.recommended_side} [{self.active_timeframe}] tại ${res.optimal_price:,.2f} | SL Cản: ${res.structural_sl:,.1f} | TP Cản: ${res.structural_tp:,.1f}", flush=True)

            elif res.recommended_type == "DUAL_BRACKET":
                sub_margin = round(res.optimal_margin * 0.5, 2)
                self.order_manager.place_order(
                    order_type="POST_ONLY",
                    symbol=self.symbol,
                    side="BUY",
                    price=res.dual_buy_price,
                    margin=sub_margin,
                    leverage=res.optimal_leverage,
                    stop_loss=round(res.dual_buy_price - 1.2 * atr, 2),
                    take_profit=round(res.dual_sell_price, 2),
                    timeframe=self.active_timeframe,
                    note=f"Biên Dưới [{self.active_timeframe}] (Long Limit 2 Đầu)"
                )
                self.order_manager.place_order(
                    order_type="POST_ONLY",
                    symbol=self.symbol,
                    side="SELL",
                    price=res.dual_sell_price,
                    margin=sub_margin,
                    leverage=res.optimal_leverage,
                    stop_loss=round(res.dual_sell_price + 1.2 * atr, 2),
                    take_profit=round(res.dual_buy_price, 2),
                    timeframe=self.active_timeframe,
                    note=f"Biên Trên [{self.active_timeframe}] (Short Limit 2 Đầu)"
                )
                print(f"🤖 [AI RẢI LỆNH 2 ĐẦU BIÊN] Buy @ ${res.dual_buy_price:,.1f} & Sell @ ${res.dual_sell_price:,.1f} [{self.active_timeframe}]", flush=True)

            elif res.recommended_type == "SCALE_RATIO":
                step_pct = 0.003
                ratios = [0.20, 0.30, 0.50]
                for idx, ratio in enumerate(ratios):
                    sub_margin = round(res.optimal_margin * ratio, 2)
                    sub_price = res.optimal_price * (1.0 - (idx * step_pct)) if direction == 1 else res.optimal_price * (1.0 + (idx * step_pct))
                    self.order_manager.place_order(
                        order_type="SCALE_RATIO",
                        symbol=self.symbol,
                        side=res.recommended_side,
                        price=round(sub_price, 2),
                        margin=sub_margin,
                        leverage=res.optimal_leverage,
                        stop_loss=res.structural_sl,
                        take_profit=res.structural_tp,
                        timeframe=self.active_timeframe,
                        note=f"Tầng {idx+1}/3 [{self.active_timeframe}] ({int(ratio*100)}% vốn)"
                    )
                print(f"🤖 [AI TỰ ĐỘNG RẢI LỆNH THANG] 3 tầng SCALE_RATIO [{self.active_timeframe}] quanh ${res.optimal_price:,.2f} | SL: ${res.structural_sl:,.1f} | TP: ${res.structural_tp:,.1f}", flush=True)

            elif res.recommended_type == "CONDITIONAL":
                self.order_manager.place_order(
                    order_type="CONDITIONAL",
                    symbol=self.symbol,
                    side=res.recommended_side,
                    price=res.optimal_price,
                    margin=res.optimal_margin,
                    leverage=res.optimal_leverage,
                    trigger_price=res.optimal_trigger_price,
                    trigger_condition=res.optimal_trigger_cond,
                    stop_loss=res.structural_sl,
                    take_profit=res.structural_tp,
                    timeframe=self.active_timeframe,
                    note=f"AI Trigger [{self.active_timeframe}] ({res.optimal_trigger_cond} ${res.optimal_trigger_price:,.1f})"
                )
                print(f"🤖 [AI TỰ ĐỘNG ĐẶT LỆNH ĐIỀU KIỆN] Kích hoạt khi {res.optimal_trigger_cond} ${res.optimal_trigger_price:,.2f} [{self.active_timeframe}]", flush=True)

            elif res.recommended_type == "TRAILING_STOP":
                self.order_manager.place_order(
                    order_type="TRAILING_STOP",
                    symbol=self.symbol,
                    side=res.recommended_side,
                    price=current_price,
                    margin=res.optimal_margin,
                    leverage=res.optimal_leverage,
                    callback_pct=res.optimal_callback_pct,
                    stop_loss=res.structural_sl,
                    take_profit=res.structural_tp,
                    timeframe=self.active_timeframe,
                    note=f"AI Trailing [{self.active_timeframe}] ({res.optimal_callback_pct}%)"
                )
                print(f"🤖 [AI TỰ ĐỘNG ĐẶT TRAILING STOP] Callback {res.optimal_callback_pct}% [{self.active_timeframe}]", flush=True)

            elif res.recommended_type == "TWAP":
                self.order_manager.place_order(
                    order_type="TWAP",
                    symbol=self.symbol,
                    side=res.recommended_side,
                    price=current_price,
                    margin=res.optimal_margin,
                    leverage=res.optimal_leverage,
                    twap_slices=res.optimal_twap_slices,
                    twap_interval_ticks=3,
                    stop_loss=res.structural_sl,
                    take_profit=res.structural_tp,
                    timeframe=self.active_timeframe,
                    note=f"AI TWAP [{self.active_timeframe}] ({res.optimal_twap_slices} lát)"
                )
                print(f"🤖 [AI TỰ ĐỘNG CHIA TWAP] {res.optimal_twap_slices} lát cắt [{self.active_timeframe}]", flush=True)

            elif res.recommended_type == "MARKET":
                exec_price = self.fee_engine.get_execution_price("MARKET", res.recommended_side)
                if exec_price <= 0:
                    exec_price = current_price
                notional = res.optimal_margin * res.optimal_leverage
                units = notional / exec_price
                sl = res.structural_sl if res.structural_sl > 0 else (exec_price - 1.5 * atr if direction == 1 else exec_price + 1.5 * atr)
                tp = res.structural_tp if res.structural_tp > 0 else (exec_price + 2.5 * atr if direction == 1 else exec_price - 2.5 * atr)
                liq_buffer = (0.98 / res.optimal_leverage) * exec_price
                liq = exec_price - liq_buffer if direction == 1 else exec_price + liq_buffer
                self.open_position(
                    direction=direction,
                    price=exec_price,
                    units=units,
                    notional=notional,
                    margin=res.optimal_margin,
                    sl=sl,
                    tp=tp,
                    liq=liq,
                    dt=datetime.now(),
                    is_maker=False,
                    timeframe=self.active_timeframe
                )
                print(f"🤖 [AI TỰ ĐỘNG VÀO LỆNH MARKET] {res.recommended_side} [{self.active_timeframe}] tại ${exec_price:,.2f} | SL: ${sl:,.1f} | TP: ${tp:,.1f}", flush=True)

    def persist_current_state(self):
        try:
            peak = getattr(self.freqtrade_protections.max_drawdown_guard, "peak_balance", self.current_balance)
            self.storage.save_account_state(
                symbol=self.symbol,
                initial_balance=self.initial_balance,
                current_balance=self.current_balance,
                peak_balance=peak,
                total_fees=self.total_fees,
                is_running=self.is_running,
                active_timeframe=self.active_timeframe,
                leverage_mode=self.leverage_mode,
                manual_leverage=self.manual_leverage,
                current_position=self.current_position
            )
        except Exception as e:
            print(f"⚠️ [PERSISTENCE ERROR] Không thể ghi trạng thái vào SQLite: {e}", flush=True)

    def close_partial_position(self, ratio: float = 0.5, reason: str = "CHỐT LỜI 50% TẠI TP1 💰", is_maker: bool = False):
        if not self.current_position:
            return None
        pos = self.current_position
        direction = pos["direction"]
        exit_price = self.live_price
        pos_tf = pos.get("timeframe", self.active_timeframe)

        close_units = pos["units"] * ratio
        close_margin = pos["margin"] * ratio
        close_entry_fee = pos["entry_fee"] * ratio

        gross_pnl = (exit_price - pos["entry_price"]) * close_units * direction
        exit_fee = self.fee_engine.calculate_fee(exit_price * close_units, is_maker=is_maker)
        total_close_fee = close_entry_fee + exit_fee
        net_pnl = gross_pnl - exit_fee

        self.current_balance += (gross_pnl - exit_fee)
        self.total_fees += exit_fee
        self.risk_manager.update_balance(self.current_balance)
        self.risk_manager.ai_cro.record_trade_result(net_pnl)

        trade_record = {
            "id": len(self.trades) + 1,
            "symbol": self.symbol,
            "timeframe": pos_tf,
            "direction": f"{'LONG' if direction == 1 else 'SHORT'} (CHỐT {int(ratio*100)}%)",
            "entry_time": pos["entry_time"],
            "entry_price": pos["entry_price"],
            "breakeven_price": pos["breakeven_price"],
            "exit_time": datetime.now().strftime("%m-%d %H:%M:%S"),
            "exit_price": exit_price,
            "fee": round(total_close_fee, 2),
            "reason": reason,
            "pnl": round(gross_pnl - total_close_fee, 2),
            "return_pct": round(((gross_pnl - total_close_fee) / close_margin) * 100.0, 2)
        }
        self.trades.append(trade_record)
        self.storage.save_trade(trade_record)

        # Update remaining position
        pos["units"] -= close_units
        pos["margin"] -= close_margin
        pos["notional"] = pos["entry_price"] * pos["units"]
        pos["entry_fee"] -= close_entry_fee
        pos["partial_tp_done"] = True
        pos["is_risk_free"] = True
        # Pull SL of remaining position to Breakeven
        pos["stop_loss"] = pos["breakeven_price"]
        pos["trailing_status"] = f"ĐÃ CHỐT {int(ratio*100)}% 💰 | SL KHÓA HÒA VỐN (FREE RIDE)"

        self.persist_current_state()
        print(f"💰 [CHỐT LỜI TỪNG PHẦN {int(ratio*100)}%] {self.symbol} [{pos_tf}] tại ${exit_price:,.2f} | Đút túi: ${trade_record['pnl']:+,.2f} | 50% còn lại thả rông gồng lãi với SL hòa vốn!", flush=True)
        return trade_record

    def lock_breakeven_now(self):
        if not self.current_position:
            return False
        pos = self.current_position
        pos["stop_loss"] = pos["breakeven_price"]
        pos["is_risk_free"] = True
        pos["trailing_status"] = "🛡️ ĐÃ KHÓA HÒA VỐN (RISK-FREE)"
        self.persist_current_state()
        print(f"🛡️ [THỦ CÔNG / AI] Dời SL về Entry + Phí tại ${pos['breakeven_price']:,.2f}", flush=True)
        return True

    def set_manual_tp_sl(self, sl: Optional[float], tp: Optional[float], lock_manual: bool = True):
        if not self.current_position:
            return False, "Không có vị thế nào đang mở để chỉnh TP/SL!"

        pos = self.current_position
        live = self.live_price
        direction = pos["direction"]

        if sl is not None and sl > 0:
            if direction == 1 and sl >= live:
                return False, f"Lệnh LONG: Stop Loss (${sl:,.2f}) phải nhỏ hơn giá hiện tại (${live:,.2f})!"
            if direction == -1 and sl <= live:
                return False, f"Lệnh SHORT: Stop Loss (${sl:,.2f}) phải lớn hơn giá hiện tại (${live:,.2f})!"
            pos["stop_loss"] = round(float(sl), 2)
            pos["initial_risk"] = abs(pos["entry_price"] - pos["stop_loss"])

        if tp is not None and tp > 0:
            if direction == 1 and tp <= live:
                return False, f"Lệnh LONG: Take Profit (${tp:,.2f}) phải lớn hơn giá hiện tại (${live:,.2f})!"
            if direction == -1 and tp >= live:
                return False, f"Lệnh SHORT: Take Profit (${tp:,.2f}) phải nhỏ hơn giá hiện tại (${live:,.2f})!"
            pos["take_profit"] = round(float(tp), 2)

        pos["is_manual_tpsl"] = lock_manual
        if lock_manual:
            pos["trailing_status"] = "🔒 ĐÃ KHÓA TP/SL THỦ CÔNG"
            pos["ai_action_status"] = "🔒 Chỉnh Tay (Manual Locked)"

        self.persist_current_state()
        print(f"🛠️ [CHỈNH TAY TP/SL] SL: ${pos.get('stop_loss', 0):,.2f} | TP: ${pos.get('take_profit', 0):,.2f} | Khóa thủ công: {lock_manual}", flush=True)
        return True, "Cập nhật TP/SL thành công!"

    def close_order_slice(self, slice_id: str, exit_price: Optional[float] = None, reason: str = "ĐÓNG LỆNH LẺ THỦ CÔNG", is_maker: bool = False):
        if not self.current_position or not self.current_position.get("orders"):
            return False, "Không có vị thế hoặc lệnh nào đang mở!"

        pos = self.current_position
        orders = pos.get("orders", [])
        target_idx = None
        target_slice = None

        for idx, ord_item in enumerate(orders):
            if str(ord_item.get("slice_id")) == str(slice_id) or str(ord_item.get("order_id")) == str(slice_id):
                target_idx = idx
                target_slice = ord_item
                break

        if target_slice is None:
            return False, f"Không tìm thấy lệnh #{slice_id} trong vị thế!"

        # If this is the only slice remaining, close the entire position
        if len(orders) <= 1:
            price_to_close = exit_price or self.live_price
            self.close_position(price_to_close, reason=reason, is_maker=is_maker)
            return True, f"Đã đóng toàn bộ vị thế do đóng lệnh cuối cùng #{slice_id}!"

        price_to_close = exit_price or self.live_price
        direction = target_slice["direction"]
        units = target_slice["units"]
        margin = target_slice["margin"]
        entry_price = target_slice["entry_price"]

        gross_pnl = (price_to_close - entry_price) * units * direction
        exit_fee = self.fee_engine.calculate_fee(price_to_close * units, is_maker=is_maker)
        net_pnl = gross_pnl - exit_fee
        total_trade_fees = target_slice.get("entry_fee", 0.0) + exit_fee

        self.current_balance += gross_pnl - exit_fee
        self.total_fees += exit_fee
        self.risk_manager.update_balance(self.current_balance)
        self.risk_manager.ai_cro.record_trade_result(net_pnl)

        trade_record = {
            "id": len(self.trades) + 1,
            "symbol": self.symbol,
            "timeframe": target_slice.get("timeframe", self.active_timeframe),
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_time": target_slice.get("entry_time", pos.get("entry_time", "")),
            "entry_price": entry_price,
            "breakeven_price": pos.get("breakeven_price", entry_price),
            "exit_time": datetime.now().strftime("%m-%d %H:%M:%S"),
            "exit_price": price_to_close,
            "fee": round(total_trade_fees, 2),
            "reason": f"{reason} (#{slice_id})",
            "pnl": round(gross_pnl - total_trade_fees, 2),
            "return_pct": round(((gross_pnl - total_trade_fees) / margin) * 100.0, 2) if margin > 0 else 0.0
        }
        self.trades.append(trade_record)
        self.storage.save_trade(trade_record)

        # Record learned memory for slice closure
        smc_dict = {
            "structure": self.ai_verdict.smc_structure if self.ai_verdict else "RANGING",
            "demand_zone": list(self.ai_verdict.demand_zone) if (self.ai_verdict and self.ai_verdict.demand_zone) else None,
            "supply_zone": list(self.ai_verdict.supply_zone) if (self.ai_verdict and self.ai_verdict.supply_zone) else None,
            "liquidity_sweep": self.ai_verdict.last_sweep_info if self.ai_verdict else "NONE"
        } if self.ai_verdict else None

        vwap_dict = {
            "vwap_status": self.ai_verdict.vwap_status if self.ai_verdict else "EQUILIBRIUM_FAIR"
        } if self.ai_verdict else None

        of_dict = {
            "absorption_signal": self.order_flow_verdict.absorption_divergence if self.order_flow_verdict else "NONE",
            "delta_momentum": self.order_flow_verdict.delta_momentum if self.order_flow_verdict else "BALANCED"
        } if self.order_flow_verdict else None

        self.trade_memory.record_trade_outcome(
            trade_id=trade_record["id"],
            direction=direction,
            entry_price=entry_price,
            indicators=self.indicators,
            of_data=of_dict,
            smc_data=smc_dict,
            vwap_data=vwap_dict,
            net_pnl=trade_record["pnl"],
            exit_reason=trade_record["reason"]
        )

        try:
            last_mem = self.trade_memory.memory_records[-1] if self.trade_memory.memory_records else None
            self.storage.save_trade_memory_record(
                trade_id=trade_record["id"],
                direction=direction,
                entry_price=entry_price,
                net_pnl=trade_record["pnl"],
                exit_reason=trade_record["reason"],
                rsi=self.indicators.get("rsi", 50.0),
                vwap_status=vwap_dict.get("vwap_status", "EQUILIBRIUM_FAIR") if vwap_dict else "EQUILIBRIUM_FAIR",
                absorption_signal=of_dict.get("absorption_signal", "NONE") if of_dict else "NONE",
                delta_momentum=of_dict.get("delta_momentum", "BALANCED") if of_dict else "BALANCED",
                smc_structure=smc_dict.get("structure", "RANGING") if smc_dict else "RANGING",
                lesson_learned=last_mem.failure_reason if (last_mem and last_mem.failure_reason) else trade_record["reason"]
            )
        except Exception as e:
            print(f"⚠️ [MEMORY SAVE ERROR - SLICE] {e}", flush=True)

        # Remove slice
        orders.pop(target_idx)
        pos["orders"] = orders

        # Re-aggregate remaining position metrics
        rem_units = sum(o["units"] for o in orders)
        rem_margin = sum(o["margin"] for o in orders)
        rem_notional = sum(o["notional"] for o in orders)
        rem_entry_fee = sum(o.get("entry_fee", 0.0) for o in orders)
        avg_entry = rem_notional / rem_units if rem_units > 0 else pos["entry_price"]

        pos["units"] = round(rem_units, 4)
        pos["margin"] = round(rem_margin, 2)
        pos["notional"] = round(rem_notional, 2)
        pos["entry_price"] = round(avg_entry, 2)
        pos["entry_fee"] = round(rem_entry_fee, 4)
        pos["breakeven_price"] = self.fee_engine.calculate_breakeven_price(avg_entry, pos["direction"])

        self.persist_current_state()
        print(f"✂️ [ĐÓNG LỆNH LẺ] #{slice_id} tại ${price_to_close:,.2f} | PnL: ${trade_record['pnl']:,.2f} | Còn lại {len(orders)} lệnh | Giá TB mới: ${pos['entry_price']:,.2f}", flush=True)
        return True, f"Đã đóng thành công lệnh #{slice_id} (PnL: ${trade_record['pnl']:+,.2f})!"

    def update_order_slice(
        self,
        slice_id: str,
        entry_price: Optional[float] = None,
        units: Optional[float] = None,
        margin: Optional[float] = None,
        timeframe: Optional[str] = None,
        order_type: Optional[str] = None
    ) -> tuple[bool, str]:
        if not self.current_position or not self.current_position.get("orders"):
            return False, "Không có vị thế hoặc lệnh nào đang mở để sửa!"

        pos = self.current_position
        orders = pos.get("orders", [])
        target_slice = None

        for ord_item in orders:
            if str(ord_item.get("slice_id")) == str(slice_id) or str(ord_item.get("order_id")) == str(slice_id):
                target_slice = ord_item
                break

        if target_slice is None:
            return False, f"Không tìm thấy lệnh #{slice_id} trong vị thế!"

        # Apply modifications
        if entry_price is not None and float(entry_price) > 0:
            target_slice["entry_price"] = round(float(entry_price), 2)

        if units is not None and float(units) > 0:
            target_slice["units"] = round(float(units), 4)

        eff_lev = self.get_effective_leverage()
        if margin is not None and float(margin) > 0:
            target_slice["margin"] = round(float(margin), 2)
        elif units is not None and float(units) > 0:
            target_slice["margin"] = round((target_slice["entry_price"] * target_slice["units"]) / eff_lev, 2)

        target_slice["notional"] = round(target_slice["entry_price"] * target_slice["units"], 2)

        if timeframe:
            target_slice["timeframe"] = str(timeframe).lower()

        if order_type:
            target_slice["order_type"] = str(order_type).upper()

        # Re-aggregate entire position
        total_units = sum(o["units"] for o in orders)
        total_margin = sum(o["margin"] for o in orders)
        total_notional = sum(o["notional"] for o in orders)
        total_entry_fee = sum(o.get("entry_fee", 0.0) for o in orders)
        avg_entry = total_notional / total_units if total_units > 0 else pos["entry_price"]

        pos["units"] = round(total_units, 4)
        pos["margin"] = round(total_margin, 2)
        pos["notional"] = round(total_notional, 2)
        pos["entry_price"] = round(avg_entry, 2)
        pos["entry_fee"] = round(total_entry_fee, 4)
        pos["breakeven_price"] = self.fee_engine.calculate_breakeven_price(avg_entry, pos["direction"])

        # Re-calculate liquidation price
        mmr = 0.005  # 0.5% Maintenance Margin Rate for BTC
        if pos["direction"] == 1:
            pos["liq_price"] = round(avg_entry * (1.0 - (1.0 / eff_lev) + mmr), 2)
        else:
            pos["liq_price"] = round(avg_entry * (1.0 + (1.0 / eff_lev) - mmr), 2)

        # Update real-time unrealized pnl for target_slice
        if self.live_price > 0:
            pnl = (self.live_price - target_slice["entry_price"]) * target_slice["units"] * target_slice["direction"]
            target_slice["unrealized_pnl"] = round(pnl, 2)
            target_slice["roe_pct"] = round((pnl / target_slice["margin"]) * 100.0, 2) if target_slice["margin"] > 0 else 0.0

        self.persist_current_state()
        print(f"✏️ [SỬA LỆNH THÀNH PHẦN] #{slice_id}: Giá vào ${target_slice['entry_price']:,.2f} | Khối lượng: {target_slice['units']} | Giá TB vị thế mới: ${pos['entry_price']:,.2f}", flush=True)
        return True, f"Đã cập nhật thành công lệnh #{slice_id}!"

    def open_position(self, direction, price, units, notional, margin, sl, tp, liq, dt, is_maker: bool = False, timeframe: str = "15m", order_id: Optional[str] = None, order_type: str = "MARKET"):
        entry_fee = self.fee_engine.calculate_fee(notional, is_maker=is_maker)
        breakeven = self.fee_engine.calculate_breakeven_price(price, direction, entry_is_maker=is_maker, exit_is_maker=False)
        self.current_balance -= entry_fee
        self.total_fees += entry_fee
        used_tf = timeframe or self.active_timeframe

        ord_id = str(order_id) if order_id is not None else f"ORD-{int(time.time() * 1000) % 1000000}"
        initial_order_slice = {
            "slice_id": ord_id,
            "order_id": ord_id,
            "order_type": order_type,
            "timeframe": used_tf,
            "side": "LONG" if direction == 1 else "SHORT",
            "direction": direction,
            "entry_price": round(price, 2),
            "units": round(units, 4),
            "margin": round(margin, 2),
            "notional": round(notional, 2),
            "entry_fee": round(entry_fee, 4),
            "fee_tier": "MAKER (0.02%)" if is_maker else "TAKER (0.05%)",
            "entry_time": dt.strftime("%m-%d %H:%M:%S") if isinstance(dt, datetime) else str(dt),
            "unrealized_pnl": 0.0,
            "roe_pct": 0.0,
        }

        self.current_position = {
            "direction": direction,
            "entry_price": price,
            "breakeven_price": breakeven,
            "units": units,
            "notional": notional,
            "margin": margin,
            "stop_loss": sl,
            "take_profit": tp,
            "liq_price": liq,
            "initial_risk": abs(price - sl),
            "peak_price": price,
            "entry_fee": entry_fee,
            "fee_tier": "MAKER (0.02%)" if is_maker else "TAKER (0.05%)",
            "entry_time": dt.strftime("%m-%d %H:%M:%S") if isinstance(dt, datetime) else str(dt),
            "open_timestamp": time.time(),
            "timeframe": used_tf,
            "unrealized_pnl": 0.0,
            "trailing_status": "Chờ đạt +1.0R",
            "is_manual_tpsl": False,
            "orders": [initial_order_slice]
        }
        self.persist_current_state()
        side_str = "LONG 🟢" if direction == 1 else "SHORT 🔴"
        print(f"🚀 [VỊ THẾ MỞ] {side_str} {self.symbol} [{used_tf}] tại ${price:,.2f} | Hòa vốn: ${breakeven:,.2f} | Phí vào: ${entry_fee:.2f} | SL: ${sl:,.2f} | TP: ${tp:,.2f}", flush=True)

    def close_position(self, exit_price: float, reason: str, is_maker: bool = False):
        if not self.current_position:
            return

        pos = self.current_position
        direction = pos["direction"]
        pos_tf = pos.get("timeframe", self.active_timeframe)
        gross_pnl = (exit_price - pos["entry_price"]) * pos["units"] * direction
        exit_fee = self.fee_engine.calculate_fee(exit_price * pos["units"], is_maker=is_maker)
        total_trade_fees = pos["entry_fee"] + exit_fee
        net_pnl = gross_pnl - exit_fee  # Entry fee already deducted from balance at open

        self.current_balance += gross_pnl - exit_fee
        self.total_fees += exit_fee
        self.last_trade_closed_time = time.time()
        self.risk_manager.update_balance(self.current_balance)
        self.risk_manager.ai_cro.record_trade_result(net_pnl)

        trade_record = {
            "id": len(self.trades) + 1,
            "symbol": self.symbol,
            "timeframe": pos_tf,
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_time": pos["entry_time"],
            "entry_price": pos["entry_price"],
            "breakeven_price": pos["breakeven_price"],
            "exit_time": datetime.now().strftime("%m-%d %H:%M:%S"),
            "exit_price": exit_price,
            "fee": round(total_trade_fees, 2),
            "reason": reason,
            "pnl": round(gross_pnl - total_trade_fees, 2),
            "return_pct": round(((gross_pnl - total_trade_fees) / pos["margin"]) * 100.0, 2)
        }
        self.trades.append(trade_record)
        self.storage.save_trade(trade_record)
        self.freqtrade_protections.on_trade_closed(trade_record)

        # Record outcome in Jesse Expectancy Engine
        self.jesse_engine.record_trade(trade_record["pnl"])

        # Record context profile in Episodic Trade Memory Bank
        smc_dict = {
            "structure": self.ai_verdict.smc_structure if self.ai_verdict else "RANGING",
            "demand_zone": list(self.ai_verdict.demand_zone) if (self.ai_verdict and self.ai_verdict.demand_zone) else None,
            "supply_zone": list(self.ai_verdict.supply_zone) if (self.ai_verdict and self.ai_verdict.supply_zone) else None,
            "liquidity_sweep": self.ai_verdict.last_sweep_info if self.ai_verdict else "NONE"
        } if self.ai_verdict else None

        vwap_dict = {
            "vwap_status": self.ai_verdict.vwap_status if self.ai_verdict else "EQUILIBRIUM_FAIR"
        } if self.ai_verdict else None

        of_dict = {
            "absorption_signal": self.order_flow_verdict.absorption_divergence if self.order_flow_verdict else "NONE",
            "delta_momentum": self.order_flow_verdict.delta_momentum if self.order_flow_verdict else "BALANCED"
        } if self.order_flow_verdict else None

        self.trade_memory.record_trade_outcome(
            trade_id=trade_record["id"],
            direction=direction,
            entry_price=pos["entry_price"],
            indicators=self.indicators,
            of_data=of_dict,
            smc_data=smc_dict,
            vwap_data=vwap_dict,
            net_pnl=trade_record["pnl"],
            exit_reason=reason
        )

        # Persist learned memory record to SQLite
        try:
            last_mem = self.trade_memory.memory_records[-1] if self.trade_memory.memory_records else None
            self.storage.save_trade_memory_record(
                trade_id=trade_record["id"],
                direction=direction,
                entry_price=pos["entry_price"],
                net_pnl=trade_record["pnl"],
                exit_reason=reason,
                rsi=self.indicators.get("rsi", 50.0),
                vwap_status=vwap_dict.get("vwap_status", "EQUILIBRIUM_FAIR") if vwap_dict else "EQUILIBRIUM_FAIR",
                absorption_signal=of_dict.get("absorption_signal", "NONE") if of_dict else "NONE",
                delta_momentum=of_dict.get("delta_momentum", "BALANCED") if of_dict else "BALANCED",
                smc_structure=smc_dict.get("structure", "RANGING") if smc_dict else "RANGING",
                lesson_learned=last_mem.failure_reason if (last_mem and last_mem.failure_reason) else reason
            )
        except Exception as e:
            print(f"⚠️ [MEMORY SAVE ERROR] {e}", flush=True)

        print(f"🏁 [VỊ THẾ ĐÓNG] {self.symbol} [{pos_tf}] tại ${exit_price:,.2f} | Lãi ròng sau phí: ${trade_record['pnl']:+,.2f} (Phí sàn: ${total_trade_fees:.2f}) | {reason}", flush=True)
        self.current_position = None
        self.persist_current_state()

    def get_state_dict(self):
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
            "grid_total_profit": self.grid_bot.total_grid_profit,
            "grid_levels": [
                {"id": g.level_id, "buy": g.buy_price, "sell": g.sell_price, "status": g.status} for g in self.grid_bot.grids
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
                "recommended_type": self.order_research.recommended_type if self.order_research else "POST_ONLY",
                "recommended_side": self.order_research.recommended_side if self.order_research else "BUY",
                "optimal_price": self.order_research.optimal_price if self.order_research else self.live_price,
                "optimal_margin": self.order_research.optimal_margin if self.order_research else 100.0,
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
                "win_probability": self.order_research.win_probability if self.order_research else 85,
                "research_rationale": self.order_research.research_rationale if self.order_research else "Đang nghiên cứu điều kiện thị trường...",
                "dual_buy_price": self.order_research.dual_buy_price if self.order_research else 0.0,
                "dual_sell_price": self.order_research.dual_sell_price if self.order_research else 0.0,
                "execution_horizon": getattr(self.order_research, "execution_horizon", "IMMEDIATE"),
                "carver_contracts": getattr(self.order_research, "carver_contracts", 0.0),
                "carver_action": getattr(self.order_research, "carver_action", "HOLD"),
                "vpin": getattr(self.order_research, "vpin", 0.35),
                "toxicity_regime": getattr(self.order_research, "toxicity_regime", "CLEAN"),
                "market_resilience_pct": getattr(self.order_research, "market_resilience_pct", 85.0),
                "lob_imbalance_20": getattr(self.order_research, "lob_imbalance_20", 0.0),
                "kelly_multiplier": getattr(self.order_research, "kelly_multiplier", 1.0),
                "octobot_tradable": getattr(self.order_research, "octobot_tradable", True)
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
                "decision": self.ai_copilot_verdict.decision if self.ai_copilot_verdict else "APPROVE",
                "confidence": self.ai_copilot_verdict.confidence if self.ai_copilot_verdict else 88,
                "market_regime_sentiment": self.ai_copilot_verdict.market_regime_sentiment if self.ai_copilot_verdict else "CHÂN TRỜI TÍCH LŨY",
                "shark_trap_warning": self.ai_copilot_verdict.shark_trap_warning if self.ai_copilot_verdict else "Không phát hiện bẫy thanh khoản bất thường",
                "thought_process": self.ai_copilot_verdict.thought_process if self.ai_copilot_verdict else "Mô hình AI 9Router sẵn sàng thẩm định sâu...",
                "strategic_advice": self.ai_copilot_verdict.strategic_advice if self.ai_copilot_verdict else "Theo dõi dải đệm Carver và rải lưới Grid khi Sideway.",
                "user_instruction_feedback": self.ai_copilot_verdict.user_instruction_feedback if self.ai_copilot_verdict else (self.ai_copilot.user_instruction or "Chưa có chỉ thị riêng từ bạn."),
                "model_used": self.ai_copilot_verdict.model_used if self.ai_copilot_verdict else f"9router/{self.ai_copilot.default_model}",
                "active_model": self.ai_copilot.default_model,
                "available_models": self.ai_copilot.get_available_models(),
                "gateway_connected": self.ai_copilot.check_gateway_health(),
                "gateway_url": self.ai_copilot.gateway_url,
                "user_instruction": self.ai_copilot.user_instruction,
                "active_intel_tab": self.storage.get_setting("active_intel_tab", "intel-copilot"),
                "timestamp": self.ai_copilot_verdict.timestamp if self.ai_copilot_verdict else datetime.now().strftime("%H:%M:%S")
            },
            "order_flow": {
                "current_cvd": (self.order_flow_verdict or self.order_flow_engine.evaluate(self.live_price)).current_cvd,
                "buy_volume_1m": (self.order_flow_verdict or self.order_flow_engine.evaluate(self.live_price)).buy_volume_1m,
                "sell_volume_1m": (self.order_flow_verdict or self.order_flow_engine.evaluate(self.live_price)).sell_volume_1m,
                "buy_ratio_pct": (self.order_flow_verdict or self.order_flow_engine.evaluate(self.live_price)).buy_ratio_pct,
                "delta_momentum": (self.order_flow_verdict or self.order_flow_engine.evaluate(self.live_price)).delta_momentum,
                "absorption_divergence": (self.order_flow_verdict or self.order_flow_engine.evaluate(self.live_price)).absorption_divergence,
                "flow_rationale": (self.order_flow_verdict or self.order_flow_engine.evaluate(self.live_price)).flow_rationale,
                "latency_ms": self.ws_latency_ms
            },
            "visual_hft": {
                "vpin": round(self.visual_hft.latest_metrics.vpin, 3),
                "toxicity_regime": self.visual_hft.latest_metrics.toxicity_regime,
                "is_toxic_flow": self.visual_hft.latest_metrics.is_toxic_flow,
                "lob_imbalance_top1": round(self.visual_hft.latest_metrics.lob_imbalance_top1, 3),
                "lob_imbalance_5": round(self.visual_hft.latest_metrics.lob_imbalance_5, 3),
                "lob_imbalance_20": round(self.visual_hft.latest_metrics.lob_imbalance_20, 3),
                "bid_depth_usdt": round(self.visual_hft.latest_metrics.bid_depth_usdt, 2),
                "ask_depth_usdt": round(self.visual_hft.latest_metrics.ask_depth_usdt, 2),
                "book_pressure": self.visual_hft.latest_metrics.book_pressure,
                "market_resilience_pct": round(self.visual_hft.latest_metrics.market_resilience_pct, 1),
                "liquidity_drought_warning": self.visual_hft.latest_metrics.liquidity_drought_warning,
                "ott_ratio": round(self.visual_hft.latest_metrics.ott_ratio, 1),
                "spoofing_detected": self.visual_hft.latest_metrics.spoofing_detected,
                "spoofing_side": self.visual_hft.latest_metrics.spoofing_side,
                "microstructure_score": round(self.visual_hft.latest_metrics.microstructure_score, 1),
                "execution_safety_status": self.visual_hft.latest_metrics.execution_safety_status,
                "hft_rationale": self.visual_hft.latest_metrics.hft_rationale
            },
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
                "profit_factor": self.jesse_engine.compute_metrics().profit_factor,
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
                "latest_council": asdict(self.vibe_swarm.latest_verdict) if self.vibe_swarm.latest_verdict else None,
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
    """
    Sub-millisecond WebSocket push loop to connected browser clients.
    Runs at smooth 12.5 FPS (every 80ms) to ensure zero browser UI lag.
    """
    while True:
        try:
            if connected_clients and state.live_price > 0:
                state.order_flow_verdict = state.order_flow_engine.evaluate(state.live_price)
                await broadcast_state()
        except Exception:
            pass
        await asyncio.sleep(0.08)


async def ai_quant_background_loop():
    """
    AI Quant Brain continuous optimization loop.
    Evaluates indicators, institutional SMC zones, VWAP deviation, Ensemble weights,
    order execution research, and CRO risk profile every 1.0s.
    """
    while True:
        try:
            if state.live_price > 0:
                state.update_indicators()
                state.ai_verdict = state.ai_brain.analyze(
                    state.data_map, state.live_price, spread=state.fee_engine.spread, active_timeframe=state.active_timeframe
                )
                state.update_leverage_advice()

                regime = state.ai_verdict.regime if state.ai_verdict else "RANGING_SIDEWAY"
                state.ensemble_result = state.ensemble_coordinator.evaluate_ensemble(
                    state.data_map, state.live_price, regime
                )
                state.candle_confluence = state.ensemble_coordinator.last_candle_confluence or state.multi_candle_engine.evaluate(state.data_map, state.live_price)

                df_struct = state.get_structure_df()
                df_macro = state.get_macro_df()
                state.order_research = state.ai_order_researcher.research(
                    current_price=state.live_price,
                    best_bid=state.fee_engine.bid_price,
                    best_ask=state.fee_engine.ask_price,
                    spread=state.fee_engine.spread,
                    indicators=state.indicators,
                    ai_verdict=state.ai_verdict,
                    ensemble_result=state.ensemble_result,
                    ai_cro=state.risk_manager.ai_cro,
                    current_balance=state.current_balance,
                    df_structure=df_struct,
                    df_macro=df_macro,
                    active_timeframe=state.active_timeframe,
                    candle_confluence=state.candle_confluence,
                    order_flow_verdict=state.order_flow_verdict,
                    effective_leverage=state.get_effective_leverage(),
                    current_position=state.current_position,
                    inventory_skew=state.hummingbot_skew.calculate_reservation_price(
                        state.live_price, state.current_position, state.indicators.get("atr", 200.0), state.current_balance
                    ),
                    visual_hft_metrics=state.visual_hft.get_metrics(),
                    jesse_metrics=state.jesse_engine.compute_metrics(),
                    octobot_consensus=state.octobot_consensus
                )

                state.order_manager.evaluate_and_clean_unsuitable_orders(
                    current_price=state.live_price,
                    indicators=state.indicators,
                    ai_verdict=state.ai_verdict,
                    ensemble_result=state.ensemble_result,
                    active_timeframe=state.active_timeframe
                )

                atr_val = state.indicators.get("atr") or (state.live_price * 0.008)
                atr_pct = (atr_val / state.live_price * 100.0) if state.live_price > 0 else 0.8
                conf = state.ai_verdict.confidence if state.ai_verdict else 80
                state.risk_manager.ai_cro.evaluate_risk_profile(
                    current_balance=state.current_balance,
                    market_regime=regime,
                    confidence=conf,
                    atr_pct=atr_pct
                )

                # -------------------------------------------------------------
                # OctoBot Tentacle Matrix & Trading Modes Evaluation
                # -------------------------------------------------------------
                smc_dict = {
                    "structure": state.ai_verdict.smc_structure if state.ai_verdict else "RANGING",
                    "demand_zone": list(state.ai_verdict.demand_zone) if (state.ai_verdict and state.ai_verdict.demand_zone) else None,
                    "supply_zone": list(state.ai_verdict.supply_zone) if (state.ai_verdict and state.ai_verdict.supply_zone) else None,
                    "liquidity_sweep": state.ai_verdict.last_sweep_info if state.ai_verdict else "NONE"
                } if state.ai_verdict else None

                vwap_status_str = getattr(state.ai_verdict, "vwap_status", "") if state.ai_verdict else ""
                vwap_dict = {
                    "vwap": state.ai_verdict.vwap_fair_price if state.ai_verdict else state.live_price,
                    "vwap_status": state.ai_verdict.vwap_status if state.ai_verdict else "EQUILIBRIUM_FAIR",
                    "dist_sigma": -1.8 if ("DISCOUNT" in vwap_status_str) else (1.8 if ("PREMIUM" in vwap_status_str) else 0.0)
                } if state.ai_verdict else None

                of_dict = {
                    "buy_ratio": state.order_flow_verdict.buy_ratio_pct if state.order_flow_verdict else 50.0,
                    "cvd_delta_60s": state.order_flow_verdict.current_cvd if state.order_flow_verdict else 0.0,
                    "absorption_signal": state.order_flow_verdict.absorption_divergence if state.order_flow_verdict else "NONE"
                } if state.order_flow_verdict else None

                mtf_dict = {
                    "score": (state.ensemble_result.consensus_score / 100.0) if state.ensemble_result else 0.0,
                    "consensus": state.ensemble_result.consensus_verdict if state.ensemble_result else "NEUTRAL",
                    "timeframes": state.ai_verdict.mtf_radar if (state.ai_verdict and state.ai_verdict.mtf_radar) else {}
                }

                state.octobot_consensus = state.octobot_matrix.evaluate_matrix(
                    current_price=state.live_price,
                    indicators=state.indicators,
                    order_flow_telemetry=of_dict,
                    smc_data=smc_dict,
                    vwap_data=vwap_dict,
                    mtf_consensus=mtf_dict
                )

                mode_name, trade_setup = state.octobot_coordinator.select_best_setup(
                    current_price=state.live_price,
                    matrix=state.octobot_consensus,
                    indicators=state.indicators,
                    smc_data=smc_dict,
                    of_data=of_dict,
                    vwap_data=vwap_dict
                )
                state.active_trading_mode = mode_name
                state.octobot_setup = trade_setup

                # Evaluate HKUDS Vibe Alpha Zoo (12 Quantitative Factors)
                if df_struct is not None and not df_struct.empty:
                    state.vibe_alpha_zoo.evaluate(
                        df=df_struct,
                        current_price=state.live_price,
                        best_bid=state.fee_engine.bid_price,
                        best_ask=state.fee_engine.ask_price,
                        spread=state.fee_engine.spread
                    )

                # Evaluate Shadow Account Behavioral Bias Scanner (Discipline Score)
                if state.trades:
                    state.shadow_account.analyze_trade_history(state.trades, state.initial_balance)

        except Exception as e:
            import traceback
            print(f"[ai_quant_background_loop Error]: {e}\n{traceback.format_exc()}", flush=True)
        await asyncio.sleep(1.0)


async def fallback_watchdog_loop():
    """
    Safety watchdog: If WebSocket stream experiences packet drop or silent disconnect,
    polls REST ticker every 3s to guarantee zero downtime.
    """
    while True:
        try:
            if state.ws_engine and state.ws_engine.total_ticks_received == 0:
                sym = state.symbol.upper()
                url = f"https://fapi.binance.com/fapi/v1/ticker/bookTicker?symbol={sym}"
                req = urllib.request.Request(url, headers={"User-Agent": "BinanceFuturesQuant/1.0"})
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    bid = float(data["bidPrice"])
                    ask = float(data["askPrice"])
                    mid = (bid + ask) / 2.0
                    state.fee_engine.update_book(bid, ask)
                    state.on_tick(mid)
                    state.update_indicators()
        except Exception:
            pass
        await asyncio.sleep(3.0)


@app.on_event("startup")
async def startup_event():
    state.initialize_history()
    state.init_ws_engine()
    await state.ws_engine.start()
    asyncio.create_task(state_broadcast_loop())
    asyncio.create_task(ai_quant_background_loop())
    asyncio.create_task(fallback_watchdog_loop())


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
async def get_klines(symbol: Optional[str] = None, interval: str = "15m", limit: int = 120):
    """
    Returns candlestick OHLCV data for TradingView chart across timeframes:
    1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d
    """
    sym = (symbol or state.symbol).upper().strip()
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={interval}&limit={limit}"
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
            if candles and state.live_price > 0 and (not symbol or sym == state.symbol):
                last_c = candles[-1]
                last_c["close"] = state.live_price
                last_c["high"] = max(last_c["high"], state.live_price)
                last_c["low"] = min(last_c["low"], state.live_price)
            return {"symbol": sym, "interval": interval, "candles": candles}
    except Exception as e:
        return {"error": str(e), "candles": []}


@app.post("/api/action/toggle")
async def toggle_bot():
    state.is_running = not state.is_running
    return {"status": "ok", "is_running": state.is_running}


@app.post("/api/action/close_all")
async def close_all():
    if state.current_position and state.live_price > 0:
        state.close_position(state.live_price, "THỦ CÔNG 🛑")
        await broadcast_state()
        return {"status": "closed"}
    return {"status": "no_position"}


@app.post("/api/action/set_symbol")
async def set_symbol(symbol: str = "BTCUSDT"):
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
async def reset_balance(amount: float = 1000.0):
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
    await broadcast_state()
    return {"status": "ok", "balance": state.current_balance}


@app.post("/api/action/manual_order")
async def manual_order(direction: int = 1):
    """
    Allows user to immediately trigger a live simulated LONG or SHORT trade
    with full ATR risk management and position sizing.
    """
    if state.current_position:
        return {"status": "already_in_position"}

    price = state.live_price
    setup = state.ai_order_researcher.structural_calculator.compute_setup(
        side="BUY" if direction == 1 else "SELL",
        entry_price=price,
        df_structure=state.get_structure_df(),
        timeframe=state.active_timeframe
    )
    sl = setup.stop_loss
    tp = setup.take_profit

    lev = state.get_effective_leverage()
    prop = state.risk_manager.evaluate_order(
        symbol=state.symbol,
        direction=direction,
        entry_price=price,
        stop_loss=sl,
        take_profit=tp,
        leverage=lev
    )

    if prop.approved:
        if hasattr(state, 'monthly_governor') and state.monthly_governor.enabled:
            gov = state.monthly_governor.evaluate(state.current_balance, state.trades)
            if gov.enabled and gov.size_multiplier != 1.0:
                prop.units = round(prop.units * gov.size_multiplier, 3)
                prop.notional_value = round(prop.units * price, 2)
                prop.required_margin = round(prop.notional_value / lev, 2)

        state.open_position(
            direction=direction,
            price=price,
            units=prop.units,
            notional=prop.notional_value,
            margin=prop.required_margin,
            sl=sl,
            tp=tp,
            liq=prop.est_liquidation_price,
            dt=datetime.now(),
            timeframe=state.active_timeframe
        )
        await broadcast_state()
        return {"status": "success", "order": state.current_position, "leverage": lev}

    return {"status": "rejected", "reason": prop.rejection_reason}


@app.post("/api/action/set_leverage")
async def set_leverage(mode: str = "AI_AUTO", val: int = 3):
    state.leverage_mode = mode
    if mode == "MANUAL":
        state.manual_leverage = max(1, min(val, 20))
    state.update_leverage_advice()
    await broadcast_state()
    return {"status": "ok", "mode": state.leverage_mode, "effective_leverage": state.get_effective_leverage()}


@app.post("/api/action/toggle_grid")
async def toggle_grid():
    state.is_grid_active = not state.is_grid_active
    if state.is_grid_active and state.live_price > 0:
        state.grid_bot.total_balance = state.current_balance
        state.grid_bot.generate_grid(state.live_price)
        print(f"⚡ [GRID BOT KÍCH HOẠT] Đã rải {len(state.grid_bot.grids)} tầng lệnh cân đối quanh ${state.live_price:,.2f}", flush=True)
    await broadcast_state()
    return {"status": "ok", "is_grid_active": state.is_grid_active, "grids": len(state.grid_bot.grids)}


@app.post("/api/action/toggle_auto_grid")
async def toggle_auto_grid():
    state.auto_grid_rotation = not state.auto_grid_rotation
    print(f"🔄 [AI AUTO-ROTATION] Chế độ xoay tua tự động Grid/Trend đã chuyển thành: {state.auto_grid_rotation}", flush=True)
    await broadcast_state()
    return {"status": "ok", "auto_grid_rotation": state.auto_grid_rotation}


@app.post("/api/action/ai_copilot_reason")
async def ai_copilot_reason(instruction: Optional[str] = None):
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
    await broadcast_state()
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
    models = state.ai_copilot.get_available_models()
    return {"status": "ok", "models": models, "active_model": state.ai_copilot.default_model}


@app.post("/api/action/set_ai_model")
async def set_ai_model(model: str = "ag/gemini-3.8-flash-high"):
    state.ai_copilot.set_model(model)
    state.storage.save_setting("ai_copilot_model", state.ai_copilot.default_model)
    print(f"🤖 [AI COPILOT] Đã chuyển sang mô hình AI: {state.ai_copilot.default_model} và lưu vĩnh viễn vào SQLite", flush=True)
    await broadcast_state()
    return {"status": "ok", "active_model": state.ai_copilot.default_model}



@app.post("/api/action/place_custom_order")
async def place_custom_order(
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

    if clean_type == "MARKET":
        if state.current_position:
            return {"status": "rejected", "reason": "Đang có vị thế mở. Hãy đóng vị thế trước hoặc dùng lệnh bồi DCA!"}
        direction = 1 if side.upper() == "BUY" else -1
        exec_price = state.fee_engine.get_execution_price("MARKET", side)
        if exec_price <= 0:
            exec_price = current_p

        setup = state.ai_order_researcher.structural_calculator.compute_setup(
            side=side.upper(),
            entry_price=exec_price,
            df_structure=state.get_structure_df(),
            timeframe=state.active_timeframe
        )
        sl = setup.stop_loss
        tp = setup.take_profit
        prop = state.risk_manager.evaluate_order(
            symbol=sym, direction=direction, entry_price=exec_price,
            stop_loss=sl, take_profit=tp, leverage=lev
        )
        if prop.approved:
            if hasattr(state, 'monthly_governor') and state.monthly_governor.enabled:
                gov = state.monthly_governor.evaluate(state.current_balance, state.trades)
                if gov.enabled and gov.size_multiplier != 1.0:
                    prop.units = round(prop.units * gov.size_multiplier, 3)
                    prop.notional_value = round(prop.units * exec_price, 2)
                    prop.required_margin = round(prop.notional_value / lev, 2)

            state.open_position(
                direction=direction, price=exec_price, units=prop.units,
                notional=prop.notional_value, margin=prop.required_margin,
                sl=sl, tp=tp, liq=prop.est_liquidation_price, dt=datetime.now(),
                is_maker=False, timeframe=state.active_timeframe
            )
            await broadcast_state()
            return {
                "status": "filled",
                "order_type": "MARKET",
                "price": exec_price,
                "side": side.upper(),
                "timeframe": state.active_timeframe,
                "fee_rate": "TAKER (0.05%)",
                "breakeven_target": state.current_position["breakeven_price"] if state.current_position else exec_price
            }
        return {"status": "rejected", "reason": prop.rejection_reason}

    else:
        # All queued / advanced order types: LIMIT, POST_ONLY, CONDITIONAL, TRAILING_STOP, TWAP, SCALE_RATIO
        order, msg = state.order_manager.place_order(
            symbol=sym,
            order_type=clean_type,
            side=side.upper(),
            price=target_price,
            margin=margin,
            leverage=lev,
            trigger_price=trigger_price,
            trigger_condition=trigger_condition,
            callback_pct=callback_pct,
            twap_slices=twap_slices,
            twap_interval_ticks=twap_interval_ticks,
            best_bid=state.fee_engine.bid_price,
            best_ask=state.fee_engine.ask_price,
            timeframe=state.active_timeframe
        )
        if order is None:
            return {"status": "rejected", "reason": msg}

        await broadcast_state()
        return {
            "status": "placed",
            "order_type": clean_type,
            "order_id": order.order_id,
            "side": order.side,
            "price": order.price,
            "margin": margin,
            "timeframe": state.active_timeframe,
            "message": msg
        }


@app.post("/api/action/cancel_order")
async def cancel_order(order_id: int):
    success = state.order_manager.cancel_order(order_id)
    await broadcast_state()
    return {"status": "ok" if success else "not_found", "order_id": order_id}


@app.post("/api/action/update_pending_order")
async def update_pending_order(
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
    success, msg = state.order_manager.update_order(
        order_id=order_id,
        price=price,
        units=units,
        margin=margin,
        timeframe=timeframe,
        order_type=order_type,
        side=side,
        trigger_price=trigger_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        leverage=leverage,
        callback_pct=callback_pct
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    state.persist_current_state()
    await broadcast_state()
    return {"status": "ok", "message": msg, "order_id": order_id}



@app.post("/api/action/execute_pending_order")
async def execute_pending_order(order_id: int):
    target_order = None
    for o in state.order_manager.orders:
        if o.order_id == order_id and o.status in ("PENDING", "ACTIVE"):
            target_order = o
            break

    if not target_order:
        return {"status": "error", "message": f"Không tìm thấy lệnh chờ #{order_id} hoặc lệnh đã kết thúc/đã bị hủy!"}

    # If already in an opposite position, prevent accidental conflicting trade
    if state.current_position and state.current_position["direction"] != target_order.direction:
        pos_dir = "LONG 🟢" if state.current_position["direction"] == 1 else "SHORT 🔴"
        order_dir = "LONG 🟢" if target_order.direction == 1 else "SHORT 🔴"
        return {
            "status": "error",
            "message": f"Đang có vị thế {pos_dir} mở! Không thể vào lệnh ngược chiều {order_dir}. Hãy đóng vị thế hiện tại trước khi vào lệnh này!"
        }

    current_p = state.live_price
    direction = target_order.direction
    side = target_order.side.upper()
    exec_price = state.fee_engine.get_execution_price("MARKET", side)
    if exec_price <= 0:
        exec_price = current_p

    # Recalculate units if needed
    units = target_order.units
    if units <= 0 and exec_price > 0:
        units = round((target_order.margin * target_order.leverage) / exec_price, 4)
        target_order.units = units

    # Send Live order to exchange if live trading is active
    if state.active_exchange_api.is_live_enabled:
        try:
            live_res = state.active_exchange_api.create_order(
                symbol=state.symbol,
                side=side,
                order_type="MARKET",
                quantity=units
            )
            if not live_res.get("success"):
                print(f"⚠️ [LIVE EXECUTE FAILED] {live_res.get('message')}", flush=True)
        except Exception as e:
            print(f"⚠️ [LIVE EXECUTE EXCEPTION] {e}", flush=True)

    # Mark order in queue as FILLED
    state.order_manager.force_execute_order(order_id, exec_price)

    fo_tf = getattr(target_order, "timeframe", state.active_timeframe) or state.active_timeframe

    if not state.current_position:
        struct_calc = state.ai_order_researcher.structural_calculator
        if target_order.stop_loss and ((direction == 1 and 0 < target_order.stop_loss < exec_price) or (direction == -1 and target_order.stop_loss > exec_price)):
            struct_sl = target_order.stop_loss
            struct_tp = target_order.take_profit
        else:
            setup = struct_calc.compute_setup(
                side=side,
                entry_price=exec_price,
                df_structure=state.get_structure_df(),
                timeframe=fo_tf
            )
            struct_sl = setup.stop_loss
            struct_tp = setup.take_profit

        notional = target_order.margin * target_order.leverage
        liq_buffer = (0.98 / target_order.leverage) * exec_price
        liq = (exec_price - liq_buffer) if direction == 1 else (exec_price + liq_buffer)

        state.open_position(
            direction=direction,
            price=exec_price,
            units=units,
            notional=notional,
            margin=target_order.margin,
            sl=struct_sl,
            tp=struct_tp,
            liq=liq,
            dt=datetime.now(),
            is_maker=False,
            timeframe=fo_tf,
            order_id=order_id,
            order_type=getattr(target_order, "order_type", "LIMIT")
        )
        msg = f"Đã vào lệnh {side} #{order_id} thành công tại ${exec_price:,.1f} (Ký quỹ: ${target_order.margin:,.1f}, {target_order.leverage}x)!"
    else:
        # Scale into position (DCA)
        pos = state.current_position
        add_margin = target_order.margin
        add_notional = target_order.margin * target_order.leverage
        add_units = units
        total_units = pos["units"] + add_units
        total_notional = pos["notional"] + add_notional
        new_avg_entry = total_notional / total_units if total_units > 0 else pos["entry_price"]

        pos["entry_price"] = round(new_avg_entry, 2)
        pos["units"] = round(total_units, 4)
        pos["margin"] += add_margin
        pos["notional"] = total_notional
        pos["breakeven_price"] = state.fee_engine.calculate_breakeven_price(new_avg_entry, direction)

        entry_fee = state.fee_engine.calculate_fee(add_notional, is_maker=False)
        state.current_balance -= entry_fee
        state.total_fees += entry_fee
        pos["entry_fee"] += entry_fee

        order_slice = {
            "slice_id": f"ORD-{order_id}",
            "order_id": order_id,
            "order_type": getattr(target_order, "order_type", "LIMIT DCA"),
            "timeframe": fo_tf,
            "side": side,
            "direction": direction,
            "entry_price": round(exec_price, 2),
            "units": round(add_units, 4),
            "margin": round(add_margin, 2),
            "notional": round(add_notional, 2),
            "entry_fee": round(entry_fee, 4),
            "fee_tier": "TAKER (0.05%)",
            "entry_time": datetime.now().strftime("%m-%d %H:%M:%S"),
            "unrealized_pnl": 0.0,
            "roe_pct": 0.0,
        }
        if "orders" not in pos or not isinstance(pos["orders"], list):
            pos["orders"] = []
        pos["orders"].append(order_slice)

        msg = f"Đã bồi thêm vị thế {side} #{order_id} thành công! Giá vào TB mới: ${new_avg_entry:,.1f}"

    print(f"⚡ [THỦ CÔNG VÀO LỆNH CHỜ] #{order_id} {side} {target_order.symbol} [{fo_tf}] tại ${exec_price:,.2f}", flush=True)
    await broadcast_state()
    return {"status": "ok", "message": msg, "order_id": order_id}


class UpdateTpSlRequest(BaseModel):
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    lock_manual: bool = True


@app.post("/api/action/update_position_tp_sl")
async def update_position_tp_sl(req: UpdateTpSlRequest):
    if not state.current_position:
        raise HTTPException(status_code=400, detail="Không có vị thế nào đang mở!")

    success, msg = state.set_manual_tp_sl(
        sl=req.stop_loss,
        tp=req.take_profit,
        lock_manual=req.lock_manual
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    await broadcast_state()
    return {
        "status": "ok",
        "message": msg,
        "stop_loss": state.current_position.get("stop_loss"),
        "take_profit": state.current_position.get("take_profit"),
        "is_manual_tpsl": state.current_position.get("is_manual_tpsl")
    }


@app.post("/api/action/close_order_slice")
async def close_order_slice(slice_id: str):
    if not state.current_position:
        raise HTTPException(status_code=400, detail="Không có vị thế nào đang mở!")

    success, msg = state.close_order_slice(slice_id=slice_id)
    if not success:
        raise HTTPException(status_code=400, detail=msg)

    await broadcast_state()
    return {"status": "ok", "message": msg, "slice_id": slice_id}


@app.post("/api/action/update_order_slice")
async def update_order_slice(
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

    await broadcast_state()
    return {"status": "ok", "message": msg, "slice_id": slice_id, "position": state.current_position}


@app.post("/api/action/set_timeframe")
async def set_timeframe(timeframe: str = "15m"):
    raw = timeframe.strip()
    if raw.lower() in ("1w", "w"):
        tf = "1w"
    elif raw.lower() in ("1m_month", "m", "1mth") or raw == "1M":
        tf = "1M"
    else:
        tf = raw.lower()

    if tf in ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w", "1M"):
        state.active_timeframe = tf
        if tf not in state.data_map or state.data_map[tf] is None or state.data_map[tf].empty:
            limit = 100 if tf in ("1w", "1M") else 300
            state.data_map[tf] = state.fetcher.fetch_klines(state.symbol, tf, limit)
        state.ai_verdict = state.ai_brain.analyze(state.data_map, state.live_price, active_timeframe=state.active_timeframe)
        df_struct = state.get_structure_df()
        df_macro = state.data_map.get("1h")
        state.order_research = state.ai_order_researcher.research(
            current_price=state.live_price,
            best_bid=state.fee_engine.bid_price or state.live_price - 0.5,
            best_ask=state.fee_engine.ask_price or state.live_price + 0.5,
            spread=state.fee_engine.spread,
            indicators=state.indicators,
            ai_verdict=state.ai_verdict,
            ensemble_result=state.ensemble_result,
            ai_cro=state.risk_manager.ai_cro,
            current_balance=state.current_balance,
            df_structure=df_struct,
            df_macro=df_macro,
            active_timeframe=state.active_timeframe,
            candle_confluence=state.candle_confluence,
            order_flow_verdict=state.order_flow_verdict,
            effective_leverage=state.get_effective_leverage(),
            current_position=state.current_position,
            inventory_skew=state.hummingbot_skew.calculate_reservation_price(
                state.live_price, state.current_position, state.indicators.get("atr", 200.0), state.current_balance
            )
        )
        await broadcast_state()
        return {"status": "ok", "active_timeframe": state.active_timeframe}
    return {"status": "error", "message": "Khung thời gian không hợp lệ"}


@app.post("/api/action/place_dual_bracket")
async def place_dual_bracket():
    res = state.order_research
    if not res:
        return {"status": "error", "message": "Nghiên cứu AI chưa sẵn sàng"}
    sub_margin = round(res.optimal_margin * 0.5, 2)
    atr = state.indicators.get("atr") or (state.live_price * 0.008)
    o1, m1 = state.order_manager.place_order(
        symbol=state.symbol,
        order_type="POST_ONLY",
        side="BUY",
        price=res.dual_buy_price,
        margin=sub_margin,
        leverage=res.optimal_leverage,
        stop_loss=round(res.dual_buy_price - 1.2 * atr, 2),
        take_profit=round(res.dual_sell_price, 2),
        timeframe=state.active_timeframe,
        note=f"Biên Dưới [{state.active_timeframe}] (Long Limit 2 Đầu)"
    )
    o2, m2 = state.order_manager.place_order(
        symbol=state.symbol,
        order_type="POST_ONLY",
        side="SELL",
        price=res.dual_sell_price,
        margin=sub_margin,
        leverage=res.optimal_leverage,
        stop_loss=round(res.dual_sell_price + 1.2 * atr, 2),
        take_profit=round(res.dual_buy_price, 2),
        timeframe=state.active_timeframe,
        note=f"Biên Trên [{state.active_timeframe}] (Short Limit 2 Đầu)"
    )
    await broadcast_state()
    return {
        "status": "ok",
        "message": f"Đã rải 2 đầu: Mua ${res.dual_buy_price} & Bán ${res.dual_sell_price}",
        "buy_order": o1.order_id if o1 else None,
        "sell_order": o2.order_id if o2 else None
    }


@app.post("/api/action/partial_close")
async def partial_close(ratio: float = 0.5):
    if not state.current_position:
        return {"status": "error", "message": "Không có vị thế nào đang mở"}
    rec = state.close_partial_position(ratio=ratio, reason=f"THỦ CÔNG: CHỐT {int(ratio*100)}% VỊ THẾ 💰", is_maker=False)
    await broadcast_state()
    return {"status": "ok", "message": f"Đã chốt {int(ratio*100)}% vị thế và kéo SL về hòa vốn!", "record": rec}


@app.post("/api/action/lock_breakeven")
async def lock_breakeven():
    if not state.current_position:
        return {"status": "error", "message": "Không có vị thế nào đang mở"}
    ok = state.lock_breakeven_now()
    await broadcast_state()
    return {"status": "ok" if ok else "error", "message": "Đã dời Stop Loss về điểm hòa vốn (Entry + Phí Sàn)!"}


@app.post("/api/action/set_risk_pct")
async def set_risk_pct(risk_pct: float = 1.5):
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
    await broadcast_state()
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
async def save_api_keys(payload: dict):
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

        await broadcast_state()
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

        await broadcast_state()
        return {"status": "ok", "message": "Đã lưu thông tin API Binance an toàn vào cơ sở dữ liệu SQLite!"}


@app.post("/api/settings/test_connection")
async def test_binance_connection(payload: Optional[dict] = None):
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
async def toggle_trading_mode(payload: dict):
    live_enabled = bool(payload.get("live_enabled", False))
    if live_enabled:
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
    state.mexc_api.is_live_enabled = live_enabled
    state.storage.save_setting("is_live_enabled", live_enabled)
    await broadcast_state()
    mode_text = "GIAO DỊCH THẬT (LIVE TRADING) 🚀" if live_enabled else "MÔ PHỎNG (PAPER TRADING) 🧪"
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
async def reset_data(amount: float = 1000.0):
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
    await broadcast_state()
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
async def import_data(payload: dict):
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
    await broadcast_state()
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
async def save_vibe_config(payload: dict):
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
    await broadcast_state()
    return {
        "status": "ok",
        "message": f"Đã lưu cấu hình Vibe AI Swarm Council (Bật: {enabled}, Ngưỡng: {min_votes}/4 phiếu)",
        "config": cfg
    }


@app.post("/api/action/run_vibe_swarm_debate")
async def run_vibe_swarm_debate():
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
    await broadcast_state()
    return {"status": "ok", "verdict": asdict(verdict) if verdict else None}


if __name__ == "__main__":
    uvicorn.run("ui.server:app", host="127.0.0.1", port=8000, reload=False)
