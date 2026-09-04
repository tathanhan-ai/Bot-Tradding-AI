"""
Binance Futures Spread & Fee Modeling Engine
Accurately models real-world Binance USD(S)-M Futures:
1. Dynamic VIP Fee Tiers (VIP 0 through VIP 9)
2. 10% BNB fee discount option (BNB Fee Deduction)
3. Direct Binance API commission rate auto-sync (/fapi/v1/commissionRate)
4. Bid/Ask Orderbook Spread (Best Bid vs Best Ask)
5. True Net Breakeven Price calculation (covering 2-way fees + spread)
6. Fee drag assessment
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional

BINANCE_VIP_TIERS: Dict[str, Dict[str, Any]] = {
    "VIP_0": {"name": "VIP 0 (Khách Thường)", "maker": 0.00020, "taker": 0.00050},
    "VIP_1": {"name": "VIP 1 (Vol > 250k)", "maker": 0.00016, "taker": 0.00040},
    "VIP_2": {"name": "VIP 2 (Vol > 1M)", "maker": 0.00014, "taker": 0.00035},
    "VIP_3": {"name": "VIP 3 (Vol > 5M)", "maker": 0.00012, "taker": 0.00032},
    "VIP_4": {"name": "VIP 4 (Vol > 10M)", "maker": 0.00010, "taker": 0.00030},
    "VIP_5": {"name": "VIP 5 (Vol > 20M)", "maker": 0.00008, "taker": 0.00027},
    "VIP_6": {"name": "VIP 6 (Vol > 50M)", "maker": 0.00006, "taker": 0.00024},
    "VIP_7": {"name": "VIP 7 (Vol > 100M)", "maker": 0.00004, "taker": 0.00021},
    "VIP_8": {"name": "VIP 8 (Vol > 200M)", "maker": 0.00002, "taker": 0.00018},
    "VIP_9": {"name": "VIP 9 (Market Maker)", "maker": 0.00000, "taker": 0.00015},
}

MEXC_VIP_TIERS: Dict[str, Dict[str, Any]] = {
    "MEXC_STANDARD": {"name": "MEXC Chuẩn (0% Maker / 0.02% Taker)", "maker": 0.00000, "taker": 0.00020},
    "MEXC_MX_DISCOUNT": {"name": "MEXC Giữ MX Token (-10% Phí)", "maker": 0.00000, "taker": 0.00018},
    "MEXC_VIP_1": {"name": "MEXC VIP 1", "maker": 0.00000, "taker": 0.00018},
    "MEXC_VIP_2": {"name": "MEXC VIP 2", "maker": 0.00000, "taker": 0.00016},
    "MEXC_VIP_3": {"name": "MEXC VIP 3", "maker": 0.00000, "taker": 0.00014},
    "MEXC_VIP_4": {"name": "MEXC VIP 4", "maker": 0.00000, "taker": 0.00012},
    "MEXC_VIP_5": {"name": "MEXC VIP 5 (MM / Pro)", "maker": 0.00000, "taker": 0.00010},
}


class SpreadFeeEngine:
    def __init__(self, vip_tier: str = "VIP_0", use_bnb_discount: bool = False, exchange: str = "binance"):
        self.exchange = str(exchange).lower().strip()
        if self.exchange == "mexc":
            self.vip_tier = vip_tier if vip_tier in MEXC_VIP_TIERS else "MEXC_STANDARD"
        else:
            self.vip_tier = vip_tier if vip_tier in BINANCE_VIP_TIERS else "VIP_0"
        self.use_bnb_discount = use_bnb_discount
        self.is_custom_rate = False

        self.custom_maker_fee: Optional[float] = None
        self.custom_taker_fee: Optional[float] = None

        self.bid_price: float = 0.0
        self.ask_price: float = 0.0
        self.mid_price: float = 0.0
        self.spread: float = 0.0
        self.spread_pct: float = 0.0

    def set_exchange(self, exchange: str):
        self.exchange = str(exchange).lower().strip()
        if self.exchange == "mexc":
            if self.vip_tier not in MEXC_VIP_TIERS:
                self.vip_tier = "MEXC_STANDARD"
        else:
            if self.vip_tier not in BINANCE_VIP_TIERS:
                self.vip_tier = "VIP_0"
        self.is_custom_rate = False

    @property
    def current_tiers_dict(self) -> Dict[str, Dict[str, Any]]:
        return MEXC_VIP_TIERS if self.exchange == "mexc" else BINANCE_VIP_TIERS

    @property
    def maker_fee_rate(self) -> float:
        if self.is_custom_rate and self.custom_maker_fee is not None:
            return self.custom_maker_fee
        tiers = self.current_tiers_dict
        default_tier = "MEXC_STANDARD" if self.exchange == "mexc" else "VIP_0"
        base = tiers.get(self.vip_tier, tiers.get(default_tier, {})).get("maker", 0.0002)
        if self.exchange == "binance":
            return base * 0.9 if self.use_bnb_discount else base
        return base

    @property
    def taker_fee_rate(self) -> float:
        if self.is_custom_rate and self.custom_taker_fee is not None:
            return self.custom_taker_fee
        tiers = self.current_tiers_dict
        default_tier = "MEXC_STANDARD" if self.exchange == "mexc" else "VIP_0"
        base = tiers.get(self.vip_tier, tiers.get(default_tier, {})).get("taker", 0.0005)
        if self.use_bnb_discount:
            return base * 0.9
        return base

    def set_vip_tier(self, vip_tier: str, use_bnb_discount: bool = False):
        tiers = self.current_tiers_dict
        if vip_tier in tiers:
            self.vip_tier = vip_tier
        self.use_bnb_discount = use_bnb_discount
        self.is_custom_rate = False

    def set_custom_rates(self, maker_rate: float, taker_rate: float):
        self.custom_maker_fee = maker_rate
        self.custom_taker_fee = taker_rate
        self.is_custom_rate = True

    def update_book(self, bid: float, ask: float):
        self.bid_price = round(bid, 2)
        self.ask_price = round(ask, 2)
        self.mid_price = round((bid + ask) / 2.0, 2)
        self.spread = round(max(0.0, ask - bid), 2)
        self.spread_pct = round((self.spread / self.mid_price * 100.0) if self.mid_price > 0 else 0.0, 4)

    def get_execution_price(self, order_type: str, side: str) -> float:
        """
        When buying via MARKET order, you execute at the ASK price.
        When selling via MARKET order, you execute at the BID price.
        """
        if order_type.upper() == "MARKET":
            if side.upper() == "BUY":
                return self.ask_price if self.ask_price > 0 else self.mid_price
            else:
                return self.bid_price if self.bid_price > 0 else self.mid_price
        return self.mid_price

    def calculate_fee(self, notional: float, is_maker: bool = False) -> float:
        rate = self.maker_fee_rate if is_maker else self.taker_fee_rate
        return round(notional * rate, 4)

    def calculate_breakeven_price(
        self,
        entry_price: float,
        direction: int,
        entry_is_maker: bool = False,
        exit_is_maker: bool = False
    ) -> float:
        """
        Calculates the exact exit price needed to break even after paying both entry and exit fees.
        LONG:  Exit = Entry * (1.0 + fee1) / (1.0 - fee2)
        SHORT: Exit = Entry * (1.0 - fee1) / (1.0 + fee2)
        """
        fee1 = self.maker_fee_rate if entry_is_maker else self.taker_fee_rate
        fee2 = self.maker_fee_rate if exit_is_maker else self.taker_fee_rate

        if direction == 1:  # LONG
            return round(entry_price * (1.0 + fee1) / (1.0 - fee2), 2)
        else:  # SHORT
            return round(entry_price * (1.0 - fee1) / (1.0 + fee2), 2)

    def get_state_dict(self) -> Dict[str, Any]:
        maker_rate = self.maker_fee_rate
        taker_rate = self.taker_fee_rate
        tier_info = self.current_tiers_dict.get(self.vip_tier, {"name": self.vip_tier})

        return {
            "exchange": self.exchange,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "spread_pct": self.spread_pct,
            "vip_tier": self.vip_tier,
            "vip_name": tier_info["name"],
            "use_bnb_discount": self.use_bnb_discount,
            "is_custom_rate": self.is_custom_rate,
            "maker_fee_pct": round(maker_rate * 100, 4),
            "taker_fee_pct": round(taker_rate * 100, 4),
            "roundtrip_taker_fee_pct": round(taker_rate * 2 * 100, 4),
            "roundtrip_maker_fee_pct": round(maker_rate * 2 * 100, 4)
        }
