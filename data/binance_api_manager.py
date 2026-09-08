"""
Binance Futures API Manager
Handles live authentication, real balance checking, fee rate querying,
and order execution on Binance USD(S)-M Futures (Mainnet & Testnet).

Features:
- HMAC-SHA256 signature authentication
- Mainnet & Testnet endpoint toggling
- Account balance and margin verification (/fapi/v2/account)
- Commission rate discovery (/fapi/v1/commissionRate)
- Real order placement with safety guardrails (/fapi/v1/order)
- Safe credential masking (protects private keys)
"""
import hashlib
import hmac
import json
import math
import re
import time
import urllib.parse
import urllib.request
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, Optional, Tuple


class BinanceAPIManager:
    MAINNET_BASE_URL = "https://fapi.binance.com"
    TESTNET_BASE_URL = "https://testnet.binancefuture.com"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        is_testnet: bool = False,
        is_live_enabled: bool = False
    ):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.is_testnet = is_testnet
        self.is_live_enabled = is_live_enabled
        self.server_time_offset_ms = 0
        self._clock_synced_at = 0.0
        self._symbol_filters: Dict[str, Any] = {}

    @property
    def base_url(self) -> str:
        return self.TESTNET_BASE_URL if self.is_testnet else self.MAINNET_BASE_URL

    def set_credentials(self, api_key: str, api_secret: str, is_testnet: bool):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.is_testnet = is_testnet
        self._clock_synced_at = 0.0
        self._symbol_filters.clear()

    def get_masked_credentials(self) -> Dict[str, Any]:
        masked_key = ""
        if self.api_key:
            if len(self.api_key) > 8:
                masked_key = self.api_key[:4] + "*" * (len(self.api_key) - 8) + self.api_key[-4:]
            else:
                masked_key = "****"

        has_secret = bool(self.api_secret)
        masked_secret = "****************" if has_secret else ""

        return {
            "api_key_masked": masked_key,
            "has_secret": has_secret,
            "api_secret_masked": masked_secret,
            "is_testnet": self.is_testnet,
            "is_live_enabled": self.is_live_enabled,
            "network_url": self.base_url
        }

    def _sign_params(self, params: Dict[str, Any]) -> str:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000 + self.server_time_offset_ms)
        params.setdefault("recvWindow", 5000)
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return f"{query_string}&signature={signature}"

    def _send_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, signed: bool = True) -> Tuple[bool, Any]:
        method = method.upper()
        if method != "GET":
            error = self._write_error()
            if error:
                return False, error
        if params is None:
            params = {}

        headers = {
            "User-Agent": "Antigravity-Binance-Bot/2.0",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key

        url = f"{self.base_url}{endpoint}"
        body = None

        if signed:
            if not self.api_key or not self.api_secret:
                return False, {"code": -1, "msg": "Chưa cấu hình API Key hoặc API Secret."}
            if time.monotonic() - self._clock_synced_at > 60 or not self._clock_synced_at:
                clock_ok, clock_data = self.synchronize_time()
                if not clock_ok:
                    return False, clock_data
            query_with_sig = self._sign_params(params)
            if method in ("GET", "DELETE"):
                url = f"{url}?{query_with_sig}"
            else:
                body = query_with_sig.encode("utf-8")
        else:
            if params:
                query_str = urllib.parse.urlencode(params)
                if method == "GET":
                    url = f"{url}?{query_str}"
                else:
                    body = query_str.encode("utf-8")

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return True, data
        except urllib.error.HTTPError as err:
            err_text = err.read().decode("utf-8")
            try:
                err_json = json.loads(err_text)
                if err_json.get("code") == -1021:
                    self._clock_synced_at = 0.0
                return False, err_json
            except Exception:
                return False, {"code": err.code, "msg": err_text or str(err)}
        except Exception as e:
            return False, {"code": -999, "msg": f"Lỗi kết nối mạng: {str(e)}"}

    def _write_error(self) -> Optional[Dict[str, Any]]:
        if not self.is_testnet:
            return {"code": -997, "msg": "Mainnet không nằm trong scope thực thi. Chỉ Binance Testnet được phép."}
        if not self.is_live_enabled:
            return {"code": -998, "msg": "Giao dịch Binance Testnet chưa được bật."}
        if not self.api_key or not self.api_secret:
            return {"code": -1, "msg": "Thiếu Binance API Key hoặc Secret Key."}
        return None

    @staticmethod
    def _positive(value: Any) -> bool:
        try:
            return not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0
        except (TypeError, ValueError, OverflowError):
            return False

    def synchronize_time(self) -> Tuple[bool, Any]:
        start = time.time() * 1000
        ok, data = self._send_request("GET", "/fapi/v1/time", signed=False)
        end = time.time() * 1000
        if not ok:
            return False, data
        if not isinstance(data, dict) or not self._positive(data.get("serverTime")):
            return False, {"code": -2, "msg": "Binance serverTime không hợp lệ."}
        self.server_time_offset_ms = float(data["serverTime"]) - (start + end) / 2
        self._clock_synced_at = time.monotonic()
        return True, {"offset_ms": self.server_time_offset_ms}

    def prepare_testnet_trading(self, symbol: str, leverage: int, hedge: bool = False) -> Tuple[bool, Any]:
        """Confirm the requested Binance position mode and final leverage before entry."""
        error = self._write_error()
        if error:
            return False, error
        if not self._positive(leverage) or int(float(leverage)) != float(leverage) or not 1 <= int(float(leverage)) <= 125:
            return False, {"code": -2, "msg": "leverage phải là số nguyên trong khoảng 1-125."}
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9_]{1,32}", symbol.upper()):
            return False, {"code": -2, "msg": "symbol không hợp lệ."}
        ok, mode = self._send_request("GET", "/fapi/v1/positionSide/dual")
        if not ok:
            return False, mode
        if not isinstance(mode, dict) or mode.get("dualSidePosition") not in (True, False, "true", "false"):
            return False, {"code": -2, "msg": "Không xác nhận được Binance position mode."}
        current_hedge = mode["dualSidePosition"] in (True, "true")
        if current_hedge != hedge:
            ok, changed = self._send_request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true" if hedge else "false"})
            if not ok:
                return False, changed
        ok, margin = self._send_request("POST", "/fapi/v1/marginType", {"symbol": symbol.upper(), "marginType": "ISOLATED"})
        if not ok and margin.get("code") != -4046:  # Already isolated.
            return False, margin
        return self._send_request("POST", "/fapi/v1/leverage", {"symbol": symbol.upper(), "leverage": int(float(leverage))})

    def normalize_order_values(self, symbol: str, quantity: float, price: Optional[float] = None, *,
                               order_type: str = "LIMIT", reference_price: Optional[float] = None,
                               reduce_only: bool = False, price_rounding: str = "down") -> Tuple[bool, Any]:
        """Floor quantity without exceeding the risk envelope; never round up to minimum notional."""
        if not self._positive(quantity) or (price is not None and not self._positive(price)):
            return False, {"code": -2, "msg": "quantity/price phải hữu hạn và lớn hơn 0."}
        if price_rounding not in ("down", "up") or not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9_]{1,32}", symbol.upper()):
            return False, {"code": -2, "msg": "symbol/price_rounding không hợp lệ."}
        symbol = symbol.upper()
        cache_key = f"{self.base_url}:{symbol}"
        cached = self._symbol_filters.get(cache_key)
        if not cached or time.monotonic() - cached[0] > 600:
            ok, info = self._send_request("GET", "/fapi/v1/exchangeInfo", signed=False)
            if not ok:
                return False, info
            if not isinstance(info, dict):
                return False, {"code": -2, "msg": "Binance exchangeInfo không hợp lệ."}
            entry = next((item for item in info.get("symbols", []) if item.get("symbol") == symbol), None)
            if not entry or entry.get("status") != "TRADING":
                return False, {"code": -2, "msg": f"{symbol} không ở trạng thái TRADING."}
            cached = (time.monotonic(), {item["filterType"]: item for item in entry.get("filters", [])})
            self._symbol_filters[cache_key] = cached
        filters = cached[1]
        try:
            lot = filters["MARKET_LOT_SIZE" if order_type in ("MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET", "TRAILING_STOP") else "LOT_SIZE"]
            step, minimum, maximum = (Decimal(lot[key]) for key in ("stepSize", "minQty", "maxQty"))
            if not all(value.is_finite() and value > 0 for value in (step, minimum, maximum)):
                raise ValueError("invalid lot filter")
            normalized_qty = (min(Decimal(str(quantity)), maximum) / step).to_integral_value(rounding=ROUND_DOWN) * step
            if normalized_qty < minimum:
                raise ValueError("quantity below minQty after floor")
            normalized_price = None
            if price is not None:
                pf = filters["PRICE_FILTER"]
                tick = Decimal(pf["tickSize"])
                if not tick.is_finite() or tick <= 0:
                    raise ValueError("invalid tickSize")
                normalized_price = (Decimal(str(price)) / tick).to_integral_value(rounding=ROUND_UP if price_rounding == "up" else ROUND_DOWN) * tick
                if normalized_price <= 0 or normalized_price < Decimal(pf["minPrice"]) or normalized_price > Decimal(pf["maxPrice"]):
                    raise ValueError("price outside PRICE_FILTER")
            notional_price = normalized_price if normalized_price is not None else reference_price
            notional_filter = filters.get("MIN_NOTIONAL", filters.get("NOTIONAL", {}))
            min_notional = Decimal(str(notional_filter.get("notional", notional_filter.get("minNotional", 0))))
            if not reduce_only and min_notional > 0:
                if not self._positive(notional_price) or normalized_qty * Decimal(str(notional_price)) < min_notional:
                    raise ValueError("notional below minimum or reference_price missing")
            return True, {"quantity": float(normalized_qty), "price": float(normalized_price) if normalized_price is not None else None,
                          "step_size": float(step), "min_quantity": float(minimum), "min_notional": float(min_notional)}
        except (KeyError, TypeError, ValueError, InvalidOperation, ZeroDivisionError) as exc:
            return False, {"code": -2, "msg": f"Binance filters: {exc}"}

    def get_wallet_snapshot(self) -> Dict[str, Any]:
        """Đọc số dư ví Futures THẬT từ sàn (wallet + khả dụng + quyền trade).

        Dùng để đồng bộ vốn live: sizing phải nghe số tiền server, không dùng
        vốn ảo nội bộ phình/xẹp theo ledger cũ. Trả về dict success/wallet_balance/
        available_balance/can_trade; thất bại thì success=False và caller giữ vốn cũ.
        """
        if not self.api_key or not self.api_secret:
            return {"success": False, "message": "Chưa cấu hình API Key/Secret.",
                    "wallet_balance": 0.0, "available_balance": 0.0, "can_trade": False}
        ok, acc = self._send_request("GET", "/fapi/v2/account", signed=True)
        if not ok or not isinstance(acc, dict):
            msg = acc.get("msg", "Lỗi đọc tài khoản") if isinstance(acc, dict) else "Lỗi đọc tài khoản"
            return {"success": False, "message": msg,
                    "wallet_balance": 0.0, "available_balance": 0.0, "can_trade": False}
        try:
            wallet = float(acc.get("totalWalletBalance", 0.0) or 0.0)
            avail = float(acc.get("availableBalance", 0.0) or 0.0)
        except (TypeError, ValueError):
            return {"success": False, "message": "Số dư sàn trả về không hợp lệ.",
                    "wallet_balance": 0.0, "available_balance": 0.0, "can_trade": False}
        if not (math.isfinite(wallet) and math.isfinite(avail)) or wallet < 0 or avail < 0:
            return {"success": False, "message": "Số dư sàn không hữu hạn.",
                    "wallet_balance": 0.0, "available_balance": 0.0, "can_trade": False}
        return {"success": True, "wallet_balance": wallet, "available_balance": avail,
                "can_trade": bool(acc.get("canTrade", False))}

    def get_position_snapshot(self, symbol: str) -> Dict[str, Any]:
        """Đọc vị thế THẬT của symbol từ sàn (positionRisk) để đối chiếu ledger.

        Trả về success + danh sách vị thế có khối lượng khác 0:
        [{position_side, amount, entry_price, unrealized_pnl, leverage}].
        """
        sym = (symbol or "").upper().strip()
        if not sym:
            return {"success": False, "message": "Thiếu symbol.", "positions": []}
        ok, data = self._send_request("GET", "/fapi/v3/positionRisk", {"symbol": sym})
        if not ok or not isinstance(data, list):
            msg = data.get("msg", "Lỗi đọc vị thế") if isinstance(data, dict) else "Lỗi đọc vị thế"
            return {"success": False, "message": msg, "positions": []}
        out = []
        for item in data:
            try:
                amt = float(item.get("positionAmt", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if abs(amt) <= 0:
                continue
            try:
                out.append({
                    "position_side": str(item.get("positionSide", "BOTH")).upper(),
                    "amount": amt,
                    "entry_price": float(item.get("entryPrice", 0.0) or 0.0),
                    "unrealized_pnl": float(item.get("unRealizedProfit", 0.0) or 0.0),
                    "leverage": float(item.get("leverage", 0.0) or 0.0),
                })
            except (TypeError, ValueError):
                continue
        return {"success": True, "positions": out}

    def test_connection(self) -> Dict[str, Any]:
        """
        Tests API connection, ping latency, and verifies account credentials.
        """
        if not self.api_key or not self.api_secret:
            return {
                "success": False,
                "message": "Vui lòng nhập đầy đủ Binance API Key và Secret Key!",
                "latency_ms": 0.0,
                "balance": 0.0,
                "can_trade": False
            }

        start_time = time.time()
        # 1. Test ping/time
        ok_time, time_data = self._send_request("GET", "/fapi/v1/time", signed=False)
        ping_latency = round((time.time() - start_time) * 1000, 1)

        if not ok_time:
            return {
                "success": False,
                "message": f"Không thể kết nối máy chủ Binance Futures: {time_data.get('msg', 'Timeout')}",
                "latency_ms": ping_latency,
                "balance": 0.0,
                "can_trade": False
            }

        # 2. Test authenticated account query
        ok_acc, acc_data = self._send_request("GET", "/fapi/v2/account", signed=True)
        if not ok_acc:
            err_code = acc_data.get("code", -1)
            raw_msg = acc_data.get("msg", "Khóa API không hợp lệ hoặc IP bị chặn")
            
            # Detailed actionable diagnostics for Binance Futures
            diag_lines = []
            if err_code == -2015:
                diag_lines.append(f"Mã lỗi -2015: {raw_msg}")
                diag_lines.append("👉 NGUYÊN NHÂN & HƯỚNG DẪN XỬ LÝ:")
                diag_lines.append("1. CHƯA BẬT QUYỀN FUTURES: Vào Binance > API Management > chọn Khóa này > bấm 'Edit restrictions' (Sửa quyền) > tích chọn [Enable Futures / Bật Hợp đồng Tương lai].")
                diag_lines.append(f"2. CHỌN SAI MẠNG: Bạn đang chọn mạng {'TESTNET' if self.is_testnet else 'MAINNET'}. Nếu khóa này tạo trên Binance Thật (Mainnet), vui lòng chuyển ô Mạng sang 'Binance Mainnet (Thật)' (hoặc ngược lại).")
                diag_lines.append("3. HẠN CHẾ IP: Nếu khóa API bật 'Restrict access to trusted IPs only', cần thêm IP mạng hiện tại của bạn vào whitelist trên Binance, hoặc chọn 'Unrestricted'.")
                diag_lines.append("4. SAI SECRET KEY: Kiểm tra xem Secret Key có bị copy thiếu ký tự nào không.")
            else:
                diag_lines.append(f"Xác thực thất bại từ Binance ({err_code}): {raw_msg}")

            return {
                "success": False,
                "code": err_code,
                "message": "\n".join(diag_lines),
                "latency_ms": ping_latency,
                "balance": 0.0,
                "can_trade": False
            }

        total_wallet = float(acc_data.get("totalWalletBalance", 0.0))
        avail_balance = float(acc_data.get("availableBalance", 0.0))
        can_trade = bool(acc_data.get("canTrade", False))

        return {
            "success": True,
            "message": f"Kết nối Binance thành công! Ví Futures: ${total_wallet:,.2f} USDT (Khả dụng: ${avail_balance:,.2f})",
            "latency_ms": ping_latency,
            "wallet_balance": total_wallet,
            "available_balance": avail_balance,
            "can_trade": can_trade,
            "account_type": "Binance USD(S)-M Futures",
            "network": "Testnet" if self.is_testnet else "Mainnet"
        }

    def fetch_commission_rate(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Fetches true user fee rate from Binance /fapi/v1/commissionRate
        """
        if not self.api_key or not self.api_secret:
            return {
                "success": False,
                "maker_commission": 0.0002,
                "taker_commission": 0.0005,
                "message": "Chưa có API Keys, sử dụng biểu phí VIP 0 mặc định."
            }

        ok, data = self._send_request("GET", "/fapi/v1/commissionRate", {"symbol": symbol}, signed=True)
        if ok and isinstance(data, dict):
            maker = float(data.get("makerCommissionRate", 0.0002))
            taker = float(data.get("takerCommissionRate", 0.0005))
            return {
                "success": True,
                "symbol": symbol,
                "maker_commission": maker,
                "taker_commission": taker,
                "maker_pct": round(maker * 100, 4),
                "taker_pct": round(taker * 100, 4),
                "message": f"Biểu phí tài khoản: Maker {maker*100:.3f}% | Taker {taker*100:.3f}%"
            }
        return {
            "success": False,
            "maker_commission": 0.0002,
            "taker_commission": 0.0005,
            "message": f"Không đọc được biểu phí: {data.get('msg', 'Lỗi API')}"
        }

    @staticmethod
    def is_filled_response(response: Any) -> bool:
        """A local position may only be written after a full exchange fill."""
        try:
            return str(response.get("status", "")).upper() == "FILLED" and BinanceAPIManager._positive(response.get("executedQty", 0.0))
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _format_number(value: float) -> str:
        text = format(Decimal(str(value)), "f")
        return text.rstrip("0").rstrip(".") if "." in text else text

    def place_order_live(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
        activation_price: Optional[float] = None,
        callback_rate: Optional[float] = None,
        close_position: bool = False,
        position_side: str = "BOTH",
        response_type: Optional[str] = None,
        working_type: str = "MARK_PRICE",
    ) -> Tuple[bool, Any]:
        """Place one validated USD-M order on the selected Binance environment.

        Binance parameters follow the current New Order specification:
        https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/trade
        """
        error = self._write_error()
        if error:
            return False, error

        requested_type = str(order_type).upper().strip()
        type_map = {"POST_ONLY": "LIMIT", "TRAILING_STOP": "TRAILING_STOP_MARKET", "CONDITIONAL": "STOP"}
        exchange_type = type_map.get(requested_type, requested_type)
        valid_types = {"LIMIT", "MARKET", "STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"}
        if exchange_type not in valid_types:
            return False, {"code": -2, "msg": f"Loại lệnh Binance Futures không hợp lệ: {order_type}"}
        if str(side).upper() not in ("BUY", "SELL"):
            return False, {"code": -2, "msg": "side chỉ được là BUY hoặc SELL."}
        position_side = str(position_side).upper()
        if position_side not in ("BOTH", "LONG", "SHORT"):
            return False, {"code": -2, "msg": "positionSide không hợp lệ."}
        if (position_side == "LONG" and str(side).upper() != "BUY" and not close_position) or (position_side == "SHORT" and str(side).upper() != "SELL" and not close_position):
            return False, {"code": -2, "msg": "positionSide không khớp side mở lệnh."}
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9_]{1,32}", symbol.upper()):
            return False, {"code": -2, "msg": "symbol không hợp lệ."}
        if not close_position and not self._positive(quantity):
            return False, {"code": -2, "msg": "quantity phải lớn hơn 0."}
        if any(value is not None and not self._positive(value) for value in (price, stop_price, activation_price, callback_rate)):
            return False, {"code": -2, "msg": "price/stopPrice/activationPrice/callbackRate phải hữu hạn và lớn hơn 0."}
        if close_position and reduce_only:
            return False, {"code": -2, "msg": "closePosition không được dùng cùng reduceOnly."}
        if position_side != "BOTH" and reduce_only:
            return False, {"code": -2, "msg": "Hedge mode không hỗ trợ reduceOnly; dùng positionSide/closePosition."}
        if client_order_id and not re.fullmatch(r"[.A-Za-z0-9_:/-]{1,36}", str(client_order_id)):
            return False, {"code": -2, "msg": "newClientOrderId không hợp lệ."}
        if working_type not in ("MARK_PRICE", "CONTRACT_PRICE"):
            return False, {"code": -2, "msg": "workingType không hợp lệ."}

        conditional = exchange_type not in ("MARKET", "LIMIT")
        params: Dict[str, Any] = {"symbol": symbol.upper(), "side": side.upper(), "type": exchange_type, "positionSide": position_side}
        if conditional:
            params["algoType"] = "CONDITIONAL"
        if not close_position:
            params["quantity"] = self._format_number(quantity)
        if reduce_only:
            params["reduceOnly"] = "true"
        if close_position:
            if exchange_type not in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
                return False, {"code": -2, "msg": "closePosition chỉ hỗ trợ STOP_MARKET hoặc TAKE_PROFIT_MARKET."}
            params["closePosition"] = "true"
        if client_order_id:
            params["clientAlgoId" if conditional else "newClientOrderId"] = client_order_id

        if exchange_type in ("LIMIT", "STOP", "TAKE_PROFIT"):
            if not self._positive(price):
                return False, {"code": -2, "msg": f"{exchange_type} cần price."}
            params["price"] = self._format_number(price)
            params["timeInForce"] = "GTX" if requested_type == "POST_ONLY" else "GTC"
        if exchange_type in ("STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
            if not self._positive(stop_price):
                return False, {"code": -3, "msg": "Lệnh điều kiện cần stopPrice."}
            params["triggerPrice"] = self._format_number(stop_price)
            params["workingType"] = working_type
        if exchange_type == "TRAILING_STOP_MARKET":
            if callback_rate is None or not 0.1 <= float(callback_rate) <= 10.0:
                return False, {"code": -3, "msg": "TRAILING_STOP_MARKET cần callbackRate trong khoảng 0.1-10."}
            params["callbackRate"] = self._format_number(callback_rate)
            params["workingType"] = working_type
            if activation_price is not None:
                if not self._positive(activation_price):
                    return False, {"code": -3, "msg": "activationPrice phải lớn hơn 0."}
                params["activatePrice"] = self._format_number(activation_price)

        params["newOrderRespType"] = response_type or ("RESULT" if exchange_type == "MARKET" else "ACK")
        if params["newOrderRespType"] not in ("ACK", "RESULT"):
            return False, {"code": -2, "msg": "newOrderRespType không hợp lệ."}
        ok, response = self._send_request("POST", "/fapi/v1/algoOrder" if conditional else "/fapi/v1/order", params, signed=True)
        if ok and conditional and (not isinstance(response, dict) or not self._positive(response.get("algoId"))):
            return False, {"code": -999, "msg": "Binance algo ACK thiếu algoId; phải reconcile theo clientAlgoId trước khi thử lại."}
        return (ok, self._normalize_algo(response)) if ok and conditional else (ok, response)

    @staticmethod
    def _normalize_algo(response: Any) -> Dict[str, Any]:
        """A triggered algo is not proof its executable child has filled."""
        result = dict(response)
        result["orderId"] = f"algo:{response['algoId']}"
        result["clientOrderId"] = response.get("clientAlgoId", "")
        status = str(response.get("algoStatus", "NEW")).upper()
        result["status"] = status if status in ("NEW", "CANCELED", "EXPIRED", "REJECTED") else "PENDING_TRIGGER"
        result["executedQty"] = "0"
        result["avgPrice"] = "0"
        return result

    @staticmethod
    def _order_reference(symbol: str, order_id: Optional[str], client_order_id: Optional[str]) -> Tuple[bool, Dict[str, Any]]:
        if not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9_]{1,32}", symbol.upper()):
            raise ValueError("symbol không hợp lệ")
        is_algo = str(order_id or "").startswith("algo:") or str(client_order_id or "").startswith("algo:")
        params: Dict[str, Any] = {} if is_algo else {"symbol": symbol.upper()}
        if order_id:
            raw_id = str(order_id).removeprefix("algo:")
            if not raw_id.isascii() or not raw_id.isdigit() or int(raw_id) <= 0:
                raise ValueError("orderId không hợp lệ")
            params["algoId" if is_algo else "orderId"] = raw_id
        elif client_order_id:
            raw_id = str(client_order_id).removeprefix("algo:")
            if not re.fullmatch(r"[.A-Za-z0-9_:/-]{1,36}", raw_id):
                raise ValueError("clientOrderId không hợp lệ")
            params["clientAlgoId" if is_algo else "origClientOrderId"] = raw_id
        return is_algo, params

    def cancel_order(self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Tuple[bool, Any]:
        error = self._write_error()
        if error:
            return False, error
        if not order_id and not client_order_id:
            return False, {"code": -2, "msg": "Cần orderId hoặc origClientOrderId để hủy lệnh."}
        try:
            is_algo, params = self._order_reference(symbol, order_id, client_order_id)
        except ValueError as exc:
            return False, {"code": -2, "msg": str(exc)}
        if not is_algo:
            return self._send_request("DELETE", "/fapi/v1/order", params, signed=True)
        ok, response = self._send_request("DELETE", "/fapi/v1/algoOrder", params, signed=True)
        if ok:
            if not isinstance(response, dict) or not self._positive(response.get("algoId")):
                return False, {"code": -999, "msg": "Binance cancel algo ACK thiếu algoId; cần reconcile."}
            return True, self._normalize_algo({**response, "algoStatus": "CANCELED"})
        # The trigger may have raced cancellation. Cancel its executable child, never invent a cancellation.
        query_ok, algo = self._send_request("GET", "/fapi/v1/algoOrder", params, signed=True)
        if query_ok and isinstance(algo, dict) and self._positive(algo.get("actualOrderId")) and self._positive(algo.get("algoId")):
            child_ok, child = self._send_request("DELETE", "/fapi/v1/order", {"symbol": symbol.upper(), "orderId": algo["actualOrderId"]}, signed=True)
            if child_ok:
                return True, {**child, "orderId": f"algo:{algo['algoId']}", "actualOrderId": algo["actualOrderId"]}
        return False, response

    def query_order(self, symbol: str, order_id: Optional[str] = None, client_order_id: Optional[str] = None) -> Tuple[bool, Any]:
        if not order_id and not client_order_id:
            return False, {"code": -2, "msg": "Cần orderId hoặc origClientOrderId để truy vấn lệnh."}
        try:
            is_algo, params = self._order_reference(symbol, order_id, client_order_id)
        except ValueError as exc:
            return False, {"code": -2, "msg": str(exc)}
        ok, response = self._send_request("GET", "/fapi/v1/algoOrder" if is_algo else "/fapi/v1/order", params, signed=True)
        if not ok or not is_algo:
            return ok, response
        if not isinstance(response, dict) or not self._positive(response.get("algoId")):
            return False, {"code": -999, "msg": "Binance query algo thiếu algoId; không xác nhận được trạng thái."}
        normalized = self._normalize_algo(response)
        if self._positive(response.get("actualOrderId")):
            child_ok, child = self._send_request("GET", "/fapi/v1/order", {"symbol": symbol.upper(), "orderId": response["actualOrderId"]}, signed=True)
            if not child_ok:
                return False, {**normalized, "reconciliationError": child}
            normalized.update(child)
            normalized["orderId"] = f"algo:{response['algoId']}"
            normalized["actualOrderId"] = response["actualOrderId"]
        return True, normalized
