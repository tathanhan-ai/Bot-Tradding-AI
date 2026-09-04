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
import time
import urllib.parse
import urllib.request
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

    @property
    def base_url(self) -> str:
        return self.TESTNET_BASE_URL if self.is_testnet else self.MAINNET_BASE_URL

    def set_credentials(self, api_key: str, api_secret: str, is_testnet: bool):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.is_testnet = is_testnet

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
        params["timestamp"] = int(time.time() * 1000)
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        return f"{query_string}&signature={signature}"

    def _send_request(self, method: str, endpoint: str, params: Optional[Dict[str, Any]] = None, signed: bool = True) -> Tuple[bool, Any]:
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
                return False, err_json
            except Exception:
                return False, {"code": err.code, "msg": err_text or str(err)}
        except Exception as e:
            return False, {"code": -999, "msg": f"Lỗi kết nối mạng: {str(e)}"}

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

    def place_order_live(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        reduce_only: bool = False
    ) -> Tuple[bool, Any]:
        """
        Executes a real order on Binance USD(S)-M Futures when live mode is active.
        """
        if not self.is_live_enabled:
            return False, {"code": -998, "msg": "Giao dịch thật (Live Trading) chưa được bật. Vui lòng bật trong Cài đặt!"}

        if not self.api_key or not self.api_secret:
            return False, {"code": -1, "msg": "Thiếu Binance API Key hoặc Secret Key."}

        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": round(quantity, 3),
        }

        if reduce_only:
            params["reduceOnly"] = "true"

        if order_type.upper() in ("LIMIT", "STOP", "TAKE_PROFIT"):
            if price is None or price <= 0:
                return False, {"code": -2, "msg": "Lệnh LIMIT cần chỉ định giá price."}
            params["price"] = round(price, 2)
            params["timeInForce"] = "GTC"

        if order_type.upper() in ("STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
            if stop_price is None or stop_price <= 0:
                return False, {"code": -3, "msg": "Lệnh điều kiện cần stopPrice."}
            params["stopPrice"] = round(stop_price, 2)

        return self._send_request("POST", "/fapi/v1/order", params, signed=True)
