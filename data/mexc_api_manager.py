"""
MEXC Futures / Contract API Manager
Handles live authentication, real balance checking, fee rate querying,
and order execution on MEXC Contract (Futures) API.

Features:
- HMAC-SHA256 signature authentication (ApiKey, Request-Time, Signature)
- Configurable base URL (https://contract.mexc.com / https://api.mexc.com)
- Optional HTTP/HTTPS proxy support (for bypassing local ISP blocks)
- Account asset and margin balance verification (/api/v1/private/account/assets)
- Open positions and risk checks (/api/v1/private/position/open_positions)
- Real order placement with safety guardrails (/api/v1/private/order/submit)
- Safe credential masking (protects private keys)
- Network diagnostics for regional/ISP blocks (VNPT/Viettel/FPT TCP reset handling)
"""
import hashlib
import hmac
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple


ALLOWED_MEXC_HOSTS = {"contract.mexc.com", "contract.mexc.co", "api.mexc.co", "api.mexc.com"}
ALLOWED_SCHEMES = {"https"}


def validate_mexc_url(url: str) -> str:
    """Strictly validates MEXC host and scheme against an immutable allowlist."""
    if not url or not isinstance(url, str) or not url.strip():
        return "https://contract.mexc.co"
    raw = url.strip().rstrip("/")
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL scheme '{parsed.scheme}'; only HTTPS is allowed")
    host = (parsed.hostname or "").lower()
    if host not in ALLOWED_MEXC_HOSTS:
        raise ValueError(f"Disallowed MEXC host '{host}'; allowed hosts: {ALLOWED_MEXC_HOSTS}")
    if parsed.username or parsed.password:
        raise ValueError("Credentials in URL are strictly forbidden")
    if parsed.port and parsed.port != 443:
        raise ValueError("Non-standard ports are forbidden")
    return f"https://{host}"


class MexcAPIManager:
    # Official mirrors: contract.mexc.co is the unblocked mirror for Vietnam & restricted regions
    DEFAULT_BASE_URL = "https://contract.mexc.co"
    ALT_BASE_URL = "https://contract.mexc.com"
    MIRROR_CANDIDATES = [
        "https://contract.mexc.co",
        "https://contract.mexc.com",
        "https://api.mexc.co"
    ]

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = "",
        proxy_url: str = "",
        is_live_enabled: bool = False
    ):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        self.base_url = validate_mexc_url(base_url or self.DEFAULT_BASE_URL)
        self.proxy_url = proxy_url.strip()
        self.is_live_enabled = is_live_enabled

    def set_credentials(self, api_key: str, api_secret: str, base_url: str = "", proxy_url: str = ""):
        self.api_key = api_key.strip()
        self.api_secret = api_secret.strip()
        if base_url:
            self.base_url = validate_mexc_url(base_url)
        if proxy_url is not None:
            self.proxy_url = proxy_url.strip()

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
            "exchange": "mexc",
            "mexc_api_key_masked": masked_key,
            "has_mexc_secret": has_secret,
            "mexc_api_secret_masked": masked_secret,
            "mexc_base_url": self.base_url,
            "mexc_proxy_url": self.proxy_url,
            "is_live_enabled": self.is_live_enabled,
            "network_url": self.base_url
        }

    def _generate_signature(self, req_time: str, param_str: str) -> str:
        """
        MEXC Contract signature: HMAC-SHA256 of (api_key + req_time + param_str)
        """
        to_sign = f"{self.api_key}{req_time}{param_str}"
        return hmac.new(
            self.api_secret.encode("utf-8"),
            to_sign.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _get_opener(self) -> urllib.request.OpenerDirector:
        if self.proxy_url:
            proxy_handler = urllib.request.ProxyHandler({
                "http": self.proxy_url,
                "https": self.proxy_url
            })
            return urllib.request.build_opener(proxy_handler)
        return urllib.request.build_opener()

    def _send_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = True
    ) -> Tuple[bool, Any]:
        if params is None:
            params = {}

        req_time = str(int(time.time() * 1000))
        headers = {
            "User-Agent": "Antigravity-Quant-Terminal/2.0",
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}{endpoint}"
        body_bytes = None
        param_str = ""

        if method in ("GET", "DELETE"):
            if params:
                sorted_items = sorted(params.items(), key=lambda x: x[0])
                param_str = urllib.parse.urlencode(sorted_items)
                url = f"{url}?{param_str}"
        else:
            if params:
                param_str = json.dumps(params, separators=(',', ':'), ensure_ascii=False)
                body_bytes = param_str.encode("utf-8")

        if signed:
            if not self.api_key or not self.api_secret:
                return False, {"code": -1, "msg": "Chưa cấu hình MEXC ApiKey hoặc SecretKey."}
            signature = self._generate_signature(req_time, param_str)
            headers["ApiKey"] = self.api_key
            headers["Request-Time"] = req_time
            headers["Signature"] = signature

        req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
        opener = self._get_opener()

        try:
            with opener.open(req, timeout=6.0) as resp:
                resp_text = resp.read().decode("utf-8")
                try:
                    data = json.loads(resp_text)
                    return True, data
                except Exception:
                    return True, {"raw": resp_text}
        except urllib.error.HTTPError as err:
            err_text = err.read().decode("utf-8")
            try:
                err_json = json.loads(err_text)
                return False, err_json
            except Exception:
                return False, {"code": err.code, "msg": err_text or str(err)}
        except (ConnectionResetError, urllib.error.URLError, Exception) as e:
            err_msg = str(e)
            is_reset = isinstance(e, ConnectionResetError) or ("10054" in err_msg) or ("reset" in err_msg.lower()) or ("refused" in err_msg.lower())
            
            # If request failed on a blocked domain, automatically fallback to unblocked mirror https://contract.mexc.co
            if is_reset and not getattr(self, "_in_fallback", False) and self.base_url != "https://contract.mexc.co":
                print(f"[MEXC API] Phát hiện tên miền bị chặn ISP ({self.base_url}). Tự động chuyển hướng sang máy chủ dự phòng unblocked: https://contract.mexc.co", flush=True)
                self._in_fallback = True
                self.base_url = "https://contract.mexc.co"
                try:
                    res = self._send_request(method, endpoint, params, signed)
                    return res
                finally:
                    self._in_fallback = False

            if is_reset:
                return False, {
                    "code": -10054,
                    "msg": (
                        "Kết nối bị ngắt quãng (Connection Reset / WinError 10054 do nhà mạng VN chặn mexc.com).\n"
                        "👉 Hệ thống khuyến nghị: Chuyển Base URL sang https://contract.mexc.co (Máy chủ dự phòng unblocked)."
                    )
                }
            return False, {"code": -999, "msg": f"Lỗi kết nối máy chủ MEXC: {err_msg}"}

    def test_connection(self) -> Dict[str, Any]:
        """
        Tests API connection, ping latency, and verifies account credentials on MEXC.
        """
        if not self.api_key or not self.api_secret:
            return {
                "success": False,
                "exchange": "MEXC",
                "message": "Vui lòng nhập đầy đủ MEXC ApiKey và SecretKey!",
                "latency_ms": 0.0,
                "balance": 0.0,
                "can_trade": False
            }

        start_time = time.time()
        # 1. Ping server
        ok_ping, ping_data = self._send_request("GET", "/api/v1/contract/ping", signed=False)
        latency = round((time.time() - start_time) * 1000, 1)

        if not ok_ping:
            return {
                "success": False,
                "exchange": "MEXC",
                "message": f"Không thể kết nối máy chủ MEXC Futures: {ping_data.get('msg', 'Timeout hoặc kết nối bị chặn')}",
                "latency_ms": latency,
                "balance": 0.0,
                "can_trade": False
            }

        # 2. Authenticated account query (/api/v1/private/account/assets)
        ok_assets, asset_data = self._send_request("GET", "/api/v1/private/account/assets", signed=True)
        if not ok_assets:
            err_code = asset_data.get("code", -1)
            raw_msg = asset_data.get("message") or asset_data.get("msg") or "Khóa API MEXC không hợp lệ hoặc IP bị chặn"

            diag_lines = []
            diag_lines.append(f"Xác thực thất bại từ MEXC (Mã: {err_code}): {raw_msg}")
            diag_lines.append("👉 HƯỚNG DẪN XỬ LÝ:")
            diag_lines.append("1. BẬT QUYỀN CONTRACT/FUTURES: Vào MEXC > API Management > Đảm bảo đã tích chọn quyền 'Contract' (Hợp đồng tương lai).")
            diag_lines.append("2. KIỂM TRA SECRET KEY: Đảm bảo Secret Key được copy chính xác, không thừa dấu cách.")
            diag_lines.append("3. IP WHITELIST: Nếu đã bật IP Whitelist trên MEXC, hãy thêm IP máy chủ hiện tại vào danh sách.")

            return {
                "success": False,
                "exchange": "MEXC",
                "code": err_code,
                "message": "\n".join(diag_lines),
                "latency_ms": latency,
                "balance": 0.0,
                "can_trade": False
            }

        # Parse balances
        assets = asset_data.get("data", [])
        total_equity = 0.0
        avail_balance = 0.0

        if isinstance(assets, list):
            for item in assets:
                curr = item.get("currency", "")
                if curr.upper() == "USDT":
                    total_equity = float(item.get("equity", item.get("cashBalance", 0.0)))
                    avail_balance = float(item.get("availableBalance", 0.0))
                    break
            else:
                for item in assets:
                    total_equity += float(item.get("equity", 0.0))
                    avail_balance += float(item.get("availableBalance", 0.0))

        return {
            "success": True,
            "exchange": "MEXC",
            "message": f"Kết nối MEXC thành công! Ví Futures: ${total_equity:,.2f} USDT (Khả dụng: ${avail_balance:,.2f})",
            "latency_ms": latency,
            "wallet_balance": total_equity,
            "available_balance": avail_balance,
            "can_trade": True,
            "account_type": "MEXC Contract (USDT-M Futures)",
            "network": "MEXC Mainnet"
        }

    def fetch_commission_rate(self, symbol: str = "BTC_USDT") -> Dict[str, Any]:
        """
        Fetches trading fee rate from MEXC or returns standard competitive rates:
        MEXC standard Futures fee: Maker 0.000%, Taker 0.020%
        """
        maker_rate = 0.0000  # 0% Maker
        taker_rate = 0.0002  # 0.02% Taker

        if not self.api_key or not self.api_secret:
            return {
                "success": True,
                "symbol": symbol,
                "maker_commission": maker_rate,
                "taker_commission": taker_rate,
                "maker_pct": 0.0,
                "taker_pct": 0.02,
                "message": "MEXC Futures Standard: Maker 0.00% | Taker 0.02%"
            }

        ok, fee_data = self._send_request("GET", "/api/v1/private/account/trade_fee_rate", signed=True)
        if ok and isinstance(fee_data, dict) and fee_data.get("success"):
            data = fee_data.get("data", {})
            maker_rate = float(data.get("makerFeeRate", maker_rate))
            taker_rate = float(data.get("takerFeeRate", taker_rate))

        return {
            "success": True,
            "symbol": symbol,
            "maker_commission": maker_rate,
            "taker_commission": taker_rate,
            "maker_pct": round(maker_rate * 100, 4),
            "taker_pct": round(taker_rate * 100, 4),
            "message": f"Biểu phí MEXC: Maker {maker_rate*100:.3f}% | Taker {taker_rate*100:.3f}%"
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
        Executes a real order on MEXC Contract (Futures) API.
        """
        if not self.is_live_enabled:
            return False, {"code": -998, "msg": "Giao dịch thật (Live Trading) chưa được bật. Vui lòng bật trong Cài đặt!"}

        if not self.api_key or not self.api_secret:
            return False, {"code": -1, "msg": "Thiếu MEXC ApiKey hoặc SecretKey."}

        # Format symbol for MEXC: e.g. BTCUSDT -> BTC_USDT
        mexc_symbol = symbol
        if "_" not in mexc_symbol and mexc_symbol.endswith("USDT"):
            base = mexc_symbol[:-4]
            mexc_symbol = f"{base}_USDT"

        side_upper = side.upper()
        if side_upper == "BUY":
            mexc_side = 4 if reduce_only else 1
        elif side_upper == "SELL":
            mexc_side = 2 if reduce_only else 3
        else:
            mexc_side = 1

        is_market = order_type.upper() in ("MARKET", "STOP_MARKET")
        mexc_type = 5 if is_market else 1

        payload: Dict[str, Any] = {
            "symbol": mexc_symbol,
            "vol": int(max(1, quantity)),
            "side": mexc_side,
            "type": mexc_type,
            "openType": 1
        }

        if not is_market:
            if price is None or price <= 0:
                return False, {"code": -2, "msg": "Lệnh LIMIT cần chỉ định mức giá price."}
            payload["price"] = round(price, 2)

        return self._send_request("POST", "/api/v1/private/order/submit", payload, signed=True)
