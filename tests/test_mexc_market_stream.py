import time
import unittest

from data.mexc_ws_stream import normalize_mexc_deal, normalize_mexc_depth, normalize_mexc_kline
from trading.pipeline import MarketSnapshot


class MexcMarketStreamTest(unittest.TestCase):
    def test_full_depth20_and_deal_keep_exchange_values(self):
        depth = normalize_mexc_depth({"channel": "push.depth.full", "data": {
            "timestamp": 1_700_000_000_000,
            "bids": [[str(60_000 - n), str(n + 1), "1"] for n in range(20)],
            "asks": [[str(60_001 + n), str(n + 1), "1"] for n in range(20)],
        }})
        self.assertIsNotNone(depth)
        bids, asks, event_time = depth
        self.assertEqual((len(bids), len(asks), event_time), (20, 20, 1_700_000_000_000))
        self.assertEqual(bids[4], [59_996.0, 5.0])
        self.assertEqual(normalize_mexc_deal({"data": {"p": "60000", "v": "0.1", "T": 2, "t": 1_700_000_000_000}}),
                         (60_000.0, 0.1, True, 1_700_000_000_000))

    def test_kline_is_real_ohlcv_and_mexc_paper_snapshot_is_allowed(self):
        row = normalize_mexc_kline({"data": {"t": 1_700_000_000, "o": "60000", "h": "60100", "l": "59900", "c": "60050", "v": "12.3", "a": "738000"}}, "1m")
        self.assertEqual(row["quote_volume"], 738000.0)
        now = time.time()
        snapshot = MarketSnapshot(
            snapshot_id="mexc", symbol="BTCUSDT", exchange="mexc", captured_at=now, price=60_000,
            source_times={"depth": now, "agg_trade": now, "kline_1m": now},
            bids=[[59_999 - n, 1] for n in range(20)], asks=[[60_001 + n, 1] for n in range(20)], environment="paper",
        )
        self.assertNotIn("MEXC market data is paper-only", snapshot.freshness_issues(now))


if __name__ == "__main__":
    unittest.main()
