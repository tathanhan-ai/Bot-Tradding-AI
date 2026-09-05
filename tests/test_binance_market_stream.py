import unittest

import pandas as pd

from data.binance_ws_stream import normalize_depth_message, normalize_kline_message
from data.fetcher import discard_forming_candle


class BinanceMarketStreamTest(unittest.TestCase):
    def test_depth20_payload_keeps_exchange_levels_without_synthesis(self):
        payload = {
            "E": 1_700_000_000_000,
            "b": [[str(60_000 - index), str(index + 1)] for index in range(20)],
            "a": [[str(60_001 + index), str(index + 1)] for index in range(20)],
        }

        depth = normalize_depth_message(payload)

        self.assertIsNotNone(depth)
        bids, asks, event_time = depth
        self.assertEqual(len(bids), 20)
        self.assertEqual(len(asks), 20)
        self.assertEqual(bids[3], [59_997.0, 4.0])
        self.assertEqual(asks[3], [60_004.0, 4.0])
        self.assertEqual(event_time, 1_700_000_000_000)

    def test_incomplete_or_invalid_depth_is_rejected(self):
        self.assertIsNone(normalize_depth_message({"b": [["1", "1"]] * 19, "a": [["1", "1"]] * 20}))
        self.assertIsNone(normalize_depth_message({"b": [["x", "1"]] * 20, "a": [["1", "1"]] * 20}))

    def test_kline_preserves_real_volume_and_close_flag(self):
        message = {
            "E": 1_700_000_060_000,
            "k": {
                "i": "15m", "t": 1_700_000_000_000, "o": "60000", "h": "60100",
                "l": "59900", "c": "60050", "v": "12.34", "q": "740000", "x": True,
            },
        }

        kline = normalize_kline_message(message)

        self.assertEqual(kline["timeframe"], "15m")
        self.assertTrue(kline["is_closed"])
        self.assertEqual(kline["volume"], 12.34)
        self.assertEqual(kline["quote_volume"], 740000.0)

    def test_rest_history_discards_the_current_forming_bar(self):
        frame = pd.DataFrame(
            {"close": [100.0, 101.0]},
            index=pd.to_datetime([880, 941], unit="s"),
        )

        closed = discard_forming_candle(frame, interval_seconds=60, now=1_000)

        self.assertEqual(list(closed["close"]), [100.0])


if __name__ == "__main__":
    unittest.main()
