import urllib.request
import json
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

res = urllib.request.urlopen('http://127.0.0.1:8000/api/state')
d = json.loads(res.read())

print(f"=== SUMMARY ===")
print(f"Balance: ${d.get('balance', 0):,.2f} | Initial: ${d.get('initial_balance', 0):,.2f}")
print(f"Total Trades: {d.get('total_trades')} | Wins: {d.get('wins')} | Losses: {d.get('losses')} | Win Rate: {d.get('win_rate'):.1f}%")
print(f"Net PnL: ${d.get('net_pnl'):,.2f} ({d.get('net_pnl_pct'):.2f}%) | Total Fees: ${d.get('total_fees'):,.2f}")

trades = d.get('trades', [])
print(f"\n=== TRADES BREAKDOWN (Total: {len(trades)}) ===")
reasons_count = {}
durations = []

for idx, t in enumerate(trades):
    reason = t.get('reason', '')
    reasons_count[reason] = reasons_count.get(reason, 0) + 1
    pnl = t.get('pnl', 0.0)
    fee = t.get('fee', 0.0)
    direction = t.get('direction', '')
    tf = t.get('timeframe', '')
    entry_p = t.get('entry_price', 0.0)
    exit_p = t.get('exit_price', 0.0)
    entry_t = t.get('entry_time', '')
    exit_t = t.get('exit_time', '')
    gross_pnl = pnl + fee  # gross before fee
    print(f"#{idx+1:02d} | {direction} [{tf}] | In: ${entry_p:,.1f} ({entry_t}) -> Out: ${exit_p:,.1f} ({exit_t}) | Gross PnL: ${gross_pnl:+.3f} | Fee: ${fee:.3f} | Net: ${pnl:+.3f} | Reason: {reason}")

print("\n=== EXIT REASONS AGGREGATION ===")
for r, c in sorted(reasons_count.items(), key=lambda x: x[1], reverse=True):
    print(f"- {c} trades ({c/len(trades)*100:.1f}%): {r}")
