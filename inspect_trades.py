import urllib.request
import json

res = urllib.request.urlopen('http://127.0.0.1:8000/api/state')
d = json.loads(res.read())

print(f"Balance: ${d.get('balance', 0):,.2f} | Initial: ${d.get('initial_balance', 0):,.2f}")
print(f"Total Trades: {d.get('total_trades')} | Wins: {d.get('wins')} | Losses: {d.get('losses')} | Win Rate: {d.get('win_rate'):.1f}%")
print(f"Net PnL: ${d.get('net_pnl'):,.2f} ({d.get('net_pnl_pct'):.2f}%) | Total Fees: ${d.get('total_fees'):,.2f}")
print(f"Current Position: {d.get('position')}")
print("\n--- ALL TRADES HISTORY ---")
for t in d.get('trades', []):
    print(f"#{t.get('id')} {t.get('direction')} [{t.get('timeframe')}] | Entry: ${t.get('entry_price'):,.2f} ({t.get('entry_time')}) | Exit: ${t.get('exit_price'):,.2f} ({t.get('exit_time')}) | PnL: ${t.get('pnl'):+,.2f} (Fee: ${t.get('fee'):.2f}) | Reason: {t.get('reason')}")
