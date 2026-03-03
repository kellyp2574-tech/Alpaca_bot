"""Quick test to inspect Massive snapshot schema"""
from massive import RESTClient
import os
from dotenv import load_dotenv
import json

load_dotenv()

client = RESTClient(os.getenv('MASSIVE_API_KEY'))
snapshot = client.get_snapshot_all('stocks', include_otc='false')

items = list(snapshot)
print(f'Total items: {len(items)}')
print(f'\nFirst item type: {type(items[0])}')

# Check if dict or object
first = items[0]
if isinstance(first, dict):
    print(f'\nFirst item is a dict with keys: {list(first.keys())}')
    print(f'\nFirst item:')
    print(json.dumps(first, indent=2, default=str))
else:
    print(f'\nFirst item is an object with attributes:')
    attrs = [x for x in dir(first) if not x.startswith('_')]
    print(f'Attributes: {attrs}')
    print(f'\nFirst item values:')
    for attr in attrs[:15]:
        try:
            val = getattr(first, attr)
            if not callable(val):
                print(f'  {attr}: {val}')
        except:
            pass

# Check prev_day structure
print(f'\n\n=== PREV DAY STRUCTURE ===')
for i, item in enumerate(items[:3]):
    if isinstance(item, dict):
        ticker = item.get('ticker') or item.get('symbol')
        prev = item.get('prev_day') or item.get('prevDay') or item.get('prev_daily_bar')
    else:
        ticker = getattr(item, 'ticker', None) or getattr(item, 'symbol', None)
        prev = getattr(item, 'prev_day', None) or getattr(item, 'prevDay', None) or getattr(item, 'prev_daily_bar', None)
    
    print(f'\nItem {i} - Ticker: {ticker}')
    print(f'  prev object type: {type(prev)}')
    
    if prev:
        if isinstance(prev, dict):
            print(f'  prev keys: {list(prev.keys())}')
            print(f'  prev values: {prev}')
        else:
            prev_attrs = [x for x in dir(prev) if not x.startswith('_')]
            print(f'  prev attributes: {prev_attrs}')
            for attr in prev_attrs[:10]:
                try:
                    val = getattr(prev, attr)
                    if not callable(val):
                        print(f'    {attr}: {val}')
                except:
                    pass
