"""Check candidate ledger stats"""
import json

with open('state/candidates/2026-03-03.json') as f:
    data = json.load(f)

print('=== MASSIVE SEED STATS ===')
print('Snapshot count seen:', data.get('snapshot_count_seen', 'N/A'))
print('Snapshot count with prev_obj:', data.get('snapshot_count_with_prev_obj', 'N/A'))
print('Seed total (usable):', data.get('seed_total', 'N/A'))
print('Seed selected:', data['seed_selected'])
print('Validated:', data['validated'])
print('Final candidates:', data['final'])
print()

print('=== DROP REASONS ===')
from collections import Counter
reasons = Counter([d['reason'] for d in data['drops']])
for r, c in reasons.most_common():
    print(f'{r}: {c}')
print()

print('=== SAMPLE DROPS ===')
for d in data['drops'][:5]:
    print(f"{d['symbol']}: {d['reason']} - {d.get('details', {})}")
