# prepare_cycle_data.py

import pandas as pd
import json
import os

# === 載入語法事件日誌
with open('wisdom_rhythm.json', 'r') as f:
    event_log = json.load(f)

# === 整理資料
cycles = []
current_cycle = []
cycle_id = 1

for event in event_log:
    event['cycle'] = cycle_id
    current_cycle.append(event)
    
    if event['event'] == "Ω":
        cycles.append(current_cycle)
        current_cycle = []
        cycle_id += 1

if current_cycle:
    cycles.append(current_cycle)

# === 為每一個cycle建立一個csv
os.makedirs('cycles', exist_ok=True)

for idx, cycle in enumerate(cycles, start=1):
    df = pd.DataFrame(cycle)
    cycle_path = os.path.join('cycles', f'cycle_{idx}.csv')
    df.to_csv(cycle_path, index=False)
    print(f"✅ 已儲存 {cycle_path}")
