import time
import pandas as pd
from llm import classify
from escalate import decide

dev = pd.read_csv("dev_set.csv")

results = []
for _, r in dev.iterrows():
    out = classify(r.message)
    exp = r.expected_category if pd.notna(r.expected_category) else "NONE"
    results.append((r.message, exp, out))
    time.sleep(1)

print("\nPer message:")
for msg, exp, out in results:
    ok = "OK " if out["category"] == exp else "BAD"
    print(f"{ok} exp={exp:12} got={out['category']:12} conf={out['confidence']:.2f}  {msg}")

print("\nSweep (wrong caught = escalated AND wrong; false alarm = escalated BUT correct)")
print("threshold | baseline rule | escalated | wrong caught | wrong missed | false alarms")
for margin in [None, 0.8, 0.5]:
    for th in [0.5, 0.7, 0.9]:
        esc = caught = missed = false_alarm = 0
        for msg, exp, out in results:
            wrong = out["category"] != exp
            needed = wrong or exp == "NONE"   # these should be escalated
            e, _ = decide(out, msg, threshold=th, baseline_margin=margin)
            esc += e
            caught += needed and e
            missed += needed and not e
            false_alarm += (not needed) and e
        label = "off" if margin is None else f">= {margin}"
        print(f"   {th}    |   {label:8}    |   {esc:2}/{len(results)}   |      {caught}       |      {missed}       |      {false_alarm}")