import json
import os
import time
import joblib
import pandas as pd
from sklearn.metrics import f1_score
from llm import classify
from escalate import decide

TEST_PATH = "data/codemixed_test.csv"
if not os.path.exists(TEST_PATH):
    raise SystemExit(f"Missing {TEST_PATH}. Create the code-mixed test set first.")

baseline = joblib.load("baseline.joblib")
metrics = {}

# ---------- Code-mixed test set (locked) ----------
test = pd.read_csv(TEST_PATH)
test["expected"] = test["expected_category"].fillna("NONE")

rows = []
for i, r in test.iterrows():
    out = classify(r.message)
    esc, why = decide(out, r.message)
    rows.append({
        "message": r.message,
        "expected": r.expected,
        "llm_pred": out["category"],
        "baseline_pred": baseline.predict([r.message.lower()])[0],
        "confidence": out["confidence"],
        "urgency": out["urgency"],
        "expected_urgency": r.expected_urgency,
        "escalated": bool(esc),
        "reason": why,
        "reply": out["reply"],
    })
    print(f"code-mixed {i + 1}/{len(test)}")
    time.sleep(2)

res = pd.DataFrame(rows)
res["correct"] = res.llm_pred == res.expected
res["needed_escalation"] = (~res.correct) | (res.expected == "NONE")
res.to_csv("results_codemixed.csv", index=False)

scored = res[res.expected != "NONE"]
metrics["codemixed_n"] = int(len(res))
metrics["codemixed_llm_macro_f1"] = float(f1_score(scored.expected, scored.llm_pred, average="macro"))
metrics["codemixed_baseline_macro_f1"] = float(f1_score(scored.expected, scored.baseline_pred, average="macro"))
metrics["codemixed_llm_accuracy"] = float((scored.llm_pred == scored.expected).mean())
metrics["codemixed_urgency_accuracy"] = float((res.urgency == res.expected_urgency).mean())
metrics["escalation_rate"] = float(res.escalated.mean())
esc = res[res.escalated]
metrics["escalated_count"] = int(len(esc))
metrics["escalation_precision"] = float(esc.needed_escalation.mean()) if len(esc) else None
needed = res[res.needed_escalation]
metrics["needed_escalation_count"] = int(len(needed))
metrics["escalation_recall"] = float(needed.escalated.mean()) if len(needed) else None

# ---------- English held-out Bitext (sample, to save API quota) ----------
SAMPLE = 60
eng = pd.read_csv("bitext_test.csv").sample(SAMPLE, random_state=42)
eng_rows = []
for n, (_, r) in enumerate(eng.iterrows(), 1):
    out = classify(r.text)
    eng_rows.append({"expected": r.category, "llm_pred": out["category"],
                     "baseline_pred": baseline.predict([r.text])[0]})
    if n % 20 == 0:
        print(f"english {n}/{SAMPLE}")
    time.sleep(2)

e = pd.DataFrame(eng_rows)
metrics["english_n"] = int(len(e))
metrics["english_llm_macro_f1"] = float(f1_score(e.expected, e.llm_pred, average="macro"))
metrics["english_baseline_macro_f1"] = float(f1_score(e.expected, e.baseline_pred, average="macro"))

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))