import json
import time

import joblib
import pandas as pd
from sklearn.metrics import f1_score

from llm import classify
from escalate import decide


baseline = joblib.load("baseline.joblib")
metrics = {}

# ---------- Code-mixed test set ----------
test = pd.read_csv("data/codemixed_test.csv")
test["expected"] = test["expected_category"].fillna("NONE")

rows = []
for _, row in test.iterrows():
	out = classify(row.message)
	escalated, reason = decide(out, row.message)
	rows.append(
		{
			"message": row.message,
			"expected": row.expected,
			"llm_pred": out["category"],
			"baseline_pred": baseline.predict([row.message.lower()])[0],
			"confidence": out["confidence"],
			"urgency": out["urgency"],
			"expected_urgency": row.expected_urgency,
			"escalated": bool(escalated),
			"reason": reason,
			"reply": out["reply"],
		}
	)
	time.sleep(1)

res = pd.DataFrame(rows)
res["correct"] = res.llm_pred == res.expected
res["needed_escalation"] = (~res.correct) | (res.expected == "NONE")
res.to_csv("results_codemixed.csv", index=False)

scored = res[res.expected != "NONE"]
metrics["codemixed_n"] = int(len(res))
metrics["codemixed_llm_macro_f1"] = float(
	f1_score(scored.expected, scored.llm_pred, average="macro")
)
metrics["codemixed_baseline_macro_f1"] = float(
	f1_score(scored.expected, scored.baseline_pred, average="macro")
)
metrics["codemixed_llm_accuracy"] = float(
	(scored.llm_pred == scored.expected).mean()
)
metrics["codemixed_urgency_accuracy"] = float(
	(res.urgency == res.expected_urgency).mean()
)
metrics["escalation_rate"] = float(res.escalated.mean())
escalated_rows = res[res.escalated]
metrics["escalated_count"] = int(len(escalated_rows))
metrics["escalation_precision"] = (
	float(escalated_rows.needed_escalation.mean()) if len(escalated_rows) else None
)
needed = res[res.needed_escalation]
metrics["needed_escalation_count"] = int(len(needed))
metrics["escalation_recall"] = (
	float(needed.escalated.mean()) if len(needed) else None
)

# ---------- English held-out Bitext (sample, to save API quota) ----------
eng = pd.read_csv("bitext_test.csv").sample(200, random_state=42)
eng_rows = []
for _, row in eng.iterrows():
	out = classify(row.text)
	eng_rows.append(
		{
			"expected": row.category,
			"llm_pred": out["category"],
			"baseline_pred": baseline.predict([row.text])[0],
		}
	)
	time.sleep(1)

english = pd.DataFrame(eng_rows)
metrics["english_n"] = int(len(english))
metrics["english_llm_macro_f1"] = float(
	f1_score(english.expected, english.llm_pred, average="macro")
)
metrics["english_baseline_macro_f1"] = float(
	f1_score(english.expected, english.baseline_pred, average="macro")
)

with open("metrics.json", "w") as metrics_file:
	json.dump(metrics, metrics_file, indent=2)

print(json.dumps(metrics, indent=2))
