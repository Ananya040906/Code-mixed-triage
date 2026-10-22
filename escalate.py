import joblib

baseline = joblib.load("baseline.joblib")


def decide(llm_out, message, threshold=0.6, baseline_margin=None):
    """Returns (escalate: bool, reason: str).
    baseline_margin: only let the baseline veto the LLM if the baseline is
    at least this confident. Set to None to turn the baseline rule off."""
    reasons = []

    if llm_out["category"] == "NONE":
        reasons.append("unclear, off-topic or too vague")
    if llm_out["confidence"] < threshold:
        reasons.append(f"low confidence ({llm_out['confidence']:.2f})")
    if "LLM call failed" in llm_out.get("reasoning", ""):
        reasons.append("model call failed")
    if llm_out["urgency"] == "high" and llm_out["confidence"] < 0.8:
        reasons.append("high-urgency with moderate confidence")

    if baseline_margin is not None and llm_out["category"] != "NONE":
        probs = baseline.predict_proba([message.lower()])[0]
        b_pred = baseline.classes_[probs.argmax()]
        b_conf = float(probs.max())
        if llm_out["category"] != b_pred and b_conf >= baseline_margin:
            reasons.append(f"LLM says {llm_out['category']} but baseline says {b_pred}")

    return len(reasons) > 0, "; ".join(reasons)