import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Code-Mixed Support Triage", layout="wide")
st.title("Code-Mixed Support Triage Assistant")
st.caption(
	"Reads English, Tanglish and Hinglish support messages, labels category and urgency, "
	"drafts a reply, and hands off to a human when unsure."
)

# ---- Load the classifier (needs GROQ_API_KEY) ----
classify = decide = None
load_error = None
try:
	from llm import classify
	from escalate import decide
except Exception as e:  # missing key or file
	load_error = str(e)

if "log" not in st.session_state:
	st.session_state.log = []

URGENCY_COLOR = {"high": "red", "medium": "orange", "low": "green"}

tab_inbox, tab_results, tab_queue = st.tabs(["Inbox", "Test results", "Queue dashboard"])

# ------------------------------------------------------------------ Inbox
with tab_inbox:
	if load_error:
		st.error(f"Could not load the model. Check that GROQ_API_KEY is set. Details: {load_error}")
	msg = st.text_area("Customer message", "order vandhuchu but box damaged, refund venum", height=100)
	if st.button("Triage", type="primary") and msg.strip() and not load_error:
		with st.spinner("Classifying..."):
			out = classify(msg)
			esc, why = decide(out, msg)
		c1, c2, c3 = st.columns(3)
		c1.metric("Category", out["category"])
		c2.markdown("**Urgency**")
		c2.markdown(f":{URGENCY_COLOR.get(out['urgency'], 'gray')}[**{out['urgency'].upper()}**]")
		c3.metric("Confidence", f"{out['confidence']:.2f}")
		st.progress(min(max(float(out["confidence"]), 0.0), 1.0))
		if esc:
			st.error(f"Escalate to human: {why}")
		else:
			st.success("Handled automatically: confidence is above the threshold.")
		st.text_area("Draft reply (editable)", out["reply"], height=100)
		st.caption(f"Model reasoning: {out['reasoning']}")
		st.session_state.log.append(
			{"message": msg, "category": out["category"], "urgency": out["urgency"],
			 "confidence": out["confidence"], "escalated": bool(esc)}
		)

# ---------------------------------------------------------------- Results
with tab_results:
	st.subheader("Evaluation on the team-written code-mixed test set")
	try:
		with open("metrics.json") as f:
			m = json.load(f)
		table = pd.DataFrame(
			{
				"English (Bitext sample)": [m.get("english_baseline_macro_f1"), m.get("english_llm_macro_f1")],
				"Code-mixed (our test set)": [m.get("codemixed_baseline_macro_f1"), m.get("codemixed_llm_macro_f1")],
			},
			index=["Baseline (TF-IDF + logistic regression)", "LLM classifier"],
		).round(3)
		st.markdown("**Macro-F1**")
		st.table(table)
		a, b, c, d = st.columns(4)
		a.metric("Code-mixed messages", m.get("codemixed_n"))
		b.metric("Escalation rate", f"{m['escalation_rate']:.0%}" if m.get("escalation_rate") is not None else "n/a")
		ep = m.get("escalation_precision")
		c.metric("Escalation precision", f"{ep:.0%}" if ep is not None else "n/a")
		er = m.get("escalation_recall")
		d.metric("Wrong or vague caught", f"{er:.0%}" if er is not None else "n/a")
		st.caption(
			f"English score uses a random sample of {m.get('english_n')} held-out Bitext rows. "
			"Escalation precision = share of escalated messages that really needed a human. "
			"Escalation recall = share of wrong or vague predictions that were escalated."
		)
	except FileNotFoundError:
		st.info("metrics.json not found. Run eval.py first.")

	try:
		res = pd.read_csv("results_codemixed.csv")
		st.markdown("**Per-message results**")
		show = res[["message", "expected", "llm_pred", "baseline_pred", "confidence",
					"urgency", "expected_urgency", "escalated", "reason"]]
		st.dataframe(show, use_container_width=True)
	except FileNotFoundError:
		st.info("results_codemixed.csv not found. Run eval.py first.")

	st.caption(
		"Notes: urgency labels are our own definition (the dataset has none). The training data is "
		"synthetic English with no Tanglish or Hinglish. The test set is small, so treat scores as indicative."
	)

# ------------------------------------------------------------------ Queue
with tab_queue:
	st.subheader("Messages triaged in this session")
	if st.session_state.log:
		log = pd.DataFrame(st.session_state.log)
		left, right = st.columns(2)
		left.markdown("**By category**")
		left.bar_chart(log["category"].value_counts())
		right.markdown("**By urgency**")
		right.bar_chart(log["urgency"].value_counts())
		st.dataframe(log, use_container_width=True)
	else:
		st.write("No messages yet. Triage a message in the Inbox tab.")

st.sidebar.markdown(
	"**Note:** replies are drafts for a human agent to review. "
	"The assistant escalates when it is unsure instead of guessing."
)
