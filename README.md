Code-Mixed Support Triage Assistant

A small tool that reads a customer support message written in English, Tanglish or Hinglish, works out what it is about and how urgent it is, drafts a reply in the customer's own language mix, and passes the message to a human, with a reason, whenever it is not sure.

Built for the hackathon, problem statement 1.1 (Agentic AI and Intelligent Automation).

Live app: https://code-mixed-triage.streamlit.app/ Code: https://github.com/Vigasni/code-mixed-triage

Why we built it this way Real customers do not write "I would like to request a refund". They write "order vandhuchu but box damaged, refund venum". Support teams spend hours reading and sorting messages like this by hand.

The easy part is getting a model to guess a label. The hard part, and the part we cared about, is knowing when not to guess. A triage bot that confidently sends a payment problem to the wrong queue is worse than one that says "I'm not sure, please look at this." So most of our effort went into measuring how the system behaves on messages it has never seen, and into making its escalations meaningful.

What it does

Paste a message into the Inbox tab and you get:

• a category (one of 11, such as REFUND, ORDER, ACCOUNT)

• an urgency level (low, medium or high)

• a confidence score

• a draft reply in the same language mix as the customer, which a human can edit

• an escalate to human flag with a one-line reason when the system is unsure

The app has three tabs: Inbox, Test results (our evaluation on a code-mixed test set), and a small Queue dashboard that counts the messages triaged in the current session.

How it works

Baseline. A TF-IDF and logistic regression classifier trained on the Bitext data. It is fast and simple, and it is our point of comparison.

LLM classifier. openai/gpt-oss-120b through Groq. The prompt contains a label guide, ten short examples we wrote ourselves, and an instruction to report low confidence when a message is unclear. The model returns JSON, which we validate. If the call fails, we treat the message as unclear rather than crash.

Escalation rules. A message goes to a human if any of these is true:

o the model says it is off-topic, a greeting, gibberish or too vague (category NONE)

o confidence is below 0.6

o urgency is high but confidence is below 0.8

o the model call failed

The LLM is called from the backend only. The API key lives in an environment variable or a hosting secret and is never in the code or the repository.

How we defined urgency

The dataset has no urgency labels, so urgency is our own definition, not ground truth:

Level Meaning

high money lost or stuck, damaged or wrong item, account locked, anger, repeated complaint medium delivery delays, order or address changes, needs an answer soon low general questions, invoices, newsletter, feedback

Data

• Training and English evaluation: the Bitext Customer Support dataset (26,872 English examples, 11 categories, 27 intents, CDLA-Sharing-1.0 licence). We removed placeholders such as {{Order Number}} and lowercased the text.

• Code-mixed test set: 40 Tanglish and Hinglish messages in data/codemixed_test.csv, with expected category, urgency and whether the message should be escalated. <Edit this line so it is true. For example: "We wrote the first drafts with help from an AI assistant, then checked and reworded them ourselves." or "We wrote these ourselves.">

• The code-mixed set was kept out of every prompt and out of threshold tuning. We tuned only on a separate 28-message development set (dev_set.csv).

A few things about the labels are worth knowing. In Bitext, "cancel my order" is ORDER, while CANCEL only covers questions about the cancellation fee. "Where is my parcel" is ORDER, while DELIVERY is about how long delivery takes. There is no category for damaged items, so a damaged item with a request for money back is labelled REFUND.

Results

Macro-F1 (higher is better, 1.0 is perfect):

English (Bitext)	Code-mixed (our test set)
Baseline, TF-IDF + logistic regression 0.974 on a 60-row sample (0.996 on the full held-out split) 0.643 LLM classifier Escalation, on the code-mixed test set of 40 messages:

Escalation rate

Escalation precision (escalated messages that really needed a human)

Escalation recall (wrong or vague predictions that were escalated)

The full per-message table is in the app's Test results tab and in results_codemixed.csv. Numbers come from eval.py, and the raw values are in metrics.json.

What we learned along the way

• English scores are almost meaningless. The baseline reaches 0.996 on Bitext because the data is templated and repetitive. On our code-mixed messages it falls to 0.643. That gap is the real problem this challenge is about.

• The English-only baseline is a bad referee. We first let it override the LLM when the two disagreed. On our development messages it disagreed with the LLM three times on clear messages and was wrong every time, and it added several false escalations without catching a single real error. We turned that rule off and kept the baseline only as a comparison.

• LLM confidence is not very informative. Messages the model understood sat between 0.75 and 0.96, while vague ones sat between 0.15 and 0.30. A threshold separates "understood" from "vague" well, but it cannot tell a right answer from a confidently wrong one. This is why we added the NONE category and the extra rules instead of relying on a threshold alone.

• Prompt edits can move the problem around. Fixing one "where is my parcel" mistake nudged another message across the ORDER and DELIVERY boundary, so we stopped editing the prompt for single examples and froze it before running the final evaluation.

Limitations

We would rather say these ourselves than have you find them:

• The training data is synthetic English, so the system has never been trained on Tanglish or Hinglish. It leans on the LLM for those.

• The code-mixed test set is small (40 messages). Treat the scores as indicative, not precise.

• Urgency is our own rule and is checked only against our own labels.

• The English score uses a random sample of 60 messages to stay within free API limits.

• Categories overlap in the data (for example ORDER versus DELIVERY), so some of our labels are a judgement call.

• Replies are drafts for a human to review. They are not approved customer communication.

• The free API tier has rate limits, so the public demo can slow down under load.

Run it yourself

You need Python 3.10 or newer and a free Groq API key.

git clone https://github.com/Vigasni/code-mixed-triage.git

cd code-mixed-triage

pip install -r requirements.txt

Set your key (PowerShell shown first, then macOS or Linux):

$env:GROQ_API_KEY="your_key_here"

export GROQ_API_KEY="your_key_here"

Start the app:

python -m streamlit run app.py

To rebuild the baseline and re-run the evaluation:

python baseline.py # trains the baseline, writes baseline.joblib and bitext_test.csv

python -u eval.py # runs both test sets, writes results_codemixed.csv and metrics.json

If the baseline fails to load because of a scikit-learn version mismatch, run python baseline.py again on your own machine.

What is in the repository:

File and Purpose

app.py-The Streamlit app (Inbox, Test results, Queue dashboard).

llm.py-Prompt, model call and output validation.

escalate.py-The escalation rules.

data.py-Loads and cleans the Bitext data.

baseline.py-Trains the TF-IDF and logistic regression baseline.

eval.py-Evaluation on the English sample and the code-mixed test set.

tune.py-Compares escalation settings on the development set.

dev_set.csv-Development messages used for tuning.

data/codemixed_test.csv-The code-mixed test set (kept out of prompts and tuning).

results_codemixed.csv, metrics.json-Evaluation outputs shown in the app.

What we would do next

• Pick few-shot examples by similarity to the incoming message instead of using a fixed set.

• Collect real code-mixed messages from volunteers and grow the test set well beyond 40.

• Teach the system the cases it currently mixes up, with a clearer boundary between ORDER and DELIVERY.

• Check whether a smaller, cheaper model holds up on the same test set.

Team Vigasni R and Ananya H.

Acknowledgements Bitext for the customer support dataset, Groq for the free model access, and Streamlit for the app framework
