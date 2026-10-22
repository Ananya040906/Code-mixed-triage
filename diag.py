import os
import time

t0 = time.time()


def p(msg):
	print(f"[{time.time() - t0:5.1f}s] {msg}", flush=True)


p("start")

# 1. Which eval.py is she running?
try:
	src = open("eval.py", encoding="utf-8").read()
	has_progress = 'print(f"code-mixed' in src
	p(f"eval.py has progress prints: {has_progress}  (False = old silent version)")
except Exception as e:
	p(f"cannot read eval.py: {e!r}")

# 2. Imports
import pandas  # noqa: E402
p("pandas imported")
import sklearn  # noqa: E402
p("sklearn imported")
import joblib  # noqa: E402
p("joblib imported")

# 3. Baseline model
try:
	joblib.load("baseline.joblib")
	p("baseline.joblib loaded")
except Exception as e:
	p(f"baseline.joblib FAILED: {e!r}")

# 4. Test file
try:
	df = pandas.read_csv("data/codemixed_test.csv")
	p(f"test csv rows: {len(df)}")
except Exception as e:
	p(f"test csv FAILED: {e!r}")

# 5. Key and one short Groq call with a hard 20 second timeout
key = os.environ.get("GROQ_API_KEY")
p(f"key is set: {bool(key)}")
if key:
	from groq import Groq
	client = Groq(api_key=key, timeout=20, max_retries=0)
	p("calling Groq (20 s timeout)...")
	try:
		r = client.chat.completions.create(
			model="openai/gpt-oss-120b",
			messages=[{"role": "user", "content": "Say hi in one word"}],
		)
		p("Groq replied: " + str(r.choices[0].message.content)[:60])
	except Exception as e:
		p(f"Groq call FAILED: {type(e).__name__}: {str(e)[:200]}")

p("done")
