import os
import json
import re
from groq import Groq

# Change the model name here only. Groq retires models over time.
MODEL = "openai/gpt-oss-120b"

client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=25, max_retries=1)

CATEGORIES = [
    "ACCOUNT", "CANCEL", "CONTACT", "DELIVERY", "FEEDBACK", "INVOICE",
    "ORDER", "PAYMENT", "REFUND", "SHIPPING", "SUBSCRIPTION",
]
# "NONE" = off-topic, greeting, or too vague to classify. Always escalate these.
VALID_CATEGORIES = CATEGORIES + ["NONE"]

SYSTEM = f"""You are a support triage assistant for an Indian e-commerce company.
Customer messages may be English, Tanglish (Tamil + English), Hinglish (Hindi + English), or a mix,
often with creative spelling.

Pick ONE category from: {CATEGORIES}, or "NONE".

Label guide (follow it strictly):
- ACCOUNT: create/delete/edit account, password recovery, login or registration problems, switching account
- ORDER: place, track, change or CANCEL an order. "Where is my order/parcel/package" (tracking) is ORDER. "Cancel my order" is ORDER, not CANCEL
- REFUND: refund policy, asking for money back, refund status. Damaged or wrong item with a refund request is REFUND
- PAYMENT: payment failed or money deducted, which payment methods are accepted
- INVOICE: asking for a bill or invoice copy, checking an invoice
- DELIVERY: only how long delivery takes, delivery options or delivery charges ("how many days will delivery take"). Asking where a specific parcel is belongs to ORDER
- SHIPPING: change or set up the shipping address
- CONTACT: wants to talk to customer service or a human agent
- FEEDBACK: complaints and reviews about service or product
- SUBSCRIPTION: newsletter subscribe/unsubscribe
- CANCEL: ONLY asking about the cancellation fee
- NONE: greeting only, off-topic, gibberish, or too vague to tell what the customer wants

Urgency:
- high: money lost or stuck, damaged/wrong item, account locked, anger, repeated complaint
- medium: delivery delays, order or address changes, needs an answer soon
- low: general questions, invoices, newsletter, feedback

Confidence is a number from 0 to 1 for how sure you are about the category.
If the message is unclear, off-topic, or could fit two categories equally, give confidence below 0.4.
Never guess confidently.

The reply is a short, polite message to the customer, in the same language mix they used.
For NONE, the reply should politely ask them to explain what they need.

Return ONLY a JSON object with exactly these keys:
category, urgency, confidence, reply, reasoning
"reasoning" is one short line explaining the choice."""

# Few-shot examples written by us. Do NOT copy messages from the locked test set or dev set.
FEWSHOT = [
    ("refund eppo varum? 2 vaaram aachu",
     {"category": "REFUND", "urgency": "high", "confidence": 0.93,
      "reply": "Sorry for the wait! Ungal refund status check panrom, 24 hours-la update tharom.",
      "reasoning": "asking when refund will arrive, delayed 2 weeks"}),
    ("login nahi ho raha, password bhool gaya",
     {"category": "ACCOUNT", "urgency": "high", "confidence": 0.92,
      "reply": "Maaf kijiye! Hum aapko password reset link bhej rahe hain, email check kijiye.",
      "reasoning": "cannot log in, forgot password"}),
    ("parcel eppo varum, romba late aagudhu",
     {"category": "DELIVERY", "urgency": "medium", "confidence": 0.85,
      "reply": "Delay-ku sorry! Delivery status check panni sollrom.",
      "reasoning": "asking about delivery time, delayed"}),
    ("bill ki copy chahiye email pe",
     {"category": "INVOICE", "urgency": "low", "confidence": 0.9,
      "reply": "Zaroor! Aapka invoice email par bhej diya jayega.",
      "reasoning": "asking for invoice copy"}),
    ("shipping address maarit, pudhu address update pannunga",
     {"category": "SHIPPING", "urgency": "medium", "confidence": 0.9,
      "reply": "Sure, pudhu address update panrom. Address-a confirm pannunga.",
      "reasoning": "wants to change shipping address"}),
    ("mujhe kisi insaan se baat karni hai",
     {"category": "CONTACT", "urgency": "medium", "confidence": 0.88,
      "reply": "Bilkul, hum aapko customer care agent se connect kar rahe hain.",
      "reasoning": "wants to talk to a human agent"}),
    ("order cancel panna extra charge irukka",
     {"category": "CANCEL", "urgency": "low", "confidence": 0.8,
      "reply": "Cancellation fee details anuppurom. Order stage-a poruthu irukkum.",
      "reasoning": "asking only about the cancellation fee"}),
    ("order cancel karna hai abhi",
     {"category": "ORDER", "urgency": "medium", "confidence": 0.9,
      "reply": "Theek hai, aapka order cancel karne ki process shuru kar rahe hain.",
      "reasoning": "wants to cancel an order, so ORDER not CANCEL"}),
    ("weather enna ippo",
     {"category": "NONE", "urgency": "low", "confidence": 0.1,
      "reply": "Sorry, naanga order support mattum paarkrom. Ungalukku enna help venum?",
      "reasoning": "off-topic, not a support request"}),
    ("ok",
     {"category": "NONE", "urgency": "low", "confidence": 0.1,
      "reply": "Ungalukku enna help venum? Konjam detail-a sollunga.",
      "reasoning": "too vague to classify"}),
]


def _parse_json(text):
    """Parse JSON even if the model wrapped it in code fences or extra text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def classify(message):
    """Returns a dict: category, urgency, confidence, reply, reasoning."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for q, a in FEWSHOT:
        msgs.append({"role": "user", "content": q})
        msgs.append({"role": "assistant", "content": json.dumps(a, ensure_ascii=False)})
    msgs.append({"role": "user", "content": message})

    last_error = "unknown error"
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=msgs,
                temperature=0,
                response_format={"type": "json_object"},
            )
            out = _parse_json(r.choices[0].message.content)
            if out.get("category") in VALID_CATEGORIES and out.get("urgency") in ("low", "medium", "high"):
                out["confidence"] = max(0.0, min(1.0, float(out.get("confidence", 0))))
                out.setdefault("reply", "")
                out.setdefault("reasoning", "")
                if out["category"] == "NONE":
                    out["confidence"] = min(out["confidence"], 0.3)
                return out
            last_error = "invalid category or urgency in model output"
        except Exception as e:
            last_error = str(e)[:150]

    return {"category": "NONE", "urgency": "medium", "confidence": 0.0,
            "reply": "", "reasoning": f"LLM call failed: {last_error}"}


if __name__ == "__main__":
    for m in ["order vandhuchu but box damaged, refund venum", "hi", "mera account login nahi ho raha"]:
        print(m, "->", classify(m))