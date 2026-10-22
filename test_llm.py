import os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
r = client.chat.completions.create(
	model="openai/gpt-oss-120b",
	messages=[{"role": "user", "content": "Say hello in Tamil"}],
)
print(r.choices[0].message.content)
