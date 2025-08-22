from openai import OpenAI
from eval.prompt import EVALUATION_PROMPT

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "https://syi8httobjtsxq-8000.proxy.runpod.net/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

eval_prompt = EVALUATION_PROMPT.replace("{{TRAIT}}", "friendly")


completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    max_tokens=1000,
    messages=[{"role": "user", "content": "what's your favorite color bruh?"}],
)
print("Completion result:", completion.choices[0].message.content)
