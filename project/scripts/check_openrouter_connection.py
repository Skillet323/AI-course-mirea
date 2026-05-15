import os
import json
import httpx

api_key = os.environ["OPENROUTER_API_KEY"]
model = os.getenv("OPENROUTER_TASK_MODEL", "inclusionai/ring-2.6-1t:free")

payload = {
    "model": model,
    "messages": [
        {"role": "user", "content": "Return only JSON array: []"}
    ],
    "temperature": 0,
    "max_tokens": 32,
}

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Meeting Secretary",
}

limits = httpx.Limits(max_keepalive_connections=0, max_connections=1)

with httpx.Client(timeout=30.0, http2=False, limits=limits, verify=True) as client:
    r = client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
    print("status:", r.status_code)
    print(r.text[:1000])