# test_openrouter.py
"""
Lightweight connectivity + key check for OpenRouter.

Usage:
1. Put your OpenRouter API key in a .env file in the same folder:
   OPENROUTER_API_KEY=sk-or-v1-...
2. Run:  python test_openrouter.py

This script does NOT print the key, it just verifies that:
- the key is present
- a simple /models call works with that key
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("❌ OPENROUTER_API_KEY not found in environment/.env")
    raise SystemExit(1)

headers = {
    "Authorization": f"Bearer {api_key}",
    "HTTP-Referer": "https://localhost/test",  # or your app URL when deployed
    "X-Title": "Vaccine Pipeline Platform Test",
}

try:
    resp = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=10)
    print(f"✅ OpenRouter reachable. Status: {resp.status_code}")
    if resp.ok:
        data = resp.json()
        model_ids = [m.get("id") for m in data.get("data", [])][:5]
        print(f"✅ Sample models: {model_ids}")
    else:
        print(f"⚠️ OpenRouter error response: {resp.text[:300]}")
except requests.exceptions.ConnectionError:
    print("❌ BLOCKED: Cannot reach openrouter.ai (likely firewall)")
except requests.exceptions.Timeout:
    print("⏱️ TIMEOUT: Network issue or proxy interference")
except Exception as e:
    print(f"❌ ERROR: {e}")