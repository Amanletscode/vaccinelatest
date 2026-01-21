# test_openrouter.py
import requests

try:
    response = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
    print(f"✅ OpenRouter accessible: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("❌ BLOCKED: Cannot reach openrouter.ai (likely firewall)")
except requests.exceptions.Timeout:
    print("⏱️ TIMEOUT: Network issue or proxy interference")
except Exception as e:
    print(f"❌ ERROR: {e}")