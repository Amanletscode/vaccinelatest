import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Test 1: Check if package is installed
print("=" * 50)
print("TEST 1: Checking if google-generativeai is installed...")
try:
    import google.generativeai as genai
    print("✅ google-generativeai is installed!")
except ImportError:
    print("❌ google-generativeai is NOT installed!")
    print("   Run: pip install google-generativeai")
    exit(1)

# Test 2: Check if API key is set
print("\n" + "=" * 50)
print("TEST 2: Checking for GEMINI_API_KEY...")
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"✅ GEMINI_API_KEY found! (starts with: {api_key[:10]}...)")
else:
    print("❌ GEMINI_API_KEY not found in .env file!")
    print("   Make sure you have a .env file with GEMINI_API_KEY=your_key")
    exit(1)

# Test 3: Try to connect to Gemini API
print("\n" + "=" * 50)
print("TEST 3: Testing API connection...")
try:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('models/gemini-robotics-er-1.5-preview')
    response = model.generate_content("Say 'Hello, the API is working!' in exactly those words.")
    print(f"✅ API Response: {response.text}")
except Exception as e:
    print(f"❌ API Error: {e}")
    print("\n   Possible issues:")
    print("   - Invalid API key")
    print("   - Network/firewall blocking the request")
    print("   - Company proxy blocking Google APIs")
    exit(1)

# Test 4: Check network connectivity
print("\n" + "=" * 50)
print("TEST 4: Testing network connectivity to Google...")
try:
    import requests
    response = requests.get("https://generativelanguage.googleapis.com", timeout=10)
    print(f"✅ Can reach Google AI endpoint (Status: {response.status_code})")
except requests.exceptions.RequestException as e:
    print(f"❌ Network error: {e}")
    print("   Your company firewall might be blocking Google APIs")

print("\n" + "=" * 50)
print("🎉 ALL TESTS PASSED! You should be able to run the full app.")
print("=" * 50)