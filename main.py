import os
from dotenv import load_dotenv
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# 1. .env laden
load_dotenv()

# 2. Prüfen, ob der Key da ist
if not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit("FEHLER: GOOGLE_API_KEY nicht gefunden. Steht er in der .env?")

print(f"✓ GOOGLE_API_KEY geladen ({len(os.getenv('GOOGLE_API_KEY'))} Zeichen)")

# 3. Gemini-Generator (neues SDK) – holt den Key automatisch aus GOOGLE_API_KEY
generator = GoogleGenAIChatGenerator(model="gemini-2.5-flash")

# 4. Mini-Testanfrage
messages = [ChatMessage.from_user("Antworte mit genau einem Wort: Funktioniert das Setup?")]
result = generator.run(messages=messages)

print("✓ Antwort von Gemini:", result["replies"][0].text)
print("\nAlles korrekt aufgesetzt – Haystack spricht mit Gemini.")