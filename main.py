import os
from dotenv import load_dotenv

# Haystack Imports
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

# Importiere die Agenten-Fabrik aus deiner externen Datei
from search_agent import create_research_agent

# 1. Globale Konfiguration & Umgebung laden
load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise SystemExit("FEHLER: GOOGLE_API_KEY nicht gefunden. Steht er in der .env?")


def main():
    print(f"✓ GOOGLE_API_KEY geladen ({len(os.getenv('GOOGLE_API_KEY'))} Zeichen)")
    
    # 2. Initialisiere den Generator
    generator = GoogleGenAIChatGenerator(model="gemini-2.5-flash-lite")

    # 3. Mini-Testanfrage (Setup-Check)
    print("Teste Verbindung zu Gemini...")
    messages = [ChatMessage.from_user("Antworte mit genau einem Wort: Funktioniert das Setup?")]
    result = generator.run(messages=messages)
    print("✓ Antwort von Gemini:", result["replies"][0].text)
    print("Alles korrekt aufgesetzt – Haystack spricht mit Gemini.\n")

    # 4. Agenten aus der externen Datei erstellen lassen
    print("Initialisiere Forschungs-Agenten aus externem File...")
    tool_calling_agent = create_research_agent(generator)

    # 5. Interaktive Test-Schleife im Terminal
    print("\n=== Wissenschafts-Agent Testmodus ===")
    print("Tippe 'exit' zum Beenden.\n")

    while True:
        user_input = input("Deine Frage an den Agenten: ")
        if user_input.lower() == 'exit':
            print("Testmodus beendet.")
            break
            
        if not user_input.strip():
            continue

        print("\n[Agent denkt nach & streamt Antwort...]")
        try:
            agent_result = tool_calling_agent.run(
                messages=[ChatMessage.from_user(user_input)],
            )
            print("\n\n--- FINALE ANTWORT ---")
            print(agent_result["last_message"].text)
            print("-----------------------\n")
        except Exception as e:
            print(f"\n[FEHLER] Da ist etwas schiefgelaufen: {e}\n")


if __name__ == "__main__":
    main()