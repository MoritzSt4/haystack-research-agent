from haystack import Pipeline
import os
from dotenv import load_dotenv
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from agents import create_research_agent, create_reviewer_agent

def main():
    load_dotenv()
    
    # LLM initialisieren
    generator = GoogleGenAIChatGenerator(model="gemini-2.5-flash-lite")
    
    # Agenten laden
    research_agent = create_research_agent(generator)
    reviewer_agent = create_reviewer_agent(generator)
    
    # Pipeline
    pipeline = Pipeline()
    pipeline.add_component("searcher", research_agent)
    pipeline.add_component("reviewer", reviewer_agent)
    
    pipeline.connect("searcher.messages", "reviewer.messages")
    

   # --- INTERAKTIVE EINGABE ---
    print("=" * 60)
    print(" 📚 ACADEMIC RESEARCH BUDDY ")
    print("=" * 60)
    
    # Wartet im Terminal auf deine Eingabe
    query = input("\n🔎 Was möchtest du erforschen? (Eingabe drücken): ")
    
    # Falls der User aus Versehen nichts eingegeben hat, abbrechen
    if not query.strip():
        print("❌ Keine Suchanfrage eingegeben. Programm beendet.")
        return

    print(f"\n🚀 Starte Forschungs- und Bewertungsprozess für deine Anfrage...\n")
    
    # 4. Pipeline starten mit der dynamischen Eingabe
    haystack_message = ChatMessage.from_user(query)
    pipeline.run(data={"searcher": {"messages": [haystack_message]}})
    
    print("\n✅ Prozess abgeschlossen.")

if __name__ == "__main__":
    main()