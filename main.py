from haystack import Pipeline
import os
from dotenv import load_dotenv
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.components.generators.chat import OpenAIChatGenerator, FallbackChatGenerator
from haystack.utils import Secret
from agents import create_research_agent, create_reviewer_agent
from coordinator import run_coordinator

def main():
    load_dotenv()
    
    # LLM initialisieren
    generator_gemini = GoogleGenAIChatGenerator(model="gemini-2.5-flash-lite")
    generator_groq = OpenAIChatGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url= "https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile"
    )
    
    generators = [generator_gemini, generator_groq]
    generator = FallbackChatGenerator(generators)
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
    
    # 4. Koordinator starten: sucht + bewertet mehrfach und liefert die besten Paper
    run_coordinator(pipeline, query)

    print("\n✅ Prozess abgeschlossen.")

if __name__ == "__main__":
    main()