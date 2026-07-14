from haystack import Pipeline
import os
from dotenv import load_dotenv
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack.components.generators.chat import OpenAIChatGenerator, FallbackChatGenerator
from haystack.utils import Secret
from agents import create_research_agent, create_reviewer_agent

def main():
    """Run the research workflow from the terminal using the configured agents."""
    load_dotenv()
    
    # Initialize the LLM
    generator = GoogleGenAIChatGenerator(model="gemini-2.5-flash-lite")
    
    # Create the agents
    research_agent = create_research_agent(generator)
    reviewer_agent = create_reviewer_agent(generator)
    
    # Build the pipeline
    pipeline = Pipeline()
    pipeline.add_component("searcher", research_agent)
    pipeline.add_component("reviewer", reviewer_agent)
    
    pipeline.connect("searcher.messages", "reviewer.messages")
    

    # Interactive input in the terminal
    print("=" * 60)
    print(" 📚 ACADEMIC RESEARCH BUDDY ")
    print("=" * 60)
    
    # Wait for the user's research question
    query = input("\n🔎 What would you like to research? (Press enter): ")
    
    # Stop if the user leaves the input empty
    if not query.strip():
        print("❌ Keine Suchanfrage eingegeben. Programm beendet.")
        return

    print(f"\n🚀 Starting the research and review process for your request...\n")
    
    # Run the pipeline with the user's input
    haystack_message = ChatMessage.from_user(query)
    pipeline.run(data={"searcher": {"messages": [haystack_message]}})
    
    print("\n✅ Prozess abgeschlossen.")

if __name__ == "__main__":
    main()