import requests
from typing import Optional, Dict, Any, List
from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.tools import Tool
from dotenv import load_dotenv
import os

# --- TOOLS (FUNKTIONEN) ---
# openalex https://developers.openalex.org/api-reference/works
def openalex_article_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Searches OpenAlex for academic papers using full-text search."""

    USER_EMAIL = os.getenv("USER_EMAIL")
    url = f"https://api.openalex.org/works?search={query}&per_page={limit}&mailto={USER_EMAIL}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return [{"error": f"OpenAlex API error: {response.status_code}"}]
        data = response.json()
        results = []

        # filtern der Daten
        for work in data.get("results", []):
            # 1. Autoren extrahieren (als saubere Liste von Namen für Zotero)
            authors = []
            for authorship in work.get("authorships", []):
                author_name = authorship.get("author", {}).get("display_name")
                if author_name:
                    authors.append(author_name)
            
            # 2. Journal-Name extrahieren
            primary_loc = work.get("primary_location") or {}
            source = primary_loc.get("source") or {}
            journal_name = source.get("display_name")
            
            # 3. Abstract aus dem Inverted Index rekonstruieren
            abstract_string = ""
            inverted_index = work.get("abstract_inverted_index")
            if inverted_index:
                # OpenAlex speichert: {"Wort": [Position1, Position2]} -> wir drehen es um zu einer Liste
                word_positions = {}
                for word, positions in inverted_index.items():
                    for pos in positions:
                        word_positions[pos] = word
                # Sortieren nach Position und zusammenfügen
                sorted_words = [word_positions[p] for p in sorted(word_positions.keys())]
                abstract_string = " ".join(sorted_words)

            # Das erweiterte, aber immer noch kompakte Ergebnis anhängen
            results.append({
                "title": work.get("display_name") or work.get("title"), # display_name ist oft sauberer formatiert
                "authors": authors[:5], # maximal 5 Autoren
                "publication_year": work.get("publication_year"),
                "journal": journal_name,
                "doi": work.get("doi"),
                "relevance_score": work.get("relevance_score"),
                "cited_by_count": work.get("cited_by_count"),
                "fwci": work.get("fwci"), # Qualitätsindikator
                "abstract": abstract_string[:1000] if abstract_string else None, # Gekürzt auf ~200 Wörter 
                "pdf_url": primary_loc.get("pdf_url"), # Direkter PDF Link, falls da
                "type": work.get("type")
            })
        return results
    except Exception as e:
        return [{"error": f"Failed to connect to OpenAlex: {str(e)}"}]


# def unpaywall_doi_lookup(doi: str) -> Dict[str, Any]:
#     """Gets Open Access (OA) status and bibliographic info for a specific DOI."""
#     url = f"https://api.unpaywall.org/v2/{doi}?email={USER_EMAIL}"
#     try:
#         response = requests.get(url, timeout=10)
#         if response.status_code != 200:
#             return {"error": f"Unpaywall API error: {response.status_code}"}
#         data = response.json()
#         best_oa_location = data.get("best_oa_location")
#         oa_url = best_oa_location.get("url") if best_oa_location else None
#         return {
#             "doi": data.get("doi"),
#             "title": data.get("title"),
#             "is_oa": data.get("is_oa"),
#             "oa_status": data.get("oa_status"), 
#             "best_oa_url": oa_url,
#             "publisher": data.get("publisher"),
#             "published_date": data.get("published_date")
#         }
#     except Exception as e:
#         return {"error": f"Failed to connect to Unpaywall: {str(e)}"}


# def unpaywall_title_search(query: str, is_oa: Optional[bool] = None, limit: int = 3) -> List[Dict[str, Any]]:
#     """Searches for academic articles by keywords in their title."""
#     url = f"https://api.unpaywall.org/v2/search?query={query}&email={USER_EMAIL}"
#     if is_oa is not None:
#         url += f"&is_oa={str(is_oa).lower()}"
#     try:
#         response = requests.get(url, timeout=10)
#         if response.status_code != 200:
#             return [{"error": f"Unpaywall API error: {response.status_code}"}]
#         data = response.json()
#         results = data.get("results", data) if isinstance(data, dict) else data
        
#         cleaned_results = []
#         for item in results[:limit]:
#             res_data = item.get("response", item)
#             best_oa = res_data.get("best_oa_location")
#             oa_url = best_oa.get("url") if best_oa else None
#             cleaned_results.append({
#                 "title": res_data.get("title"),
#                 "doi": res_data.get("doi"),
#                 "is_oa": res_data.get("is_oa"),
#                 "best_oa_url": oa_url,
#                 "publisher": res_data.get("publisher"),
#                 "year": res_data.get("year")
#             })
#         return cleaned_results
#     except Exception as e:
#         return [{"error": f"Failed to connect to Unpaywall: {str(e)}"}]


# Tools für Haystack erstellen
# 1. OpenAlex Tool Definition
openalex_search_tool = Tool(
    function=openalex_article_search,
    name="openalex_article_search",
    description="Searches OpenAlex for academic papers using full-text search. Returns the most relevant papers.",
    parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The research question or scientific terms to search for."},
                "limit": {"type": "integer", "description": "Maximum number of papers to return (default is 5)."}
            },
            "required": ["query"]
        }
    )

#     # 2. Unpaywall DOI Tool Definition
# unpaywall_doi_tool = Tool(
#     function=unpaywall_doi_lookup,
#     name="unpaywall_doi_lookup",
#     description="Gets Open Access (OA) status and bibliographic info for a specific DOI.",
#     parameters={
#             "type": "object",
#             "properties": {
#                 "doi": {"type": "string", "description": "The DOI string to look up, e.g., '10.1038/nature12373'."}
#             },
#             "required": ["doi"]
#         }
#     )

#     # 3. Unpaywall Title Tool Definition
# unpaywall_search_tool = Tool(
#     function=unpaywall_title_search,
#     name="unpaywall_title_search",
#     description="Searches for academic articles by keywords in their title.",
#     parameters={
#             "type": "object",
#             "properties": {
#                 "query": {"type": "string", "description": "The text keywords to search for in the title."},
#                 "limit": {"type": "integer", "description": "Maximum number of results to return (default is 3)."}
#             },
#             "required": ["query"]
#         }
#     )


# --- Agent---

def create_research_agent(generator) -> Agent:
    """
    Erstellt und konfiguriert den Forschungs-Agenten mit den entsprechenden Tools.
    Erwartet einen initialisierten Haystack LLM-Generator als Übergabe.
    """
    agent = Agent(
        chat_generator=generator,
        system_prompt=(
            "You are an expert academic research assistant. "
            "Your task is to help users find scientific literature, research papers, and check their Open Access availability. "
            "Always use 'openalex_article_search' or 'unpaywall_title_search' to find relevant papers based on the user's topic. "
            "If you find a DOI, you can use 'unpaywall_doi_lookup' to gather more detailed open access links."
        ),
        tools=[openalex_search_tool],
        streaming_callback=print_streaming_chunk,
    )
    return agent