import requests
from typing import Optional, Dict, Any, List
from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.tools import Tool
from dotenv import load_dotenv

load_dotenv()
USER_EMAIL="emma.fantasias@gmail.com"

# --- TOOLS (FUNKTIONEN) ---

def openalex_article_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Searches OpenAlex for academic papers using full-text search."""
    url = f"https://api.openalex.org/works?search={query}&per_page={limit}&mailto={USER_EMAIL}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return [{"error": f"OpenAlex API error: {response.status_code}"}]
        data = response.json()
        results = []

        # filtern der Daten
        for work in data.get("results", []):
            results.append({
                "title": work.get("title"),
                "publication_year": work.get("publication_year"),
                "doi": work.get("doi"),
                "relevance_score": work.get("relevance_score"),
                "cited_by_count": work.get("cited_by_count"),
                "primary_location": work.get("primary_location", {}).get("landing_page_url") if work.get("primary_location") else None,
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