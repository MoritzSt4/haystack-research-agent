import requests
from typing import Optional, Dict, Any, List
from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from haystack.tools import Tool
import os
from dotenv import load_dotenv

# --- TOOLS ---
# OpenAlex API reference: https://developers.openalex.org/api-reference/works
def openalex_article_search(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search OpenAlex for academic papers and return a compact metadata summary."""
    OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")
    url = f"https://api.openalex.org/works?search={query}&per_page={limit}&api_key={OPENALEX_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return [{"error": f"OpenAlex API error: {response.status_code}"}]
        data = response.json()
        results = []

        # Filter and normalize the API response
        for work in data.get("results", []):
            # Extract author names as a clean list
            authors = []
            for authorship in work.get("authorships", []):
                author_name = authorship.get("author", {}).get("display_name")
                if author_name:
                    authors.append(author_name)
            
            # Extract the journal name
            primary_loc = work.get("primary_location") or {}
            source = primary_loc.get("source") or {}
            journal_name = source.get("display_name")
            
            # Rebuild the abstract from the inverted index
            abstract_string = ""
            inverted_index = work.get("abstract_inverted_index")
            if inverted_index:
                # OpenAlex stores words as positions; rebuild them into a readable text
                word_positions = {}
                for word, positions in inverted_index.items():
                    for pos in positions:
                        word_positions[pos] = word
                # Sort by position and join the words
                sorted_words = [word_positions[p] for p in sorted(word_positions.keys())]
                abstract_string = " ".join(sorted_words)

            # Add a compact result object with the most useful fields
            results.append({
                "title": work.get("display_name") or work.get("title"), # display_name ist oft sauberer formatiert
                "authors": authors[:5], # maximal 5 Autoren
                "publication_year": work.get("publication_year"),
                "journal": journal_name,
                "doi": work.get("doi"),
                "relevance_score": work.get("relevance_score"),
                "cited_by_count": work.get("cited_by_count"),
                "fwci": work.get("fwci"), # Quality indicator
                "abstract": abstract_string[:1000] if abstract_string else None, # Keep the abstract concise
                "pdf_url": primary_loc.get("pdf_url"), # Direct PDF link if available
                "type": work.get("type")
            })
        return results
    except Exception as e:
        return [{"error": f"Failed to connect to OpenAlex: {str(e)}"}]

# https://unpaywall.org/products/api  https://unpaywall.org/data-format
def unpaywall_doi_lookup(doi: str) -> Dict[str, Any]:
    """Look up a DOI in Unpaywall and return open-access metadata."""
    USER_EMAIL = os.getenv("USER_EMAIL")
    url = f"https://api.unpaywall.org/v2/{doi}?email={USER_EMAIL}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"error": f"Unpaywall API error: {response.status_code}"}
        data = response.json()
        best_oa_location = data.get("best_oa_location")
        oa_url = best_oa_location.get("url") if best_oa_location else None
        return {
            "doi": data.get("doi", doi),
            "title": data.get("title", ""),
            "is_oa": data.get("is_oa", False),
            "oa_status": data.get("oa_status", "unknown"),
            "best_oa_url": oa_url or "",
            "publisher": data.get("publisher", ""),
            "published_date": data.get("published_date", "")
        }

    except Exception as e:
        return {
            "error": f"Failed to connect to Unpaywall: {str(e)}"
        }


# Register the tools for the Haystack agents
openalex_search_tool = Tool(
    function=openalex_article_search,
    name="openalex_article_search",
    description=(
        "Finds academic papers and literature for a given research question or keywords. "
        "Use this tool as the first step when you need to research a scientific topic or look for sources. "
        "Returns rich bibliographic data including title, up to 5 authors, publication year, "
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string", 
                "description": (
                    "The scientific terms, keywords, or full research question to search for. "
    
                )
            },
            "limit": {
                "type": "integer", 
                "description": "Optional. Maximum number of research papers to return. Default is 5. Max recommended is 10."
            }
        },
        "required": ["query"]
    }
)
unpaywall_doi_tool = Tool(
    function=unpaywall_doi_lookup,
    name="unpaywall_doi_lookup",
    description="Gets Open Access (OA) status and bibliographic info for a specific DOI.",
    parameters={
            "type": "object",
            "properties": {
                "doi": {"type": "string", "description": "The DOI string to look up, e.g., '10.1038/nature12373'."}
            },
            "required": ["doi"]
        }
    )


