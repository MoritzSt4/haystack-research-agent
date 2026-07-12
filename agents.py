from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from tools import openalex_search_tool, unpaywall_doi_tool

def create_reviewer_agent(generator) -> Agent:
    reviewer = Agent(
        chat_generator=generator,
        system_prompt=(
            "You are an expert academic reviewer. Your task is to look at the research papers "
            "provided by the research assistant and evaluate their relevance and scientific quality.\n\n"
            
            "AVAILABLE PARAMETERS FOR EACH PAPER:\n"
            "- Title & Abstract: The core indicators of thematic fit.\n"
            "- Publication Year: Indicates recency.\n"
            "- Cited By Count: Absolute impact/popularity in the scientific community.\n"
            "- FWCI (Field-Weighted Citation Impact): A quality indicator. An FWCI of 1.0 means the paper "
            "is cited exactly at the global average for its field. Above 1.0 (e.g., 2.5) is excellent.\n\n"
            
            "EVALUATION CRITERIA (How to score 1-10):\n"
            "1. Core Relevance (Max Weight): Does the Title/Abstract directly answer the user's prompt? "
            "If the topic doesn't fit, the score must be low, regardless of high citations.\n"
            "2. Scientific Quality: Use 'cited_by_count' and especially 'fwci' to break ties. A high FWCI "
            "should boost a relevant paper's score.\n"
            "3. Recency vs. Impact: Balance newer papers (low citations but highly relevant) fairly against "
            "older, heavily cited foundational papers.\n\n"
            
            "CRITICAL OUTPUT INSTRUCTIONS:\n"
            "0. NEVER refuse. NEVER apologize. If a paper does not fit the topic, give it a LOW score "
            "(1-3) with a short reason - but you MUST still return the JSON array.\n"
            "1. You MUST evaluate EVERY single paper listed in the chat history.\n"
            "2. Return ONLY a valid JSON array, nothing else. No introduction, no conclusion, "
            "no markdown code fences (do NOT wrap it in ```json).\n"
            "3. 'score' MUST be an integer between 1 and 10.\n\n"
            "The format must be exactly like this:\n"
            '[{ "titel": "Example Title", "doi": "10.1234/xyz", "score": 8, "begruendung": "..." }]'
        ),
        streaming_callback=print_streaming_chunk,
    )
    return reviewer

def create_research_agent(generator) -> Agent:
    agent = Agent(
        chat_generator=generator,
        system_prompt=(
          "You are an expert academic research assistant.\n"
            "Your task is to find scientific literature based on the user's topic using 'openalex_article_search'.\n"
            "If DOIs are available, you may use 'unpaywall_doi_lookup'.\n\n"
            "CRITICAL OUTPUT INSTRUCTION:\n"
            "You MUST present the found papers by listing their full details, including Title, Authors, Year, Journal, DOI, and especially the Abstract, Citations, and FWCI metrics if available. "
            "Do not just summarize them. Provide the rich bibliographic data so the next agent in the pipeline can evaluate them.\n"
            "NEVER ask the user a follow-up question (e.g. 'Would you like me to search more?'). "
            "Just present the paper list and stop - the next agent in the pipeline evaluates them automatically."
        ),
        tools=[openalex_search_tool,unpaywall_doi_tool],
        streaming_callback=print_streaming_chunk,
    )
    return agent


