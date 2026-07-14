from haystack.components.agents import Agent
from haystack.components.generators.utils import print_streaming_chunk
from tools import openalex_search_tool, unpaywall_doi_tool


# Create the research agent with a task-specific prompt and the available tools.
def create_research_agent(generator, streaming_callback=print_streaming_chunk) -> Agent:
    """Create an agent that searches for academic papers and collects their metadata."""
    agent = Agent(
        chat_generator=generator,
        system_prompt=(
          "You are an expert academic research assistant.\n"
            "Your task is to identify relevant scientific literature related to the user's topic or research question using 'openalex_article_search'.\n"
            "If a DOI is available from 'openalex_article_search', always use 'unpaywall_doi_lookup' to check for accessible versions of the publication.\n\n"
            "CRITICAL OUTPUT INSTRUCTION:\n"
            "You MUST present the found papers by listing their full details, including Title, Authors, Year, Journal, DOI, and especially the Abstract, Citations, and FWCI metrics if available. "
            "Do not just summarize them. Provide the rich bibliographic data so the next agent in the pipeline can evaluate them."
        ),
        tools=[openalex_search_tool,unpaywall_doi_tool],
        streaming_callback=streaming_callback,
    )
    return agent

# Create the reviewer agent that evaluates the retrieved papers.
def create_reviewer_agent(generator, streaming_callback=print_streaming_chunk) -> Agent:
    """Create an agent that rates the retrieved papers by relevance and quality."""
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
            "1. You MUST evaluate EVERY single paper listed in the chat history.\n"
            "2. DO NOT write any introduction or conclusion. Output ONLY the Markdown table.\n\n"
            "Format the table exactly like this:\n"
            "| Title | Relevance Score (1-10) | Justification (Mention why based on fit & metrics) |\n"
            "| :--- | :--- | :--- |"
        ),
        streaming_callback=streaming_callback,
    )
    return reviewer



