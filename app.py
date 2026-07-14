"""
Web frontend for the Academic Research Buddy.

This builds the same pipeline as main.py, but streams agent output live to the browser
through Server-Sent Events (SSE) instead of the terminal.

Run with: uv run python app.py
"""
import json
import queue
import threading
import webbrowser
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from haystack import Pipeline
from haystack.components.generators.chat import OpenAIChatGenerator, FallbackChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret

from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator

from agents import create_research_agent, create_reviewer_agent

load_dotenv()

app = FastAPI(title="Academic Research Buddy")
STATIC_DIR = Path(__file__).parent / "static"

HOST = "127.0.0.1"
PORT = 8000


def make_streaming_callback(agent_name: str, events: queue.Queue):
    """Create a streaming callback that sends chunks as events into the queue."""

    def callback(chunk):
        """Forward agent output chunks to the event queue."""
        if chunk.tool_calls:
            for tool_call in chunk.tool_calls:
                if chunk.start and tool_call.tool_name:
                    events.put({"agent": agent_name, "kind": "tool_call", "text": tool_call.tool_name})
                if tool_call.arguments:
                    events.put({"agent": agent_name, "kind": "tool_args", "text": tool_call.arguments})
        if chunk.tool_call_result:
            events.put({"agent": agent_name, "kind": "tool_result", "text": str(chunk.tool_call_result.result)})
        if chunk.content:
            events.put({"agent": agent_name, "kind": "content", "text": chunk.content})

    return callback


def build_pipeline(searcher_callback, reviewer_callback) -> Pipeline:
    """Build the same pipeline as in main.py, but with custom streaming callbacks."""
    generator_gemini = GoogleGenAIChatGenerator(model="gemini-2.5-flash-lite")


    research_agent = create_research_agent(generator_gemini, streaming_callback=searcher_callback)
    reviewer_agent = create_reviewer_agent(generator_gemini, streaming_callback=reviewer_callback)

    pipeline = Pipeline()
    pipeline.add_component("searcher", research_agent)
    pipeline.add_component("reviewer", reviewer_agent)
    pipeline.connect("searcher.messages", "reviewer.messages")
    return pipeline


@app.get("/")
def index():
    """Return the main HTML page for the web frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/research")
def research(query: str):
    """Start the research workflow and stream its results to the browser."""
    events: queue.Queue = queue.Queue()

    def run_pipeline():
        """Run the pipeline in a background thread and push events to the queue."""
        try:
            pipeline = build_pipeline(
                make_streaming_callback("searcher", events),
                make_streaming_callback("reviewer", events),
            )
            message = ChatMessage.from_user(query)
            pipeline.run(data={"searcher": {"messages": [message]}})
            events.put({"kind": "done"})
        except Exception as e:
            events.put({"kind": "error", "text": str(e)})

    threading.Thread(target=run_pipeline, daemon=True).start()

    def event_stream():
        """Yield SSE events until the pipeline finishes or fails."""
        while True:
            item = events.get()
            yield f"data: {json.dumps(item)}\n\n"
            if item["kind"] in ("done", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{PORT}")).start()
    uvicorn.run(app, host=HOST, port=PORT)
