"""
Web-Frontend für den Academic Research Buddy.

Baut die gleiche Pipeline wie main.py, streamt die Agenten-Ausgabe aber
per Server-Sent-Events (SSE) live in den Browser statt ins Terminal.

Starten mit:  uv run python app.py
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
from haystack import Pipeline, component
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
    """Erzeugt einen Streaming-Callback, der Chunks als Events in die Queue legt
    (gleiche Fälle wie haystacks print_streaming_chunk, nur ohne stdout)."""

    def callback(chunk):
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


@component
class PapersToUserMessage:
    """Zwischen-Komponente zwischen Searcher und Reviewer.

    Der Searcher liefert seinen gesamten Gespraechsverlauf, der mit einer ASSISTANT-
    Nachricht (der Paper-Liste) endet. Gibt man diesen Verlauf direkt an den Reviewer,
    antwortet Gemini oft mit NICHTS - weil die letzte Nachricht schon vom Assistenten
    stammt und es aus Modellsicht 'nichts zu antworten' gibt.

    Diese Komponente nimmt die letzte (Paper-)Nachricht und verpackt sie als USER-
    Nachricht. So bekommt der Reviewer eine klare Aufforderung, auf die er antworten
    kann - und liefert zuverlaessig eine Bewertung.

    Da diese Komponente genau zwischen Suche und Bewertung laeuft, meldet sie ausserdem
    per 'notify'-Callback, dass jetzt der Reviewer startet - so kann die Oberflaeche
    "Bewertung wird erstellt ..." anzeigen, waehrend der Reviewer noch rechnet.
    """

    def __init__(self, notify=None):
        self.notify = notify

    @component.output_types(messages=list[ChatMessage])
    def run(self, messages: list[ChatMessage]):
        if self.notify:
            self.notify()  # Suche fertig -> Reviewer startet gleich
        paper_liste = messages[-1].text if messages else ""
        user_msg = ChatMessage.from_user(
            "Here are the papers found by the research assistant. "
            "Evaluate them as instructed:\n\n" + (paper_liste or "")
        )
        return {"messages": [user_msg]}


def build_pipeline(searcher_callback, reviewer_callback, on_review_start=None) -> Pipeline:
    """Identisch zum Aufbau in main.py, nur mit eigenen Streaming-Callbacks.

    on_review_start: wird aufgerufen, sobald die Suche fertig ist und der Reviewer startet.
    """
    generator_gemini = GoogleGenAIChatGenerator(model="gemini-2.5-flash-lite")
    generator_groq = OpenAIChatGenerator(
        api_key=Secret.from_env_var("GROQ_API_KEY"),
        api_base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
    )
    generator = FallbackChatGenerator([generator_gemini, generator_groq])

    research_agent = create_research_agent(generator, streaming_callback=searcher_callback)
    reviewer_agent = create_reviewer_agent(generator, streaming_callback=reviewer_callback)

    pipeline = Pipeline()
    pipeline.add_component("searcher", research_agent)
    pipeline.add_component("bridge", PapersToUserMessage(notify=on_review_start))  # Paper-Liste -> User-Nachricht
    pipeline.add_component("reviewer", reviewer_agent)
    # Searcher -> bridge -> reviewer, damit der Reviewer eine USER-Nachricht bekommt
    pipeline.connect("searcher.messages", "bridge.messages")
    pipeline.connect("bridge.messages", "reviewer.messages")
    return pipeline


@app.get("/")
def index():
    # no-store -> der Browser cached die Seite nicht, du bekommst immer die aktuelle Version.
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/research")
def research(query: str):
    events: queue.Queue = queue.Queue()

    def run_pipeline():
        """Fuehrt die Such-/Bewertungs-Pipeline aus und streamt die Ausgabe per Events.

        Robustheit gegen leere Reviewer-Antworten: Das kleine Modell (gemini-2.5-flash-lite)
        liefert gelegentlich eine komplett leere Antwort zurueck (kein Fehler, nur kein Text).
        Dann gaebe es keine Bewertung zu sehen. Deshalb pruefen wir nach jedem Lauf, ob der
        Reviewer wirklich Text produziert hat, und starten bei Bedarf bis zu 2x neu.
        """
        try:
            for versuch in range(1, 3):  # bis zu 2 Versuche
                if versuch > 1:
                    # Vor dem neuen Versuch die Anzeige leeren, damit sich die (leere)
                    # erste Runde nicht mit der zweiten stapelt.
                    events.put({"kind": "reset"})

                pipeline = build_pipeline(
                    make_streaming_callback("searcher", events),
                    make_streaming_callback("reviewer", events),
                    # Suche fertig -> Oberflaeche zeigt sofort "Bewertung wird erstellt ..."
                    on_review_start=lambda: events.put({
                        "agent": "reviewer", "kind": "status", "text": "Bewertung wird erstellt …"
                    }),
                )
                message = ChatMessage.from_user(query)
                # include_outputs_from -> wir bekommen die Reviewer-Antwort zurueck und
                # koennen pruefen, ob ueberhaupt eine Bewertung erzeugt wurde.
                result = pipeline.run(
                    data={"searcher": {"messages": [message]}},
                    include_outputs_from={"reviewer"},
                )

                # Hat der Reviewer wirklich Text (= eine Bewertung) geliefert?
                last = result.get("reviewer", {}).get("last_message")
                if last and last.text and last.text.strip():
                    events.put({"kind": "done"})
                    return  # Erfolg -> fertig

            # Beide Versuche kamen leer zurueck -> klare Meldung statt stiller Leere.
            events.put({
                "kind": "error",
                "text": "Der Reviewer hat keine Bewertung geliefert (leere Modell-Antwort). Bitte erneut starten.",
            })
        except Exception as e:
            events.put({"kind": "error", "text": str(e)})

    threading.Thread(target=run_pipeline, daemon=True).start()

    def event_stream():
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
