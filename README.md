# Haystack Research Agent

Ein Research-Agent gebaut mit [Haystack](https://haystack.deepset.ai/) und der [Gemini API](https://aistudio.google.com) von Google.

## Voraussetzungen

- Python 3.10+
- [uv](https://astral.sh/uv) (Package Manager)
- Gemini API Key → kostenlos unter [aistudio.google.com](https://aistudio.google.com)

### uv installieren (Auf MacOS, aber gerne nochmal überprüfen)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Installation

```bash
# Repository klonen
git clone https://github.com/MoritzSt4/haystack-research-agent.git
cd haystack-research-agent

# Abhängigkeiten installieren (erstellt automatisch ein .venv)
uv sync
```

## Konfiguration
Die .env datei anlegen im selben ordner wie die main. Inhalt der .env aus .env.example kopieren.
`.env.example` als `.env` kopieren und den eigenen API Key eintragen:

```bash
cp .env.example .env
```

Dann in `.env`:

```
GOOGLE_API_KEY=dein_api_key_hier
```

Den Key bekommst du kostenlos unter [aistudio.google.com](https://aistudio.google.com) → **Get API key**.

**! Wichtig den API Key niemals ins GitRepo pushen und nur in der .env Datei liegen lassen. (Überprüfe ob die .env Datei im der .gitignore Datei aufgelistet ist, dann wird sie auch nicht ins Reo gepusht. Den API Key im Code dann auch nur über die .env Variable verwenden und nicht direkt eingeben.)**
## Projekt starten

```bash
uv run python main.py
```

## Entwicklung in PyCharm (Das ist die IDE die ich verwende gehen auch andere)

1. Ordner in PyCharm öffnen
2. Interpreter setzen: **Settings → Python Interpreter → Add → Existing → `.venv/bin/python`**
3. PyCharm erkennt alle Pakete automatisch
