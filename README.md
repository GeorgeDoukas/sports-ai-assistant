# Sports AI Assistant

Sports AI Assistant is a locally-run assistant that processes sports news and statistics, generates daily summaries, and answers questions using a local LLM.

Features
- Summarize sports news articles and generate daily reports.
- Scrape news and stats from configured sources.
- Store and query embeddings via a vector store.
- Local LLM chat interface for asking questions about processed data.

Prerequisites
- Python 3.10+ (or compatible)
- Install dependencies from `requirements.txt`
- A working local LLM setup (see project-specific LLM config in `.env`)

Quickstart
1. Create and activate a virtual environment:

	python -m venv venv
	venv\Scripts\activate  # Windows

2. Install dependencies:

	pip install -r requirements.txt

3. Copy and edit environment variables:

	- Create a `.env` file in the project root (the CLI can open your editor).
	- Configure LLM settings, storage paths, and any API keys required by scrapers.

Running the CLI
- Launch the interactive CLI:

	python cli.py

The CLI provides options to:
- Configure environment (`Configuration`)
- Run individual modules (DB store, Vector store, scrapers, processing)
- Execute the full "Get News" workflow (scrape -> update vector store -> DB ingest)
- Process articles and generate daily reports
- Open the local LLM chat

Project layout (high level)
- `cli.py` — interactive command-line interface for workflows
- `llm/` — LLM integration, report generation, and processing logic
- `scrapers/` — news and stats scrapers
- `storage/` — DB and vector store management
- `data/` — raw, processed, and generated report data

Configuration
- The project reads configuration from a `.env` file. Typical variables:
  - `LANGUAGE` — default language for processors (e.g., `greek`)
  - LLM model/endpoint settings (project-specific keys)
  - Storage and DB connection settings
