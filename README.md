# OllamaSearch

OllamaSearch is a Python project for comparing two approaches to search-enabled LLM workflows:

1. **Native/model-provided tools**
2. **Custom web search function powered by a locally deployed SearXNG engine**

The goal is to evaluate trade-offs in **answer quality, controllability, privacy, and security** when integrating web search with Ollama-hosted models.

---

## Why this project?

Many LLM toolchains rely on external APIs for search. This project explores a more controlled approach by routing search through **self-hosted SearXNG**, allowing you to:

- keep search infrastructure in your environment
- reduce third-party data exposure
- tune search behavior for your use case
- compare native vs custom integration patterns side by side

---

## Features

- Python-based experimentation setup
- Ollama model integration
- Native tool-calling comparison flow
- Custom search function using SearXNG
- Local-first architecture for better privacy and control
- Foundation for benchmarking and evaluation

---

## Architecture (high level)

```text
User Query
   ├─> Flow A: Native tool path (model/tool defaults)
   └─> Flow B: Custom function path
            └─> Local SearXNG instance
                    └─> Aggregated web results

Both flows -> Response generation -> Comparison/evaluation
```

---

## Tech Stack

- **Language:** Python
- **Model Runtime:** Ollama
- **Search Engine:** SearXNG (self-hosted/local)

---

## Prerequisites

Before running this project, ensure you have:

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- At least one Ollama model pulled locally (for example, `llama3.1`)
- A running SearXNG instance (local Docker deployment recommended)

---

## Getting Started

### 1) Clone the repository

```bash
git clone https://github.com/nurasik14/OllamaSearch.git
cd OllamaSearch
```

### 2) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows PowerShell
```

### 3) Install dependencies

If a requirements file is present:

```bash
pip install -r requirements.txt
```

If not, install your core packages manually as needed.

### 4) Configure environment variables

Create a `.env` file in the project root (example):

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
SEARXNG_BASE_URL=http://localhost:8080
SEARCH_TIMEOUT=20
```

> Adjust URLs/ports to match your local setup.

---

## Running the Project

Use your main script or module entry point. Typical examples:

```bash
python main.py
# or
python -m ollamasearch
```

If your repo includes dedicated scripts for each flow, run them separately for clear comparisons:

```bash
python native_tools_flow.py
python custom_search_flow.py
```

---

## What to Compare

When evaluating native vs custom search flow, track:

- **Relevance:** Are returned answers on-topic?
- **Grounding:** Are claims supported by retrieved data?
- **Latency:** End-to-end response time
- **Stability:** Repeatability across multiple runs
- **Control:** Ability to tune search sources and behavior
- **Privacy/Security:** External data exposure vs local handling

---

## Security & Privacy Notes

Using local SearXNG helps improve operational control, but still review:

- outbound network policies from your host
- SearXNG engine/source configuration
- logging/redaction practices for prompts and results
- model/tool traces saved during experimentation

For sensitive environments, prefer isolated runtime and strict firewall rules.

---

## Suggested Project Structure

```text
OllamaSearch/
├── README.md
├── requirements.txt
├── .env.example
├── main.py
├── native_tools_flow.py
├── custom_search_flow.py
├── search/
│   ├── searxng_client.py
│   └── adapters.py
├── evaluation/
│   ├── benchmark.py
│   └── metrics.py
└── tests/
```

---

## Roadmap

- [ ] Add reproducible benchmark dataset
- [ ] Add automated metrics (accuracy, latency, citation quality)
- [ ] Add structured logging and run reports
- [ ] Add Docker Compose for full local stack (Ollama + app + SearXNG)
- [ ] Add CI tests for search adapters and response parsing

---

## Contributing

Contributions are welcome. If you want to propose improvements:

1. Fork the repo
2. Create a feature branch
3. Commit your changes
4. Open a pull request with a clear description

