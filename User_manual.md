# Daily Biology arXiv Updates - Technical Manual

## 1. Overview
The **Daily Biology arXiv Updates** project is an automated, serverless data pipeline and static website designed to fetch, summarize, and present the latest academic papers from arXiv. 

The system operates entirely on GitHub Actions, running a daily cron job that scrapes new papers, utilizes Large Language Models (LLMs) to generate structured summaries, and publishes the results to a statically hosted frontend (GitHub Pages).

## 2. Technology Stack
- **Web Scraping:** [Scrapy](https://scrapy.org/) for HTML parsing, and the official [arXiv Python library](https://github.com/lukasschwab/arxiv.py) for metadata extraction.
- **AI & NLP:** [LangChain](https://www.langchain.com/) (`langchain_openai`) using OpenAI-compatible APIs to process and structure text.
- **Workflow Automation:** GitHub Actions (`run.yml`, `static.yml`).
- **Environment Management:** [uv](https://github.com/astral-sh/uv), an extremely fast Python package installer and resolver.
- **Frontend:** Vanilla HTML, CSS, JavaScript, and [marked.js](https://marked.js.org/) for client-side Markdown rendering.

## 3. Overall Workflow
The entire process is orchestrated by `run.sh` and triggered daily at 16:30 UTC by the `.github/workflows/run.yml` GitHub Action.

1. **Environment Setup:** GitHub Actions checks out the code, installs Python dependencies using `uv`, and exports necessary environment variables (API keys, models, target categories).
2. **Data Scraping (`daily_arxiv`):** A Scrapy spider visits arXiv's new paper lists for specified categories. It identifies new paper IDs and passes them to a pipeline that fetches full metadata (title, authors, summary, etc.) via the arXiv API. Data is saved to `data/{YYYY-MM-DD}.jsonl`.
3. **AI Enhancement (`ai`):** The raw JSONL data is passed to an LLM. The model is prompted to extract and structure the paper's summary into specific fields (TLDR, Motivation, Method, Result, Conclusion). The enhanced data is saved as a new JSONL file.
4. **Markdown Generation (`to_md`):** The AI-enhanced JSONL file is converted into a formatted Markdown document (`data/{YYYY-MM-DD}.md`). Papers are grouped and sorted by categories based on user preferences.
5. **Index Generation:** A Python script parses all generated Markdown files to build a hierarchical JSON index (`index.json` and `index_{year}.json`). This acts as a static API for the frontend.
6. **Commit & Deploy:** The GitHub Action commits the new data files back to the `main` branch. A secondary workflow (`static.yml`) detects the run completion and deploys the updated repository to GitHub Pages.

## 4. Key Scripts & Functions

### 4.1 Orchestration
- **`.github/workflows/run.yml`**: The main CI/CD pipeline. It runs `uv sync` to install dependencies, sets up the environment variables (e.g., `OPENAI_API_KEY`, `CATEGORIES`), executes `run.sh`, and handles the `git commit` and `push` back to the repository.
- **`run.sh`**: The master bash script that sequentially executes the Scrapy spider, the AI enhancement script, the Markdown converter, and the index generator.

### 4.2 Scraping (`daily_arxiv/`)
- **`daily_arxiv/spiders/arxiv.py`**: A Scrapy spider (`ArxivSpider`). 
  - *Function*: Reads `CATEGORIES` from the environment, visits `https://arxiv.org/list/{cat}/new`, and uses CSS selectors to extract the IDs of new papers. It prevents scraping older papers by checking against anchor elements on the page.
- **`daily_arxiv/pipelines.py`**: 
  - *Function*: `DailyArxivPipeline.process_item` receives paper IDs from the spider. It uses the `arxiv` Python library's `Search` client to fetch comprehensive metadata (authors, title, categories, summary, PDF links) robustly without further HTML scraping.

### 4.3 AI Enhancement (`ai/`)
- **`ai/enhance.py`**:
  - *Function*: Connects to the configured LLM using LangChain's `ChatOpenAI`. 
  - It uses the `.with_structured_output(Structure)` method to force the LLM to return data matching a strict Pydantic schema (TLDR, motivation, method, result, conclusion).
  - Contains error handling: If the LLM fails to parse or output the correct structure, it gracefully catches `OutputParserException` and assigns "Error" to the fields to prevent pipeline failure.

### 4.4 Formatting & Indexing
- **`to_md/convert.py`**:
  - *Function*: Reads the AI-enhanced JSONL file and formats it using `to_md/paper_template.md`. It generates a Table of Contents and groups papers by their primary category, sorting categories based on the `CATEGORIES` environment variable priority.
- **`generate_json_index.py`**:
  - *Function*: `generate_split_json_index()` scans the `data/` directory for `YYYY-MM-DD.md` files. It groups them by year and outputs `data/index_{year}.json` files containing the list of available dates. It also generates a master `data/index.json` listing all available years. This splitting prevents the frontend from loading a massive JSON file on startup.

## 5. Website Structure (Frontend)
The website is a static Single Page Application (SPA) designed for fast loading and low maintenance.

- **`index.html`**: Contains the core layout, Dark/Light mode toggle SVGs, and the main JavaScript logic.
  - *State Management*: Uses a simple `currentState` object to track the user's navigation level (`years`, `months`, `days`, `content`).
  - *Dynamic Rendering*: Uses asynchronous `fetch()` calls to load `data/index.json`, `data/index_{year}.json`, and the specific `data/{date}.md` files.
  - *Markdown Parsing*: Uses the externally loaded `marked.min.js` to convert the fetched Markdown content into HTML on the fly.
- **`style.css`**: Provides styling using CSS variables for theming. It implements responsive design and a clean, modern aesthetic.
- **`data/` Directory**: Acts as the backend database. By organizing data into static JSON and MD files, the website requires no active server infrastructure.

## 6. Technical Issues & Maintenance Guide

During the lifecycle of maintaining this website, several technical issues may arise:

### 6.1 Scraping Failures
- **arXiv HTML Changes:** If arXiv updates the DOM structure of `arxiv.org/list/*/new`, the CSS selectors in `daily_arxiv/spiders/arxiv.py` will fail to extract paper IDs. *Fix: Inspect the new arXiv HTML and update the CSS selectors in the spider.*
- **IP Blocking/Rate Limiting:** Scrapy might get temporarily blocked if it hits arXiv too aggressively. *Fix: Adjust Scrapy's `DOWNLOAD_DELAY` in `settings.py` if blocking occurs.*

### 6.2 AI & LLM API Issues
- **API Key Expiration / Insufficient Quota:** The GitHub Action will fail if the OpenAI API key runs out of funds or expires. *Fix: Monitor API usage and update the `OPENAI_API_KEY` secret in GitHub Repository Settings.*
- **Context Length Limitations:** If an arXiv summary is extraordinarily long, it might exceed the LLM's context window. *Fix: Implement truncation in `enhance.py` before sending the prompt to the LLM.*
- **Output Parsing Errors:** LLMs occasionally fail to output valid JSON matching the LangChain schema. The script currently handles this by outputting "Error". *Fix: If errors become frequent, refine the prompts (`template.txt`, `system.txt`) or switch to a more capable model via the `MODEL_NAME` GitHub variable.*

### 6.3 Repository Size Accumulation
- **Git Bloat:** Storing daily `.jsonl` and `.md` files will eventually cause the Git repository size to grow significantly. While text compresses well, years of daily updates might slow down the `actions/checkout` step. *Fix: Periodically archive old data or implement a script to squash old commits.*

### 6.4 Frontend Maintenance
- **CDN Failures:** The frontend relies on `cdn.jsdelivr.net` to load `marked.min.js`. If this CDN goes down, papers will not render. *Fix: Download `marked.min.js` locally and serve it from the repository if higher reliability is needed.*
- **Cache Invalidation:** Users might not see the latest updates if their browser aggressively caches `data/index.json`. *Fix: The current implementation relies on GitHub Pages cache headers, which are generally short-lived, but adding cache-busting query parameters to `fetch()` calls could force fresh loads if users report stale data.*
