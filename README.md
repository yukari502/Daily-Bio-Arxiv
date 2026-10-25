# Daily Biology arXiv Updates

> 🌐 **Live Website**: [https://yukari502.github.io/Daily-Bio-Arxiv/](https://yukari502.github.io/Daily-Bio-Arxiv/)

## About

This project is modified from the daily-arXiv-ai-enhanced tool. It automatically fetches new papers from [arXiv.org](https://arxiv.org/) (primarily in quantitative biology, genomics, and related AI/CS categories), uses Large Language Models (LLMs) to generate concise structured summaries, and publishes the daily reports to an interactive static website hosted on **GitHub Pages**.

---

## 🌐 Online Website Overview

Visit the live website: **[https://yukari502.github.io/Daily-Bio-Arxiv/](https://yukari502.github.io/Daily-Bio-Arxiv/)**

The frontend is a lightweight, responsive Single Page Application (SPA) providing an effortless literature reading experience:

- **⚡ Quick Access Links**: Jump straight into the latest paper releases from recent days with a single click.
- **📚 Multi-Year Archive Navigation**: Easily explore historical daily updates grouped by **Year $\rightarrow$ Month $\rightarrow$ Day** through a fast, split-index catalog.
- **🧠 5-Point AI Structured Summaries**: Every paper summary is distilled into 5 high-signal sections:
  - **TL;DR**: One-sentence core takeaway.
  - **Motivation**: Why the research was conducted.
  - **Method**: The technical approach or experimental methodology.
  - **Result**: Key discoveries and quantitative benchmarks.
  - **Conclusion**: Main implications and future directions.
- **🌓 Dark / Light Mode**: Seamless theme switching with system-preference detection and local persistence.
- **📄 Direct Paper & PDF Links**: Quick access to original arXiv abstracts (`/abs/...`) and official PDF files (`/pdf/...`).
- **📱 Fully Responsive**: Clean layout optimized for mobile phones, tablets, and desktop browsers.

---

## Features

-   **Daily Updates**: Automatically crawls arXiv daily for new papers in your chosen categories (e.g., biology, computer science).
-   **AI-Powered Summaries**: Utilizes an LLM of your choice (e.g., DeepSeek Chat) to generate concise summaries.
-   **Customizable**: Easily configure the arXiv categories, LLM, and summary language to fit your needs.
-   **Automated Deployment**: Publishes the summaries to a clean web interface hosted on GitHub Pages.

---

## Getting Started

### 1. Fork the Repository

First, fork this repository to your own GitHub account.

### 2. Configure Secrets and Variables

Navigate to `Settings` > `Secrets and variables` > `Actions` in your forked repository.

#### Repository secrets

Create the following encrypted secrets:

-   `OPENAI_API_KEY`: Your API key for the LLM service (e.g., `sk-or-v1-...` for OpenRouter, `sk-...` for DeepSeek/OpenAI).
-   `OPENAI_BASE_URL`: The base URL for the LLM service's API endpoint.
    > [!IMPORTANT]
    > **Base URL Format Rules**: Do **not** append `/chat/completions` to the end of the base URL, as the OpenAI SDK automatically appends it.
    > - **OpenRouter**: `https://openrouter.ai/api/v1` *(must include `/v1`)*
    > - **DeepSeek Official**: `https://api.deepseek.com`
    > - **OpenAI Official**: `https://api.openai.com/v1`
    > - **SiliconFlow**: `https://api.siliconflow.cn/v1`

#### Repository variables

Create the following variables:

-   `CATEGORIES`: A comma-separated list of arXiv categories to track (e.g., `q-bio.GN,cs.CV`).
-   `LANGUAGE`: The target language for the summaries (e.g., `Chinese`, `English`).
-   `MODEL_NAME`: The model identifier for the LLM service.
-   `EMAIL`: The email address to use for Git commits.
-   `NAME`: The name to use for Git commits.

### 3. Run the GitHub Action

Navigate to the `Actions` tab in your repository and select the **arXiv-daily** workflow.

-   **Manual Run**: You can click `Run workflow` to manually trigger the process and test your configuration.
-   **Scheduled Run**: By default, the action is scheduled to run automatically at 16:30 UTC every day. You can customize the schedule by editing the cron job in `.github/workflows/run.yml`.
    > **Note**: The default time is set to coincide with typically lower API costs for some services.

---

## Customization & Maintenance

- **Appearance**: To change the appearance of the web page, modify the `index.html` and `style.css` files.
- **Maintenance**: For a detailed technical overview of the project's architecture, core scripts, automation workflow, and troubleshooting guidelines, please refer to the [`User_manual.md`](User_manual.md) file.

---

## Buy Me a Coffee

If you find this project helpful, consider supporting it by buying me a coffee! ☕

<img src="./img/qr-code.png" alt="Buy Me a Coffee QR Code" width="300" />
