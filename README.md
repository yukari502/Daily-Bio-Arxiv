# Daily arXiv AI Summarizer

## About

This project is a modified by [daily-arXiv-ai-enhanced](https://github.com/Starlento/daily-arXiv-ai-enhanced) tool. It automatically fetches new papers from arXiv.org, uses a Large Language Model (LLM) to summarize them, and publishes the summaries to a GitHub Pages website.

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

-   `OPENAI_API_KEY`: Your API key for the LLM service.
-   `OPENAI_BASE_URL`: The base URL for the LLM service's API endpoint.

#### Repository variables

Create the following variables:

-   `CATEGORIES`: A comma-separated list of arXiv categories to track (e.g., `q-bio.GN,cs.CV`).
-   `LANGUAGE`: The target language for the summaries (e.g., `Chinese`, `English`).
-   `MODEL_NAME`: The identifier for the LLM you want to use (e.g., `deepseek-chat`).
-   `EMAIL`: The email address to use for Git commits.
-   `NAME`: The name to use for Git commits.

### 3. Run the GitHub Action

Navigate to the `Actions` tab in your repository and select the **arXiv-daily** workflow.

-   **Manual Run**: You can click `Run workflow` to manually trigger the process and test your configuration.
-   **Scheduled Run**: By default, the action is scheduled to run automatically at 16:30 UTC every day. You can customize the schedule by editing the cron job in `.github/workflows/run.yml`.
    > **Note**: The default time is set to coincide with typically lower API costs for some services.

---

## Customization

To change the appearance of the web page, you can modify the `index.html` file.
