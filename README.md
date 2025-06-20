# About
This is a forked and modified repo form。daily-arXiv-ai-enhanced

This tool will daily crawl https://arxiv.org and use LLMs to summarize them then show them in github pages.

# What this repo can do 
This repo will daily crawl arXiv papers mainly about **biology** (optional) and use **DeepSeek chat** (optional) to summarize the papers in **Chinese**. (Other language optional)

If you wish to crawl other arXiv categories, use other LLMs or other language, please follow the bellow instructions.
Otherwise, you can directly use this repo. 

### Instructions:
1. Fork/modify this repo to your own account
2. Go to: your-own-repo -> Settings -> Secrets and variables -> Actions
3. Go to Secrets. Secrets are encrypted and are used for sensitive data
   Create two repository secrets named `OPENAI_API_KEY` and `OPENAI_BASE_URL`, and input corresponding values.
4. Go to Variables. Variables are shown as plain text and are used for non-sensitive data. Create the following repository variables:
   1. `CATEGORIES`: separate the categories with ",", such as "q-bio.GN,cs.CV"
   2. `LANGUAGE`: such as "Chinese","English",or other Language you perfer.(comperhensived by AI)
   3. `MODEL_NAME`: such as "deepseek-chat"
   4. `EMAIL`: your email for push to github
   5. `NAME`: your name for push to github
5. Go to your-own-repo -> Actions -> arXiv-daily
6. You can manually click **Run workflow** to test if it works well. By default, this action will automatically run every day(16:30 UTC, Cos using deepseek api will be cheapper during this period). \
Otherwise You can modify it in `.github/workflows/run.yml`
7. You can modify the index.html to change the web as you want.
