import os
import json
import sys
import asyncio
import argparse
import dotenv
from openai import AsyncOpenAI
from structure import Structure

if os.path.exists('.env'):
    dotenv.load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CURRENT_DIR, "system.txt"), "r", encoding="utf-8") as f:
    SYSTEM_TEMPLATE = f.read()
with open(os.path.join(CURRENT_DIR, "template.txt"), "r", encoding="utf-8") as f:
    HUMAN_TEMPLATE = f.read()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--concurrency", type=int, default=5, help="max concurrent requests")
    return parser.parse_args()

async def call_llm_with_fallback(client: AsyncOpenAI, model_name: str, system_prompt: str, user_prompt: str) -> dict:
    """
    尝试使用原生 Structured Outputs (json_schema) 严格解析；
    若特定模型/提供商不支持严格 json_schema，自动回退到 json_object 模式解析。
    """
    try:
        completion = await client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format=Structure,
        )
        parsed = completion.choices[0].message.parsed
        if parsed:
            return parsed.model_dump()
    except Exception as e:
        # 回退到 json_object 模式解析并由 Pydantic 进行校验
        fallback_prompt = (
            system_prompt + "\n\nRespond with valid JSON containing the following keys: "
            "tldr, motivation, method, result, conclusion."
        )
        completion = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": fallback_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        if content:
            raw_dict = json.loads(content)
            return Structure(**raw_dict).model_dump()
        raise e

    raise ValueError("LLM returned empty parsed structure.")

async def process_single_paper(client: AsyncOpenAI, sem: asyncio.Semaphore, model_name: str, language: str, item: dict, idx: int, total: int, max_retries: int = 2) -> dict:
    system_prompt = SYSTEM_TEMPLATE.format(language=language)
    user_prompt = HUMAN_TEMPLATE.format(content=item['summary'])

    async with sem:
        for attempt in range(max_retries + 1):
            try:
                res_dict = await call_llm_with_fallback(client, model_name, system_prompt, user_prompt)
                item['AI'] = res_dict
                print(f"[{idx+1}/{total}] Finished: {item['id']}", file=sys.stderr)
                return item
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"[{idx+1}/{total}] Retrying {item['id']} in {wait_time}s due to error: {e}", file=sys.stderr)
                    await asyncio.sleep(wait_time)
                else:
                    print(f"[{idx+1}/{total}] Error for {item['id']} after {max_retries+1} attempts: {e}", file=sys.stderr)
                    item['AI'] = {
                        "tldr": "Error",
                        "motivation": "Error",
                        "method": "Error",
                        "result": "Error",
                        "conclusion": "Error"
                    }
                    return item

async def main_async():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME")
    language = os.environ.get("LANGUAGE", "English")

    data = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)

    data = unique_data
    total = len(data)

    print('Open:', args.data, file=sys.stderr)
    print('Connect to:', model_name, file=sys.stderr)
    print(f'Total papers: {total}, concurrency: {args.concurrency}', file=sys.stderr)

    output_path = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    # 提前初始化目标文件，防止下游脚本出现 FileNotFoundError
    open(output_path, "w", encoding="utf-8").close()

    default_headers = {
        "HTTP-Referer": "https://github.com/yukari502/Daily-Bio-Arxiv",
        "X-Title": "Daily Bio Arxiv"
    }

    client = AsyncOpenAI(
        base_url=os.environ.get("OPENAI_BASE_URL"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        default_headers=default_headers,
    )

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        process_single_paper(client, sem, model_name, language, item, idx, total)
        for idx, item in enumerate(data)
    ]

    enhanced_data = await asyncio.gather(*tasks)

    with open(output_path, "w", encoding="utf-8") as f:
        for item in enhanced_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Finished all {total} papers. Output saved to {output_path}", file=sys.stderr)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
