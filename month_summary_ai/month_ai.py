
import os
import json
import sys
import datetime # 导入 datetime 模块用于日期处理

import dotenv
import argparse # 虽然大部分参数被自动化，但保留 argparse 便于未来扩展或调试

import langchain_core.exceptions
from langchain_google_generativeai import ChatGoogleGenerativeAI
from langchain.prompts import (
  ChatPromptTemplate,
  SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
)
from structure import Structure # 确保您的 structure.py 文件存在并定义了 Structure 类

# 加载环境变量 (例如 GOOGLE_API_KEY, MODEL_NAME, LANGUAGE)
if os.path.exists('.env'):
    dotenv.load_dotenv()

# 从文件中读取系统提示和用户提示模板
# 请确保 template.txt 和 system.txt 位于脚本的同一目录下
template = open("template.txt", "r", encoding="utf-8").read()
system = open("system.txt", "r", encoding="utf-8").read()

def get_current_month_file_paths(language: str):
    """
    根据当前月份和语言生成输入和输出文件的完整路径。
    文件路径格式为：上级目录/month/yyyy-mm.json (输入)
    和 上级目录/month/yyyy-mm_AI_language.json (输出)
    """
    current_date = datetime.datetime.now()
    # 格式化为 yyyy-mm
    month_str = current_date.strftime("%Y-%m")
    
    # 构建月文件夹路径：假设脚本在某个子目录，要访问上级目录的 month 文件夹
    # 获取当前脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建到上级目录的 'month' 文件夹的路径
    month_folder_path = os.path.join(script_dir, '..', 'month')
    # 确保 month 文件夹存在
    os.makedirs(month_folder_path, exist_ok=True)

    input_filename = f"{month_str}.json"
    output_filename = f"{month_str}_AI_{language}.json"

    input_filepath = os.path.join(month_folder_path, input_filename)
    output_filepath = os.path.join(month_folder_path, output_filename)
    
    return input_filepath, output_filepath

def main():
    # --- 修改点 1：移除命令行参数解析，改为自动获取文件路径 ---
    # args = parse_args() # 不再需要此行
    
    model_name = os.environ.get("MODEL_NAME")
    language = os.environ.get("LANGUAGE")

    if not model_name:
        print("错误：环境变量 'MODEL_NAME' 未设置。", file=sys.stderr)
        sys.exit(1)
    if not language:
        print("错误：环境变量 'LANGUAGE' 未设置。", file=sys.stderr)
        sys.exit(1)

    # 获取当前月份的输入和输出文件路径
    input_filepath, output_filepath = get_current_month_file_paths(language)

    # --- 修改点 2：读取 JSON 文件而非 JSONL 文件，并处理新的数据结构 ---
    try:
        with open(input_filepath, "r", encoding="utf-8") as f:
            # 输入文件现在是一个完整的 JSON 对象，键是类别，值是文章列表
            data_by_category = json.load(f)
        print(f'成功读取输入文件: {input_filepath}', file=sys.stderr)
    except FileNotFoundError:
        print(f"错误：未找到输入文件: {input_filepath}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"错误：解码 JSON 文件失败: {input_filepath} - {e}", file=sys.stderr)
        sys.exit(1)

    # 实例化 ChatGoogleGenerativeAI 并应用结构化输出
    # 确保 GOOGLE_API_KEY 环境变量已设置
    llm = ChatGoogleGenerativeAI(model=model_name).with_structured_output(Structure)
    print(f'已连接到模型: {model_name}', file=sys.stderr)

    # --- 修改点 3：更新提示模板以适应类别总结 ---
    # `system` 和 `template` 文件的内容需要适配新的任务
    # system.txt 建议内容: "你是一个专业的学术文章总结助手，擅长从多个文章中提取核心信息并进行结构化总结。"
    # template.txt 建议内容: "请总结以下【{category_name}】类别的所有文章内容。请确保你的总结是客观、全面，并按照指定的结构化格式输出。所有文章的详细内容如下：\n{all_articles_content}"
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system),
        HumanMessagePromptTemplate.from_template(template=template)
    ])

    chain = prompt_template | llm

    all_category_summaries = {} # 用于存储所有类别的总结结果
    
    # --- 修改点 4：遍历类别并为每个类别生成总结 ---
    total_categories = len(data_by_category)
    for idx, (category_name, articles_list) in enumerate(data_by_category.items()):
        print(f"正在处理类别: {category_name} ({idx+1}/{total_categories})", file=sys.stderr)
        
        # 将当前类别下所有文章的关键信息拼接起来作为模型的输入
        all_articles_content = ""
        for article in articles_list:
            # 确保 article 字典包含所有预期字段，否则跳过或处理
            tldr = article.get('tldr', '')
            motivation = article.get('motivation', '')
            result = article.get('result', '')
            conclusion = article.get('conclusion', '')
            
            # 拼接每篇文章的信息，用清晰的分隔符
            all_articles_content += (
                f"文章 ID: {article.get('id', '未知ID')}\n"
                f"TLDR: {tldr}\n"
                f"Motivation: {motivation}\n"
                f"Result: {result}\n"
                f"Conclusion: {conclusion}\n"
                "--------------------\n" # 文章间的分隔符
            )
        
        try:
            # 调用链来生成总结
            response: Structure = chain.invoke({
                "language": language, # 传递语言参数
                "category_name": category_name, # 传递类别名称
                "all_articles_content": all_articles_content # 传递拼接后的所有文章内容
            })
            # 将结构化输出转换为字典并存储
            all_category_summaries[category_name] = response.model_dump()
        except langchain_core.exceptions.OutputParserException as e:
            # 处理模型输出解析错误
            print(f"类别 {category_name} 出现错误: {e}", file=sys.stderr)
            all_category_summaries[category_name] = {
                 "tldr": "Error",
                 "motivation": "Error",
                 "method": "Error",
                 "result": "Error",
                 "conclusion": "Error",
                 "error_message": str(e) # 记录错误信息
            }
        except Exception as e:
            # 捕获其他可能的错误，例如 API 错误
            print(f"类别 {category_name} 发生未知错误: {e}", file=sys.stderr)
            all_category_summaries[category_name] = {
                 "tldr": "Error",
                 "motivation": "Error",
                 "method": "Error",
                 "result": "Error",
                 "conclusion": "Error",
                 "error_message": f"General Error: {str(e)}"
            }
        
        # 每次处理一个类别后，暂时不写入文件，待所有处理完毕后再统一写入
        print(f"完成类别 {category_name}", file=sys.stderr)
    
    # --- 修改点 5：所有类别处理完毕后，统一写入输出文件 ---
    try:
        with open(output_filepath, "w", encoding="utf-8") as f:
            json.dump(all_category_summaries, f, indent=2, ensure_ascii=False)
        print(f"\n所有类别总结已成功保存到: {output_filepath}", file=sys.stderr)
    except IOError as e:
        print(f"错误：无法写入输出文件: {output_filepath} - {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

