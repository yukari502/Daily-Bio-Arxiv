import json
import argparse
import os
from itertools import count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Path to the jsonline file")
    args = parser.parse_args()
    data = []
    preference = os.environ.get('CATEGORIES', 'cs.CV, cs.CL').split(',')
    preference = list(map(lambda x: x.strip(), preference))
    def rank(cate):
        for idx, p in enumerate(preference):
            if cate == p or cate.startswith(p + "."):
                return idx
        return len(preference)

    if not os.path.exists(args.data):
        print(f"Warning: Data file {args.data} does not exist. Skipping conversion.", file=sys.stderr)
        exit(0)

    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    def is_preferred(primary_cat):
        for p in preference:
            if primary_cat == p or primary_cat.startswith(p + "."):
                return True
        return False

    # Filter out cross-listed papers: only keep papers whose PRIMARY category matches our preference
    data = [item for item in data if is_preferred(item["categories"][0])]

    categories = set([item["categories"][0] for item in data])
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template = open(os.path.join(current_dir, "paper_template.md"), "r", encoding="utf-8").read()
    categories = sorted(categories, key=rank)
    cnt = {cate: 0 for cate in categories}
    for item in data:
        cnt[item["categories"][0]] += 1

    markdown = f"<div id=toc></div>\n\n# Table of Contents\n\n"
    for idx, cate in enumerate(categories):
        markdown += f"- [{cate}](#{cate}) [Total: {cnt[cate]}]\n"

    idx = count(1)
    for cate in categories:
        markdown += f"\n\n<div id='{cate}'></div>\n\n"
        markdown += f"# {cate} [[Back]](#toc)\n\n"
        markdown += "\n\n".join(
            [
                template.format(
                    title=item["title"],
                    authors=",".join(item["authors"]),
                    summary=item["summary"],
                    url=item['abs'],
                    tldr=item['AI']['tldr'],
                    motivation=item['AI']['motivation'],
                    method=item['AI']['method'],
                    result=item['AI']['result'],
                    conclusion=item['AI']['conclusion'],
                    cate=item['categories'][0],
                    idx=next(idx)
                )
                for item in data if item["categories"][0] == cate
            ]
        )
    md_filename = (args.data.split('_AI_enhanced_')[0] if '_AI_enhanced_' in args.data else args.data.rsplit('.', 1)[0]) + '.md'
    with open(md_filename, "w", encoding="utf-8") as f:
        f.write(markdown)
