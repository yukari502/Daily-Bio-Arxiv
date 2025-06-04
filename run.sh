today=`date -u "+%Y-%m-%d"` # Current date in UTC


cd daily_arxiv
scrapy crawl arxiv -o ../data/${today}.jsonl

cd ../ai
python enhance.py --data ../data/${today}.jsonl

cd ../to_md
python convert.py --data ../data/${today}_AI_enhanced_${LANGUAGE}.jsonl

cd ..
python update_readme.py

python generate_json_index.py
