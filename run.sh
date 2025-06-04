today=`date -u "+%Y-%m-%d"` # Current date in UTC
cd daily_arxiv
scrapy crawl arxiv -o ../NC_updates/${today}.jsonl

cd ../ai
python enhance.py --data ../NC_updates/${today}.jsonl

cd ../to_md
python convert.py --data ../NC_updates/${today}_AI_enhanced_${LANGUAGE}.jsonl

cd ..
python update_readme.py
