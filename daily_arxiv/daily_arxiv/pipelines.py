# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

import arxiv


class DailyArxivPipeline:
    def __init__(self):
        self.client = None

    def process_item(self, item: dict, spider):
        # 如果爬虫已直接从页面解析出全部元数据，直接跳过耗时的官方 API 单篇调用
        if item.get("title") and item.get("summary") and item.get("authors"):
            return item

        # 仅在页面字段缺失时回退到官方 API 查询
        if self.client is None:
            self.client = arxiv.Client(page_size=100)

        item["pdf"] = f"https://arxiv.org/pdf/{item['id']}"
        item["abs"] = f"https://arxiv.org/abs/{item['id']}"
        try:
            search = arxiv.Search(id_list=[item["id"]])
            paper = next(self.client.results(search))
            item["authors"] = [a.name for a in paper.authors]
            item["title"] = paper.title
            item["categories"] = paper.categories
            item["comment"] = paper.comment
            item["summary"] = paper.summary
        except Exception as e:
            spider.logger.warning(f"Fallback API query failed for {item.get('id')}: {e}")

        return item
