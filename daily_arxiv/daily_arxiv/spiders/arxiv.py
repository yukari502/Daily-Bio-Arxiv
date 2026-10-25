import os
import re
import scrapy


class ArxivSpider(scrapy.Spider):
    name = "arxiv"  # 爬虫名称
    allowed_domains = ["arxiv.org"]  # 允许爬取的域名

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        categories = os.environ.get("CATEGORIES", "q-bio.GN")
        cat_list = [c.strip() for c in categories.split(",") if c.strip()]
        self.start_urls = [
            f"https://arxiv.org/list/{cat}/new" for cat in cat_list
        ]

    def parse(self, response):
        anchors = []
        for li in response.css("div[id=dlpage] ul li"):
            href = li.css("a::attr(href)").get()
            if href and "item" in href:
                try:
                    anchors.append(int(href.split("item")[-1]))
                except ValueError:
                    pass

        dts = response.css("div[id=dlpage] dl dt")
        dds = response.css("div[id=dlpage] dl dd")

        for dt, dd in zip(dts, dds):
            item_name = dt.css("a[name^='item']::attr(name)").get()
            if item_name and anchors:
                try:
                    item_num = int(item_name.split("item")[-1])
                    if item_num >= anchors[-1]:
                        continue
                except ValueError:
                    pass

            abs_href = dt.css("a[title='Abstract']::attr(href)").get()
            if not abs_href:
                continue
            paper_id = abs_href.split("/")[-1]

            # 直接从页面 HTML 提取全部元数据，无需单篇调用官方慢速 API
            title_raw = "".join(dd.css("div.list-title ::text").getall()).replace("Title:", "").strip()
            title = re.sub(r"\s+", " ", title_raw)

            authors = [a.strip() for a in dd.css("div.list-authors a::text").getall() if a.strip()]

            comments_raw = "".join(dd.css("div.list-comments ::text").getall()).replace("Comments:", "").strip()
            comment = re.sub(r"\s+", " ", comments_raw) if comments_raw else None

            subjects_text = "".join(dd.css("div.list-subjects ::text").getall())
            categories = re.findall(r"\(([a-zA-Z\-]+(?:\.[a-zA-Z\-]+)?)\)", subjects_text)

            summary_raw = "".join(dd.css("p.mathjax ::text").getall()).strip()
            summary = re.sub(r"\s+", " ", summary_raw)

            yield {
                "id": paper_id,
                "pdf": f"https://arxiv.org/pdf/{paper_id}",
                "abs": f"https://arxiv.org/abs/{paper_id}",
                "authors": authors,
                "title": title,
                "categories": categories,
                "comment": comment,
                "summary": summary,
            }
