# structure.py 示例
from pydantic import BaseModel, Field
from typing import Optional

class Structure(BaseModel):
    tldr: str = Field(description="文章集合的精简总结")
    motivation: str = Field(description="文章集合背后的核心动机或要解决的问题")
    method: str = Field(description="文章集合中描述的主要方法或技术")
    result: str = Field(description="文章集合中主要研究成果或发现")
    conclusion: str = Field(description="文章集合的整体结论或影响")

    # 针对类别总结，可能需要调整或添加字段，例如：
    # category_overall_summary: str = Field(description="该类别的整体概述")
    # key_themes: List[str] = Field(description="该类别中反复出现的主要主题")