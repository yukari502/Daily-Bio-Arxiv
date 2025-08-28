from pydantic.v1 import BaseModel, Field

class Structure(BaseModel):
    """定义结构化输出的格式"""
    tldr: str = Field(description="A very brief, one-sentence summary of the paper's core contribution (TL;DR).")
    
    motivation: str = Field(description="What problem or motivation does this paper address?")
    method: str = Field(description="What method did the authors propose?")
    result: str = Field(description="What are the experimental results or main findings?")
    conclusion: str = Field(description="The main conclusion of the paper.")
    summary_zh: str = Field(description="Translate the original paper summary into clear, fluent Chinese.")
