from typing import List, Literal, Tuple

class TextSimilarity:
    """计算文本相似度"""

    def __init__(self, method: Literal["levenshtein", "jaccard", "simhash"]) -> None: ...
    def calculate(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度"""
        ...
    def batch_calculate(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """计算多个字符串对的相似度"""
        ...

    def calculate_simple(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度(不通过jieba分词)"""
        ...

    def batch_calculate_simple(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """计算多个字符串对的相似度(不通过jieba分词)"""
        ...
