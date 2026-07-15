"""可插拔知识库检索器，默认使用 jieba 分词和 BM25。"""

import json
import math
import os
from abc import ABC, abstractmethod
from collections import Counter
from typing import Dict, List

try:
    import jieba

    def _tokenize(text: str) -> List[str]:
        return [t for t in jieba.lcut(text.lower()) if t.strip()]

except ImportError:  # jieba未安装时退化为字符级bigram

    def _tokenize(text: str) -> List[str]:
        text = text.lower().replace(" ", "")
        return [text[i : i + 2] for i in range(len(text) - 1)] or [text]


KNOWLEDGE_BASE_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")


class BaseRetriever(ABC):
    """检索器抽象接口 — 便于替换为向量检索。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """返回 [{"title": ..., "content": ..., "score": ...}, ...]"""
        ...


class BM25Retriever(BaseRetriever):
    """
    BM25 检索器。

    BM25公式: score(q,d) = Σ IDF(qi) * (f(qi,d)*(k1+1)) / (f(qi,d) + k1*(1-b+b*|d|/avgdl))
    """

    def __init__(self, docs: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b

        # 预处理：分词、文档频率
        self.doc_tokens = [_tokenize(d["title"] + " " + d["content"]) for d in docs]
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)

        self.df: Counter = Counter()
        for tokens in self.doc_tokens:
            for term in set(tokens):
                self.df[term] += 1
        self.n_docs = len(docs)

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        query_terms = _tokenize(query)
        scores = []

        for idx, tokens in enumerate(self.doc_tokens):
            tf = Counter(tokens)
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                f = tf[term]
                score += self._idf(term) * (
                    f * (self.k1 + 1)
                ) / (f + self.k1 * (1 - self.b + self.b * self.doc_lens[idx] / self.avgdl))
            scores.append((score, idx))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score <= 0:
                continue
            doc = self.docs[idx]
            results.append(
                {"title": doc["title"], "content": doc["content"], "score": f"{score:.2f}"}
            )
        return results


class VectorRetriever(BaseRetriever):
    """
    向量检索器骨架 — 升级用（需安装 faiss-cpu + sentence-transformers）。

    实现思路：
    1. 用 sentence-transformers（如 BAAI/bge-small-zh）编码所有文档
    2. 建 FAISS 索引（IndexFlatIP + 归一化 = 余弦相似度）
    3. 查询时编码query，检索top_k
    """

    def __init__(self, docs: List[Dict[str, str]]):
        raise NotImplementedError(
            "向量检索升级路径：pip install faiss-cpu sentence-transformers 后实现本类"
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        raise NotImplementedError


_retriever: BaseRetriever | None = None


def _load_knowledge_base() -> List[Dict[str, str]]:
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        with open(KNOWLEDGE_BASE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return []


def get_retriever() -> BaseRetriever:
    """获取检索器单例（默认BM25）。"""
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever(_load_knowledge_base())
    return _retriever
