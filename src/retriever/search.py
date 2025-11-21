"""
FSC-RAG 搜尋模組

整合 Embedding 生成和 Qdrant 向量搜尋
"""
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class SearchResult:
    """搜尋結果"""
    doc_id: str
    chunk_id: str
    text: str
    score: float
    data_type: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'chunk_id': self.chunk_id,
            'text': self.text,
            'score': self.score,
            'data_type': self.data_type,
            'metadata': self.metadata
        }


class FSCRetriever:
    """
    金管會文件檢索器

    整合查詢 embedding 生成和 Qdrant 向量搜尋
    """

    def __init__(
        self,
        prefer_api: bool = True,
        lazy_load: bool = True
    ):
        """
        初始化檢索器

        Args:
            prefer_api: 是否優先使用 HF API 進行 embedding
            lazy_load: 是否延遲載入模型（首次查詢時載入）
        """
        self.prefer_api = prefer_api
        self.embedder = None
        self.qdrant_client = None
        self.lazy_load = lazy_load

        if not lazy_load:
            self._init_components()

    def _init_components(self):
        """初始化元件"""
        if self.embedder is None:
            from src.embedding.embedder import QueryEmbedder
            logger.info("初始化 Query Embedder...")
            self.embedder = QueryEmbedder(prefer_api=self.prefer_api)

        if self.qdrant_client is None:
            from src.vectordb.qdrant_client import FSCQdrantClient
            logger.info("初始化 Qdrant Client...")
            self.qdrant_client = FSCQdrantClient()

    def search(
        self,
        query: str,
        top_k: int = 10,
        data_types: Optional[List[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        搜尋相關文件

        Args:
            query: 查詢文本
            top_k: 返回結果數量
            data_types: 限制搜尋的資料類型 ['penalty', 'law_interpretation', 'announcement']
            score_threshold: 最低分數門檻

        Returns:
            搜尋結果列表
        """
        # 確保元件已初始化
        self._init_components()

        # 生成查詢向量
        logger.debug(f"生成查詢向量: {query[:50]}...")
        query_vector = self.embedder.embed(query).tolist()

        # 執行向量搜尋
        results = []

        if data_types:
            # 分別搜尋每個資料類型
            for data_type in data_types:
                type_results = self.qdrant_client.search(
                    query_vector=query_vector,
                    top_k=top_k,
                    data_type=data_type,
                    score_threshold=score_threshold,
                )
                results.extend(type_results)

            # 按分數排序並取 top_k
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
        else:
            # 搜尋所有類型
            results = self.qdrant_client.search(
                query_vector=query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
            )

        # 轉換為 SearchResult
        search_results = []
        for r in results:
            search_results.append(SearchResult(
                doc_id=r.doc_id,
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.score,
                data_type=r.metadata.get('data_type', ''),
                metadata=r.metadata
            ))

        return search_results

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        data_types: Optional[List[str]] = None,
        max_tokens: int = 4000,
    ) -> str:
        """
        取得用於 LLM 的上下文文本

        Args:
            query: 查詢文本
            top_k: 返回結果數量
            data_types: 限制搜尋的資料類型
            max_tokens: 最大 token 數（粗略估計）

        Returns:
            格式化的上下文文本
        """
        results = self.search(
            query=query,
            top_k=top_k,
            data_types=data_types,
        )

        if not results:
            return "未找到相關文件。"

        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 2  # 粗略估計：1 token ≈ 2 字元（中文）

        for i, r in enumerate(results, 1):
            # 格式化單一結果
            part = f"【文件 {i}】\n"
            part += f"類型: {self._translate_data_type(r.data_type)}\n"

            if r.metadata.get('title'):
                part += f"標題: {r.metadata['title']}\n"
            if r.metadata.get('date'):
                part += f"日期: {r.metadata['date']}\n"
            if r.metadata.get('doc_number'):
                part += f"文號: {r.metadata['doc_number']}\n"

            part += f"內容:\n{r.text}\n"
            part += f"(相關度: {r.score:.3f})\n"
            part += "-" * 40 + "\n"

            if total_chars + len(part) > max_chars:
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n".join(context_parts)

    def _translate_data_type(self, data_type: str) -> str:
        """翻譯資料類型"""
        mapping = {
            'penalty': '裁罰案件',
            'law_interpretation': '法令函釋',
            'announcement': '重要公告',
        }
        return mapping.get(data_type, data_type)


# 快捷函數
def search(
    query: str,
    top_k: int = 10,
    data_types: Optional[List[str]] = None,
) -> List[SearchResult]:
    """
    快捷搜尋函數

    Args:
        query: 查詢文本
        top_k: 返回結果數量
        data_types: 限制搜尋的資料類型

    Returns:
        搜尋結果列表
    """
    retriever = FSCRetriever()
    return retriever.search(query, top_k=top_k, data_types=data_types)


def get_context(
    query: str,
    top_k: int = 5,
    data_types: Optional[List[str]] = None,
) -> str:
    """
    快捷取得上下文函數

    Args:
        query: 查詢文本
        top_k: 返回結果數量
        data_types: 限制搜尋的資料類型

    Returns:
        格式化的上下文文本
    """
    retriever = FSCRetriever()
    return retriever.get_context(query, top_k=top_k, data_types=data_types)


# 測試
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__file__).rsplit('/', 3)[0])

    print("=" * 60)
    print("FSC-RAG 搜尋測試")
    print("=" * 60)

    retriever = FSCRetriever()

    # 測試查詢
    test_queries = [
        "保險公司違反洗錢防制規定的裁罰案例",
        "證券交易法第171條相關函釋",
    ]

    for query in test_queries:
        print(f"\n查詢: {query}")
        print("-" * 40)

        results = retriever.search(query, top_k=3)

        for i, r in enumerate(results, 1):
            print(f"\n{i}. [{r.data_type}] {r.doc_id}")
            print(f"   分數: {r.score:.4f}")
            print(f"   內容: {r.text[:100]}...")
