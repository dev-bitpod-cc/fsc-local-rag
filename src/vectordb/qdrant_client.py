"""
Qdrant Cloud 向量資料庫操作模組
"""
import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
)
from dotenv import load_dotenv
from loguru import logger


# 載入環境變數
load_dotenv()

# BGE-M3 向量維度
VECTOR_DIMENSION = 1024

# Collection 名稱
COLLECTION_NAME = "fsc_documents"


@dataclass
class SearchResult:
    """搜尋結果"""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class FSCQdrantClient:
    """金管會文件向量資料庫客戶端"""

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = COLLECTION_NAME
    ):
        """
        初始化 Qdrant 客戶端

        Args:
            url: Qdrant Cloud URL，預設從環境變數讀取
            api_key: Qdrant API Key，預設從環境變數讀取
            collection_name: Collection 名稱
        """
        self.url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection_name = collection_name

        if not self.url or not self.api_key:
            raise ValueError(
                "請設定 QDRANT_URL 和 QDRANT_API_KEY 環境變數，"
                "或在初始化時提供 url 和 api_key 參數"
            )

        self.client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )
        logger.info(f"已連接到 Qdrant Cloud: {self.url}")

    def create_collection(self, recreate: bool = False) -> bool:
        """
        建立 Collection

        Args:
            recreate: 是否重新建立（會刪除現有資料）

        Returns:
            是否成功建立
        """
        # 檢查 collection 是否存在
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists:
            if recreate:
                logger.warning(f"刪除現有 collection: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                logger.info(f"Collection 已存在: {self.collection_name}")
                return True

        # 建立 collection
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION,
                distance=Distance.COSINE,
            ),
        )

        # 建立索引以加速篩選查詢
        self._create_payload_indexes()

        logger.info(f"已建立 collection: {self.collection_name}")
        return True

    def _create_payload_indexes(self):
        """建立 payload 欄位索引"""
        # Keyword 索引
        keyword_fields = ["doc_id", "data_type", "date", "source", "chunk_type"]
        for field in keyword_fields:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

        # Integer 索引
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="chunk_index",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        # Text 索引（全文搜尋）
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="text",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.MULTILINGUAL,
                min_token_len=2,
                max_token_len=20,
            ),
        )

        logger.info("已建立 payload 索引")

    def get_collection_info(self) -> Dict:
        """取得 Collection 資訊"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "status": str(info.status),
                "vector_size": info.config.params.vectors.size,
                "distance": str(info.config.params.vectors.distance),
            }
        except Exception as e:
            logger.error(f"取得 collection 資訊失敗: {e}")
            return {}

    def upsert_chunks(
        self,
        chunk_ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict],
        batch_size: int = 100
    ) -> int:
        """
        批量上傳向量

        Args:
            chunk_ids: Chunk ID 列表
            vectors: 向量列表
            payloads: Payload 列表
            batch_size: 批次大小

        Returns:
            成功上傳的數量
        """
        if len(chunk_ids) != len(vectors) != len(payloads):
            raise ValueError("chunk_ids, vectors, payloads 長度必須一致")

        total = len(chunk_ids)
        uploaded = 0

        for i in range(0, total, batch_size):
            batch_ids = chunk_ids[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            batch_payloads = payloads[i:i + batch_size]

            points = [
                PointStruct(
                    id=idx,
                    vector=vec,
                    payload=payload
                )
                for idx, (vec, payload) in enumerate(
                    zip(batch_vectors, batch_payloads),
                    start=i
                )
            ]

            # 使用 chunk_id 作為 payload 的一部分
            for j, point in enumerate(points):
                point.payload["chunk_id"] = batch_ids[j]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

            uploaded += len(points)
            logger.info(f"已上傳 {uploaded}/{total} 個向量")

        return uploaded

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        data_type: Optional[str] = None,
        source: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """
        向量搜尋

        Args:
            query_vector: 查詢向量
            top_k: 返回結果數量
            data_type: 篩選資料類型
            source: 篩選來源
            date_from: 日期起始（含）
            date_to: 日期結束（含）
            score_threshold: 最低分數門檻

        Returns:
            搜尋結果列表
        """
        # 建立篩選條件
        must_conditions = []

        if data_type:
            must_conditions.append(
                FieldCondition(
                    key="data_type",
                    match=MatchValue(value=data_type)
                )
            )

        if source:
            must_conditions.append(
                FieldCondition(
                    key="source",
                    match=MatchValue(value=source)
                )
            )

        if date_from:
            must_conditions.append(
                FieldCondition(
                    key="date",
                    range=models.Range(gte=date_from)
                )
            )

        if date_to:
            must_conditions.append(
                FieldCondition(
                    key="date",
                    range=models.Range(lte=date_to)
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # 執行搜尋 (使用新版 API: query_points)
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        # 轉換結果
        search_results = []
        for hit in results.points:
            payload = hit.payload or {}
            search_results.append(SearchResult(
                chunk_id=payload.get("chunk_id", ""),
                doc_id=payload.get("doc_id", ""),
                text=payload.get("text", ""),
                score=hit.score,
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ["chunk_id", "doc_id", "text"]
                }
            ))

        return search_results

    def delete_by_doc_id(self, doc_id: str) -> bool:
        """
        刪除指定文件的所有 chunks

        Args:
            doc_id: 文件 ID

        Returns:
            是否成功
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="doc_id",
                                match=MatchValue(value=doc_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"已刪除文件 {doc_id} 的所有 chunks")
            return True
        except Exception as e:
            logger.error(f"刪除失敗: {e}")
            return False

    def count_by_data_type(self) -> Dict[str, int]:
        """
        統計各資料類型的數量

        Returns:
            各類型的數量
        """
        counts = {}
        for data_type in ["penalty", "law_interpretation", "announcement"]:
            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=Filter(
                    must=[
                        FieldCondition(
                            key="data_type",
                            match=MatchValue(value=data_type)
                        )
                    ]
                )
            )
            counts[data_type] = result.count
        return counts


# 測試用
if __name__ == "__main__":
    # 測試連線
    try:
        client = FSCQdrantClient()
        print("連線成功！")

        # 取得 collection 資訊
        info = client.get_collection_info()
        if info:
            print(f"Collection: {info['name']}")
            print(f"向量數量: {info['vectors_count']}")
            print(f"狀態: {info['status']}")
        else:
            print("Collection 不存在，嘗試建立...")
            client.create_collection()
            print("Collection 建立成功！")

    except Exception as e:
        print(f"錯誤: {e}")
        print("\n請確認 .env 檔案已設定 QDRANT_URL 和 QDRANT_API_KEY")
