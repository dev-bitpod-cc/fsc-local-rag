"""
BGE-M3 Embedding 模組

本地使用 FlagEmbedding 進行向量化
查詢時使用 HuggingFace Inference API
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()


def get_secret(key: str, default: str = None) -> Optional[str]:
    """
    取得密鑰，優先使用 Streamlit secrets，否則使用環境變數
    """
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


class LocalBGEM3Embedder:
    """本地 BGE-M3 Embedding（用於批量處理文件）"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        """
        初始化本地 BGE-M3 模型

        Args:
            model_name: 模型名稱，預設 BAAI/bge-m3
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """載入 BGE-M3 模型"""
        try:
            from FlagEmbedding import BGEM3FlagModel
            logger.info(f"載入 BGE-M3 模型: {self.model_name}")
            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=True,  # M3 Ultra 支援
                device="mps"    # Apple Silicon
            )
            logger.info("BGE-M3 模型載入完成")
        except Exception as e:
            logger.error(f"載入模型失敗: {e}")
            raise

    def embed_single(self, text: str) -> np.ndarray:
        """
        向量化單一文本

        Args:
            text: 輸入文本

        Returns:
            向量 (1024 維)
        """
        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )
        return result['dense_vecs'][0]

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        批量向量化文本

        Args:
            texts: 文本列表
            batch_size: 批次大小
            show_progress: 是否顯示進度條

        Returns:
            向量陣列 (N, 1024)
        """
        all_embeddings = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding")

        for i in iterator:
            batch = texts[i:i + batch_size]
            result = self.model.encode(
                batch,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False
            )
            all_embeddings.append(result['dense_vecs'])

        return np.vstack(all_embeddings)

    def embed_and_save(
        self,
        texts: List[str],
        ids: List[str],
        metadata: List[Dict],
        output_path: str,
        batch_size: int = 32
    ):
        """
        向量化並儲存到檔案

        Args:
            texts: 文本列表
            ids: ID 列表
            metadata: 元資料列表
            output_path: 輸出路徑
            batch_size: 批次大小
        """
        embeddings = self.embed_batch(texts, batch_size)

        output = {
            'model': self.model_name,
            'dimension': embeddings.shape[1],
            'count': len(ids),
            'data': []
        }

        for i, (id_, emb, meta) in enumerate(zip(ids, embeddings, metadata)):
            output['data'].append({
                'id': id_,
                'embedding': emb.tolist(),
                'metadata': meta
            })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"已儲存 {len(ids)} 筆向量到 {output_path}")


class HFInferenceEmbedder:
    """HuggingFace Inference API Embedding（用於查詢時向量化）"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        api_token: Optional[str] = None
    ):
        """
        初始化 HF Inference API

        Args:
            model_name: 模型名稱
            api_token: HuggingFace API Token
        """
        self.model_name = model_name
        self.api_token = api_token or get_secret("HF_API_TOKEN")

        if not self.api_token:
            logger.warning("未設定 HF_API_TOKEN，可能會有速率限制")

        # 使用 huggingface_hub InferenceClient
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(token=self.api_token)
            logger.info(f"HF Inference API 初始化完成: {model_name}")
        except ImportError:
            logger.error("請安裝 huggingface_hub: pip install huggingface_hub")
            raise

    def embed_single(self, text: str) -> np.ndarray:
        """
        透過 API 向量化單一文本

        Args:
            text: 輸入文本

        Returns:
            向量 (1024 維)
        """
        try:
            result = self.client.feature_extraction(
                text,
                model=self.model_name
            )
            # 結果可能是 list 或 nested list
            arr = np.array(result)

            # 如果是 2D (sequence_len, hidden_dim)，取平均或第一個 token
            if arr.ndim == 2:
                # 取平均 (mean pooling)
                return arr.mean(axis=0)
            return arr

        except Exception as e:
            logger.error(f"HF API 錯誤: {e}")
            raise

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量向量化（API 版本）

        Args:
            texts: 文本列表

        Returns:
            向量陣列
        """
        embeddings = []
        for text in tqdm(texts, desc="API Embedding"):
            emb = self.embed_single(text)
            embeddings.append(emb)
        return np.array(embeddings)


class TEIEmbedder:
    """
    Text Embedding Inference (TEI) Embedder

    使用 HuggingFace TEI 服務，可以部署在自己的伺服器上，
    或使用 HuggingFace Inference Endpoints。

    TEI 專門為 embedding 優化，支援 BGE-M3 等模型。
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_token: Optional[str] = None
    ):
        """
        初始化 TEI Embedder

        Args:
            endpoint_url: TEI 服務端點 URL
            api_token: API Token（如果需要認證）
        """
        self.endpoint_url = endpoint_url or get_secret("TEI_ENDPOINT_URL")
        self.api_token = api_token or get_secret("HF_API_TOKEN")

        if not self.endpoint_url:
            raise ValueError(
                "請設定 TEI_ENDPOINT_URL 環境變數或提供 endpoint_url 參數"
            )

        logger.info(f"TEI Embedder 初始化: {self.endpoint_url}")

    def embed_single(self, text: str) -> np.ndarray:
        """向量化單一文本"""
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        response = requests.post(
            f"{self.endpoint_url}/embed",
            headers=headers,
            json={"inputs": text}
        )

        if response.status_code != 200:
            raise Exception(f"TEI API 錯誤: {response.status_code} - {response.text}")

        result = response.json()
        return np.array(result[0] if isinstance(result[0], list) else result)

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """批量向量化"""
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        all_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="TEI Embedding"):
            batch = texts[i:i + batch_size]

            response = requests.post(
                f"{self.endpoint_url}/embed",
                headers=headers,
                json={"inputs": batch}
            )

            if response.status_code != 200:
                raise Exception(f"TEI API 錯誤: {response.status_code}")

            result = response.json()
            all_embeddings.extend(result)

        return np.array(all_embeddings)


class QueryEmbedder:
    """
    查詢 Embedding 管理器

    自動選擇最佳的 embedding 方式：
    1. 優先使用 HF Inference API（如果 token 有效）
    2. 回退到本地 BGE-M3 模型
    """

    def __init__(
        self,
        prefer_api: bool = True,
        api_token: Optional[str] = None,
        model_name: str = "BAAI/bge-m3"
    ):
        """
        初始化查詢 Embedder

        Args:
            prefer_api: 是否優先使用 API
            api_token: HuggingFace API Token
            model_name: 模型名稱
        """
        self.model_name = model_name
        self.api_token = api_token or get_secret("HF_API_TOKEN")
        self.embedder = None
        self.mode = None

        if prefer_api and self._test_api():
            self.mode = "api"
            logger.info("使用 HF Inference API 進行查詢向量化")
        else:
            self._init_local()

    def _test_api(self) -> bool:
        """測試 API 是否可用"""
        if not self.api_token or self.api_token == "your_huggingface_token_here":
            logger.info("HF API Token 未設定，將使用本地模型")
            return False

        try:
            self.embedder = HFInferenceEmbedder(
                model_name=self.model_name,
                api_token=self.api_token
            )
            # 測試一個簡單查詢
            test_result = self.embedder.embed_single("test")
            if test_result is not None and len(test_result) > 0:
                return True
        except Exception as e:
            logger.warning(f"HF API 測試失敗: {e}")

        return False

    def _init_local(self):
        """初始化本地模型"""
        try:
            logger.info("初始化本地 BGE-M3 模型...")
            self.embedder = LocalBGEM3Embedder(model_name=self.model_name)
            self.mode = "local"
            logger.info("使用本地模型進行查詢向量化")
        except Exception as e:
            logger.error(f"無法初始化本地模型: {e}")
            raise RuntimeError(
                "無法初始化 embedding 模型。請確認：\n"
                "1. 已設定有效的 HF_API_TOKEN 環境變數，或\n"
                "2. 已安裝 FlagEmbedding 和 torch（本地模式）"
            )

    def embed(self, text: str) -> np.ndarray:
        """
        向量化查詢文本

        Args:
            text: 查詢文本

        Returns:
            向量 (1024 維)
        """
        return self.embedder.embed_single(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量向量化"""
        return self.embedder.embed_batch(texts)


# 測試用
if __name__ == "__main__":
    # 測試本地 embedding
    print("測試本地 BGE-M3...")
    embedder = LocalBGEM3Embedder()

    test_texts = [
        "金管會裁罰案件查詢系統",
        "保險法第171條之1第4項規定",
        "證券交易法違規處分"
    ]

    embeddings = embedder.embed_batch(test_texts, batch_size=2)
    print(f"向量維度: {embeddings.shape}")
    print(f"第一個向量前 5 維: {embeddings[0][:5]}")
