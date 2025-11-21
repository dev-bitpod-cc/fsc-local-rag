"""
從 FSC 專案的 optimized plaintext 生成 Embedding 並上傳到 Qdrant

流程：
1. 讀取 FSC/data/plaintext_optimized/ 下的 txt 檔案
2. 分塊（短文件不分塊，長文件按 500 字分塊）
3. 生成 BGE-M3 Embedding
4. 上傳到 Qdrant Cloud

確保內容與 Gemini File Search 完全一致！
"""
import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from loguru import logger

# 設定
FSC_PROJECT_PATH = Path.home() / "Projects" / "FSC"
PLAINTEXT_BASE = FSC_PROJECT_PATH / "data" / "plaintext_optimized"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
SHORT_DOC_THRESHOLD = 800


@dataclass
class Chunk:
    """文本區塊"""
    chunk_id: str
    doc_id: str
    text: str
    data_type: str
    chunk_index: int
    metadata: Dict


def load_env():
    """載入環境變數"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()


def extract_metadata_from_text(text: str, data_type: str) -> Dict:
    """從 optimized plaintext 中提取 metadata"""
    metadata = {'data_type': data_type}

    # 分割 metadata 和 content
    parts = text.split('---', 1)
    if len(parts) == 2:
        header = parts[0].strip()

        # 解析 header 中的欄位
        for line in header.split('\n'):
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == '發文日期':
                    metadata['date'] = value
                elif key == '來源單位':
                    metadata['source'] = value
                elif key == '機構名稱':
                    metadata['entity_name'] = value
                elif key == '罰款金額':
                    metadata['penalty_amount_text'] = value
                elif key == '發文字號':
                    metadata['doc_number'] = value
                elif key == '標題':
                    metadata['title'] = value
                elif key == '相關法規':
                    metadata['law_name'] = value
                elif key == '函釋類型':
                    metadata['category'] = value

    return metadata


def split_into_chunks(
    text: str,
    doc_id: str,
    data_type: str,
    metadata: Dict
) -> List[Chunk]:
    """將文本分塊"""
    chunks = []

    # 短文件不分塊
    if len(text) <= SHORT_DOC_THRESHOLD:
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_chunk_0",
            doc_id=doc_id,
            text=text,
            data_type=data_type,
            chunk_index=0,
            metadata=metadata
        ))
        return chunks

    # 長文件按句子分塊
    sentences = re.split(r'([。！？；\n])', text)

    # 重組句子
    full_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        full_sentences.append(sentences[i] + sentences[i + 1] if i + 1 < len(sentences) else sentences[i])
    if len(sentences) % 2 == 1 and sentences[-1]:
        full_sentences.append(sentences[-1])

    # 分塊
    current_chunk = ""
    chunk_index = 0

    for sentence in full_sentences:
        if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
            current_chunk += sentence
        else:
            if current_chunk.strip():
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_chunk_{chunk_index}",
                    doc_id=doc_id,
                    text=current_chunk.strip(),
                    data_type=data_type,
                    chunk_index=chunk_index,
                    metadata=metadata
                ))
                chunk_index += 1

            # 帶 overlap 開始新塊
            overlap = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else ""
            current_chunk = overlap + sentence

    # 最後一塊
    if current_chunk.strip():
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_chunk_{chunk_index}",
            doc_id=doc_id,
            text=current_chunk.strip(),
            data_type=data_type,
            chunk_index=chunk_index,
            metadata=metadata
        ))

    return chunks


def load_plaintext_files(data_type: str) -> List[Chunk]:
    """載入並處理 plaintext 檔案"""
    type_mapping = {
        'penalty': 'penalties_individual',
        'law_interpretation': 'law_interpretations_individual',
        'announcement': 'announcements_individual',
    }

    dir_name = type_mapping.get(data_type)
    if not dir_name:
        logger.warning(f"未知類型: {data_type}")
        return []

    dir_path = PLAINTEXT_BASE / dir_name
    if not dir_path.exists():
        logger.warning(f"目錄不存在: {dir_path}")
        return []

    all_chunks = []
    txt_files = list(dir_path.glob("*.txt"))

    logger.info(f"處理 {data_type}: {len(txt_files)} 個檔案")

    for txt_file in tqdm(txt_files, desc=f"  {data_type}"):
        doc_id = txt_file.stem

        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()

        metadata = extract_metadata_from_text(text, data_type)
        chunks = split_into_chunks(text, doc_id, data_type, metadata)
        all_chunks.extend(chunks)

    return all_chunks


def generate_embeddings(chunks: List[Chunk], batch_size: int = 16) -> List[List[float]]:
    """生成 embedding"""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.embedding.embedder import LocalBGEM3Embedder

    logger.info("初始化 BGE-M3 模型...")
    embedder = LocalBGEM3Embedder()

    texts = [chunk.text for chunk in chunks]
    logger.info(f"生成 {len(texts)} 個 embedding...")

    embeddings = embedder.embed_batch(texts, batch_size=batch_size, show_progress=True)

    return [emb.tolist() for emb in embeddings]


def upload_to_qdrant(chunks: List[Chunk], vectors: List[List[float]], batch_size: int = 100):
    """上傳到 Qdrant"""
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct

    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url or not api_key:
        raise ValueError("請設定 QDRANT_URL 和 QDRANT_API_KEY")

    client = QdrantClient(url=url, api_key=api_key)
    collection_name = "fsc_documents"

    logger.info(f"上傳 {len(chunks)} 個向量到 Qdrant...")

    for i in tqdm(range(0, len(chunks), batch_size), desc="上傳"):
        batch_chunks = chunks[i:i + batch_size]
        batch_vectors = vectors[i:i + batch_size]

        points = [
            PointStruct(
                id=idx,
                vector=vec,
                payload={
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'text': chunk.text,
                    'data_type': chunk.data_type,
                    'chunk_index': chunk.chunk_index,
                    **chunk.metadata
                }
            )
            for idx, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors), start=i)
        ]

        client.upsert(collection_name=collection_name, points=points)

    info = client.get_collection(collection_name)
    logger.info(f"✓ 上傳完成！Collection 共有 {info.points_count} 個點")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='生成 Embedding 並上傳到 Qdrant')
    parser.add_argument(
        '--types',
        nargs='+',
        default=['penalty', 'law_interpretation', 'announcement'],
        help='要處理的資料類型'
    )
    parser.add_argument('--batch-size', type=int, default=16, help='Embedding 批次大小')
    parser.add_argument('--skip-upload', action='store_true', help='跳過上傳')
    args = parser.parse_args()

    load_env()

    logger.info("=" * 60)
    logger.info("FSC-RAG: 生成 Embedding 並上傳到 Qdrant")
    logger.info("=" * 60)
    logger.info(f"資料來源: {PLAINTEXT_BASE}")

    # 載入並分塊
    all_chunks = []
    for data_type in args.types:
        chunks = load_plaintext_files(data_type)
        all_chunks.extend(chunks)
        logger.info(f"  {data_type}: {len(chunks)} chunks")

    logger.info(f"\n總計: {len(all_chunks)} chunks")

    # 生成 embedding
    vectors = generate_embeddings(all_chunks, batch_size=args.batch_size)

    # 上傳
    if not args.skip_upload:
        upload_to_qdrant(all_chunks, vectors)
    else:
        logger.info("跳過上傳")

    logger.info("\n✅ 完成！")


if __name__ == "__main__":
    logger.add("logs/generate_and_upload.log", rotation="10 MB", level="DEBUG")
    main()
