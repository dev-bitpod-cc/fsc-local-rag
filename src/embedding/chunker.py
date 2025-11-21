"""
文本分塊模組

針對金管會資料設計的分塊策略
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class Chunk:
    """文本區塊"""
    id: str              # chunk ID (doc_id + chunk_index)
    doc_id: str          # 原始文件 ID
    text: str            # 區塊文本
    metadata: Dict       # 元資料
    chunk_index: int     # 區塊索引
    start_char: int      # 起始字元位置
    end_char: int        # 結束字元位置


class FSCChunker:
    """金管會資料專用分塊器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 100,
        short_doc_threshold: int = 800
    ):
        """
        初始化分塊器

        Args:
            chunk_size: 目標區塊大小（字元數）
            chunk_overlap: 區塊重疊大小
            min_chunk_size: 最小區塊大小（太小的會合併）
            short_doc_threshold: 短文件閾值，低於此值不分塊
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.short_doc_threshold = short_doc_threshold

    def chunk_penalty(self, doc: Dict) -> List[Chunk]:
        """
        裁罰案件分塊策略

        結構：
        1. 標題 + 基本資訊（日期、來源、受處分人、罰款）
        2. 主旨
        3. 事實（可能很長，需分塊）
        4. 理由及法令依據

        Args:
            doc: 裁罰案件原始資料

        Returns:
            區塊列表
        """
        chunks = []
        doc_id = doc.get('id', '')
        content = doc.get('content', {}).get('text', '')
        metadata = doc.get('metadata', {})

        # 清理內容
        content = self._clean_content(content)

        # 提取結構化部分
        header = self._build_penalty_header(doc)

        # 分割內容段落
        sections = self._split_penalty_sections(content)

        chunk_index = 0

        # Chunk 0: Header + 主旨
        header_text = header
        if sections.get('主旨'):
            header_text += f"\n\n主旨：{sections['主旨']}"

        # 如果 header 太長（結構不標準的文件），需要分塊
        if len(header_text) <= self.short_doc_threshold:
            chunks.append(Chunk(
                id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                text=header_text,
                metadata={
                    'source': metadata.get('source', ''),
                    'date': doc.get('date', ''),
                    'penalty_amount': metadata.get('penalty_amount', 0),
                    'entity_name': metadata.get('penalized_entity', {}).get('name', ''),
                    'chunk_type': 'header'
                },
                chunk_index=chunk_index,
                start_char=0,
                end_char=len(header_text)
            ))
            chunk_index += 1
        else:
            # Header 太長，需要分塊處理
            header_chunks = self._split_long_text(
                header_text,
                prefix="",
                doc_id=doc_id,
                base_metadata={
                    'source': metadata.get('source', ''),
                    'date': doc.get('date', ''),
                    'penalty_amount': metadata.get('penalty_amount', 0),
                    'entity_name': metadata.get('penalized_entity', {}).get('name', ''),
                    'chunk_type': 'header'
                },
                start_index=chunk_index
            )
            chunks.extend(header_chunks)
            chunk_index += len(header_chunks)

        # Chunk 1+: 事實（可能需要分成多塊）
        if sections.get('事實'):
            fact_chunks = self._split_long_text(
                sections['事實'],
                prefix="事實：",
                doc_id=doc_id,
                base_metadata={
                    'source': metadata.get('source', ''),
                    'date': doc.get('date', ''),
                    'chunk_type': 'fact'
                },
                start_index=chunk_index
            )
            chunks.extend(fact_chunks)
            chunk_index += len(fact_chunks)

        # Chunk: 理由及法令依據
        if sections.get('理由'):
            reason_chunks = self._split_long_text(
                sections['理由'],
                prefix="理由及法令依據：",
                doc_id=doc_id,
                base_metadata={
                    'source': metadata.get('source', ''),
                    'date': doc.get('date', ''),
                    'chunk_type': 'legal_basis'
                },
                start_index=chunk_index
            )
            chunks.extend(reason_chunks)

        return chunks

    def chunk_law_interpretation(self, doc: Dict) -> List[Chunk]:
        """
        法令函釋分塊策略

        Args:
            doc: 法令函釋原始資料

        Returns:
            區塊列表
        """
        chunks = []
        doc_id = doc.get('id', '')
        content = doc.get('content', {}).get('text', '')
        metadata = doc.get('metadata', {})

        content = self._clean_content(content)

        # 法令函釋通常較短，可能不需要分塊
        header = self._build_interpretation_header(doc)
        full_text = f"{header}\n\n{content}"

        if len(full_text) <= self.short_doc_threshold:
            # 不分塊
            chunks.append(Chunk(
                id=f"{doc_id}_chunk_0",
                doc_id=doc_id,
                text=full_text,
                metadata={
                    'source': metadata.get('source', ''),
                    'date': doc.get('date', ''),
                    'type': metadata.get('type', ''),
                    'chunk_type': 'full'
                },
                chunk_index=0,
                start_char=0,
                end_char=len(full_text)
            ))
        else:
            # 需要分塊
            text_chunks = self._split_long_text(
                content,
                prefix=header + "\n\n",
                doc_id=doc_id,
                base_metadata={
                    'source': metadata.get('source', ''),
                    'date': doc.get('date', ''),
                    'type': metadata.get('type', ''),
                    'chunk_type': 'content'
                },
                start_index=0
            )
            chunks.extend(text_chunks)

        return chunks

    def chunk_announcement(self, doc: Dict) -> List[Chunk]:
        """
        重要公告分塊策略

        Args:
            doc: 重要公告原始資料

        Returns:
            區塊列表
        """
        # 公告結構與法令函釋類似
        return self.chunk_law_interpretation(doc)

    def _build_penalty_header(self, doc: Dict) -> str:
        """建立裁罰案件標題區塊"""
        metadata = doc.get('metadata', {})
        entity = metadata.get('penalized_entity', {})

        lines = [
            f"裁罰案件",
            f"日期：{doc.get('date', '')}",
            f"來源：{doc.get('source_raw', '')}",
            f"受處分人：{entity.get('name', '')}",
            f"罰款金額：{metadata.get('penalty_amount_text', '')}",
            f"發文字號：{metadata.get('doc_number', '')}"
        ]

        return "\n".join(lines)

    def _build_interpretation_header(self, doc: Dict) -> str:
        """建立法令函釋標題區塊"""
        metadata = doc.get('metadata', {})

        lines = [
            f"法令函釋",
            f"日期：{doc.get('date', '')}",
            f"來源：{doc.get('source_raw', '')}",
            f"標題：{doc.get('title', '')}",
            f"發文字號：{metadata.get('doc_number', '')}"
        ]

        return "\n".join(lines)

    def _split_penalty_sections(self, content: str) -> Dict[str, str]:
        """分割裁罰案件的各段落"""
        sections = {
            '主旨': '',
            '事實': '',
            '理由': ''
        }

        # 嘗試匹配各段落
        patterns = {
            '主旨': r'主旨[：:]\s*(.*?)(?=事實[：:]|$)',
            '事實': r'事實[：:]\s*(.*?)(?=理由|$)',
            '理由': r'理由(?:及法令依據)?[：:]\s*(.*?)(?=繳款方式|注意事項|正本|$)'
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[key] = match.group(1).strip()

        return sections

    def _split_long_text(
        self,
        text: str,
        prefix: str,
        doc_id: str,
        base_metadata: Dict,
        start_index: int
    ) -> List[Chunk]:
        """
        將長文本分成多個區塊

        Args:
            text: 要分塊的文本
            prefix: 每個區塊的前綴
            doc_id: 文件 ID
            base_metadata: 基礎元資料
            start_index: 起始區塊索引

        Returns:
            區塊列表
        """
        chunks = []

        if len(text) <= self.chunk_size:
            # 不需要分塊
            chunks.append(Chunk(
                id=f"{doc_id}_chunk_{start_index}",
                doc_id=doc_id,
                text=prefix + text,
                metadata=base_metadata.copy(),
                chunk_index=start_index,
                start_char=0,
                end_char=len(text)
            ))
            return chunks

        # 按句子分割
        sentences = self._split_sentences(text)

        current_chunk = prefix
        current_start = 0
        chunk_index = start_index

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                # 儲存當前區塊
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        id=f"{doc_id}_chunk_{chunk_index}",
                        doc_id=doc_id,
                        text=current_chunk.strip(),
                        metadata=base_metadata.copy(),
                        chunk_index=chunk_index,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    ))
                    chunk_index += 1

                # 開始新區塊（帶重疊）
                overlap_text = current_chunk[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current_chunk = prefix + overlap_text + sentence
                current_start += len(current_chunk) - len(sentence) - len(prefix) - len(overlap_text)

        # 最後一個區塊
        if current_chunk.strip() and len(current_chunk) >= self.min_chunk_size:
            chunks.append(Chunk(
                id=f"{doc_id}_chunk_{chunk_index}",
                doc_id=doc_id,
                text=current_chunk.strip(),
                metadata=base_metadata.copy(),
                chunk_index=chunk_index,
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            ))

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """分割句子"""
        # 中文句子分割
        pattern = r'([。！？；\n])'
        parts = re.split(pattern, text)

        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sentences.append(parts[i] + parts[i + 1])

        if len(parts) % 2 == 1 and parts[-1]:
            sentences.append(parts[-1])

        return sentences

    def _clean_content(self, content: str) -> str:
        """清理內容"""
        # 移除網頁雜訊
        noise_patterns = [
            r'FACEBOOK.*?(?=\n|$)',
            r'Line.*?(?=\n|$)',
            r'Twitter.*?(?=\n|$)',
            r'友善列印.*?(?=\n|$)',
            r'回上頁.*?(?=\n|$)',
            r'瀏覽人次.*?(?=\n|$)',
            r'更新日期.*?(?=\n|$)',
            r'_\s*\n',
        ]

        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)

        # 移除多餘空白
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)

        return content.strip()


# 測試用
if __name__ == "__main__":
    import json

    # 讀取測試資料
    with open('data/penalties/raw.jsonl', 'r', encoding='utf-8') as f:
        doc = json.loads(f.readline())

    chunker = FSCChunker(chunk_size=512, chunk_overlap=50)
    chunks = chunker.chunk_penalty(doc)

    print(f"文件 ID: {doc['id']}")
    print(f"分成 {len(chunks)} 個區塊")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ({chunk.metadata.get('chunk_type', '')}) ---")
        print(f"長度: {len(chunk.text)} 字元")
        print(chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text)
