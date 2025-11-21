# FSC-RAG

自建 RAG 系統，與 Gemini File Search 做效果比較。

## 專案概述

**目標**：使用開源 Embedding 模型 + 向量資料庫建立 RAG 系統，與 Google Gemini File Search 比較檢索效果。

**資料來源**：共用 FSC 專案的爬蟲資料（透過軟連結）
- 裁罰案件：630 筆
- 法令函釋：2,872 筆
- 重要公告：1,642 筆

## 技術架構

```
本地（一次性）                    雲端查詢
────────────                    ────────────
文件 → BGE-M3                   問題 → HF Inference API (BGE-M3)
      ↓                               ↓
   Qdrant Cloud  ←───────────── 向量搜尋
                                      ↓
                                 Gemini LLM 回答
```

## 技術選型

| 元件 | 選擇 | 說明 |
|------|------|------|
| Embedding 模型 | BGE-M3 | 多語言、效果好 |
| 本地 Embedding | FlagEmbedding | 本地批量處理 |
| 查詢 Embedding | HF Inference API | 雲端無需算力 |
| 向量資料庫 | Qdrant Cloud | 免費額度、功能強 |
| LLM | Gemini 2.5 Flash | 與現有系統一致 |
| 部署 | Streamlit Cloud | 現有經驗 |

## 目錄結構

```
FSC-RAG/
├── data/                    # 軟連結 → FSC/data
├── embeddings/              # 本地生成的向量
│   ├── penalties/
│   ├── law_interpretations/
│   └── announcements/
├── src/
│   ├── embedding/
│   │   ├── chunker.py       # 文本分塊
│   │   └── embedder.py      # BGE-M3 向量生成
│   ├── vectordb/
│   │   └── qdrant_client.py # Qdrant 操作
│   └── retriever/
│       └── search.py        # 檢索邏輯
├── scripts/
│   ├── generate_embeddings.py
│   └── upload_to_qdrant.py
├── app.py                   # Streamlit 查詢介面
├── requirements.txt
└── .env
```

## 環境設定

```bash
# 建立虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env 填入 API keys
```

## 開發進度

- [x] 專案結構建立
- [x] 資料軟連結
- [x] BGE-M3 本地 embedding 測試
- [x] Chunking 策略設計
- [x] Qdrant Cloud 設定
- [x] 批量 embedding 生成（13,637 chunks）
- [x] 上傳至 Qdrant
- [x] HF Inference API 整合
- [ ] Streamlit 查詢介面
- [ ] 效果比較測試
