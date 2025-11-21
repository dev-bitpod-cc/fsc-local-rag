"""
FSC-RAG 查詢介面

金管會文件 RAG 系統 - Streamlit 前端
"""
import os
import streamlit as st
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 頁面設定
st.set_page_config(
    page_title="金管會文件智慧查詢",
    page_icon="🏛️",
    layout="wide",
)

# 標題
st.title("🏛️ 金管會文件智慧查詢")
st.info("💡 本系統使用 BGE-M3 + Qdrant + Gemini 的自建 RAG 架構")

# 側邊欄設定
with st.sidebar:
    st.header("⚙️ 設定")

    # 資料類型篩選
    st.subheader("資料類型")
    filter_penalty = st.checkbox("裁罰案件", value=True)
    filter_law = st.checkbox("法令函釋", value=True)
    filter_announcement = st.checkbox("重要公告", value=True)

    # 搜尋參數
    st.subheader("搜尋參數")
    top_k = st.slider("搜尋結果數量", min_value=1, max_value=20, value=5)

    # 顯示模式
    st.subheader("顯示模式")
    show_sources = st.checkbox("顯示參考來源", value=True)

    st.markdown("---")
    st.markdown("**關於本系統**")
    st.markdown("""
    - Embedding: BGE-M3
    - 向量資料庫: Qdrant Cloud
    - LLM: Gemini 2.0 Flash
    """)

    # 版本號
    st.markdown("---")
    st.caption("v1.0.0")


@st.cache_resource
def get_retriever():
    """初始化檢索器（快取）"""
    from src.retriever.search import FSCRetriever
    return FSCRetriever(prefer_api=True, lazy_load=False)


@st.cache_data
def load_url_mapping():
    """載入 doc_id -> URL 映射"""
    import json
    from pathlib import Path

    mapping_file = Path(__file__).parent / 'doc_url_mapping.json'
    if mapping_file.exists():
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def get_secret(key: str, default: str = None):
    """取得密鑰，優先使用 Streamlit secrets"""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


@st.cache_resource
def get_llm():
    """初始化 Gemini LLM（快取）"""
    import google.generativeai as genai

    api_key = get_secret("GEMINI_API_KEY")
    if not api_key:
        return None

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def generate_answer(llm, query: str, context: str) -> str:
    """使用 LLM 生成回答"""
    prompt = f"""你是金融監督管理委員會的專業助理，請根據以下參考資料回答問題。

問題：{query}

參考資料：
{context}

請根據參考資料提供準確、專業的回答。如果參考資料中沒有相關資訊，請明確說明。

回答格式要求：
1. **開頭概述**：先用 1-2 句話簡要概述找到的資料情況（例如：「根據檢索到的 X 筆相關文件，主要涉及...」）
2. **主要內容**：詳細回答問題，引用具體的法規條文或裁罰案例
3. 使用繁體中文
4. 條理清晰，重點突出
"""

    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"生成回答時發生錯誤：{str(e)}"


# 初始化 session state
if 'current_query' not in st.session_state:
    st.session_state.current_query = ""

# 主要查詢區域
query = st.text_area(
    "請輸入查詢內容：",
    value=st.session_state.current_query,
    placeholder="例如：保險公司違反洗錢防制規定會受到什麼處罰？",
    height=100
)

# 快速查詢按鈕
st.markdown("#### 🚀 快速查詢")

quick_queries = [
    "違反金控法利害關係人規定會受到什麼處罰？",
    "請問在證券因為專業投資人資格審核的裁罰有哪些？",
    "辦理共同行銷被裁罰的案例有哪些？",
    "金管會對創投公司的裁罰有哪些？",
    "證券商遭主管機關裁罰「警告」處分，有哪些業務會受限制？",
    "內線交易有罪判決所認定重大訊息成立的時點"
]

cols = st.columns(2)
for idx, quick_query in enumerate(quick_queries):
    col_idx = idx % 2
    with cols[col_idx]:
        if st.button(f"📌 {quick_query}", key=f"quick_{idx}", use_container_width=True):
            st.session_state.current_query = quick_query
            st.rerun()

st.markdown("")  # 空行分隔

# 查詢按鈕
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_button = st.button("🔍 查詢", type="primary", use_container_width=True)
with col2:
    clear_button = st.button("🗑️ 清除", use_container_width=True)

if clear_button:
    st.session_state.current_query = ""
    st.rerun()

# 執行查詢
if search_button and query:
    # 建立資料類型篩選
    data_types = []
    if filter_penalty:
        data_types.append("penalty")
    if filter_law:
        data_types.append("law_interpretation")
    if filter_announcement:
        data_types.append("announcement")

    if not data_types:
        st.warning("請至少選擇一種資料類型")
    else:
        try:
            # 初始化元件
            retriever = get_retriever()
            llm = get_llm()

            # 第一階段：向量搜尋
            with st.spinner("🔍 正在搜尋相關文件..."):
                results = retriever.search(
                    query=query,
                    top_k=top_k,
                    data_types=data_types if len(data_types) < 3 else None
                )

            if not results:
                st.info("未找到相關文件，請嘗試其他關鍵字。")
            else:
                # 顯示搜尋結果數量
                st.info(f"📄 找到 {len(results)} 筆相關文件，正在準備上下文...")

                # 生成上下文
                context = retriever.get_context(
                    query=query,
                    top_k=top_k,
                    data_types=data_types if len(data_types) < 3 else None
                )

                # 第二階段：LLM 生成回答
                if llm:
                    with st.spinner("🤖 正在生成 AI 回答..."):
                        answer = generate_answer(llm, query, context)
                    st.success("✅ 查詢完成")
                    st.markdown("---")
                    st.subheader("📝 AI 回答")
                    st.markdown(answer)
                else:
                    st.warning("未設定 GEMINI_API_KEY，無法生成 AI 回答")

                # 顯示參考來源
                if show_sources:
                    st.markdown("---")
                    st.subheader(f"📚 參考來源 ({len(results)} 筆，依時間排序）")

                    # 載入 URL 映射
                    url_mapping = load_url_mapping()

                    # 按日期排序（從新到舊）
                    sorted_results = sorted(
                        results,
                        key=lambda x: x.metadata.get("date", ""),
                        reverse=True
                    )

                    for i, r in enumerate(sorted_results, 1):
                        # 資料類型標籤
                        type_labels = {
                            "penalty": "🔴 裁罰案件",
                            "law_interpretation": "🔵 法令函釋",
                            "announcement": "🟢 重要公告"
                        }
                        type_label = type_labels.get(r.data_type, r.data_type)

                        # 標題：類型 + 日期 + 名稱
                        date_str = r.metadata.get("date", "")
                        title = r.metadata.get("title", "") or r.metadata.get("entity_name", "") or r.doc_id
                        display_title = title[:40] + "..." if len(title) > 40 else title
                        expander_title = f"{type_label} | {date_str} | {display_title}"

                        with st.expander(expander_title, expanded=False):
                            # 相關度
                            st.markdown(f"**相關度:** {r.score:.2%}")

                            # 過濾無效的文號
                            doc_number = r.metadata.get("doc_number", "")
                            if doc_number and len(doc_number) < 50 and "行政院" not in doc_number and "裁罰案件" not in doc_number:
                                st.markdown(f"**文號:** {doc_number}")

                            # 內容
                            st.markdown("**內容:**")
                            display_text = r.text[:500] + "..." if len(r.text) > 500 else r.text
                            st.text(display_text)

                            # 原始連結
                            original_url = url_mapping.get(r.doc_id, "")
                            if original_url:
                                st.markdown(f"🔗 [查看金管會原始公告]({original_url})")

        except Exception as e:
            st.error(f"搜尋時發生錯誤：{str(e)}")
            st.exception(e)

elif search_button and not query:
    st.warning("⚠️ 請輸入查詢內容")

# 頁尾
st.divider()
st.caption("資料來源：金融監督管理委員會")
