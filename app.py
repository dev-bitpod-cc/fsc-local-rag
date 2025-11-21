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
st.markdown("使用 BGE-M3 + Qdrant + Gemini 的 RAG 系統")

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
    - LLM: Gemini 2.5 Flash
    """)


@st.cache_resource
def get_retriever():
    """初始化檢索器（快取）"""
    from src.retriever.search import FSCRetriever
    return FSCRetriever(prefer_api=True, lazy_load=False)


@st.cache_resource
def get_llm():
    """初始化 Gemini LLM（快取）"""
    import google.generativeai as genai

    api_key = os.getenv("GEMINI_API_KEY")
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
回答時請：
1. 引用具體的法規條文或裁罰案例
2. 使用繁體中文
3. 條理清晰，重點突出
"""

    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"生成回答時發生錯誤：{str(e)}"


# 主要查詢區域
query = st.text_input(
    "🔍 請輸入您的問題",
    placeholder="例如：保險公司違反洗錢防制規定會受到什麼處罰？"
)

# 查詢按鈕
if st.button("搜尋", type="primary") or (query and st.session_state.get("auto_search")):
    if not query:
        st.warning("請輸入查詢問題")
    else:
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
            with st.spinner("正在搜尋相關文件..."):
                try:
                    # 初始化元件
                    retriever = get_retriever()
                    llm = get_llm()

                    # 執行搜尋
                    results = retriever.search(
                        query=query,
                        top_k=top_k,
                        data_types=data_types if len(data_types) < 3 else None
                    )

                    if not results:
                        st.info("未找到相關文件，請嘗試其他關鍵字。")
                    else:
                        # 生成上下文
                        context = retriever.get_context(
                            query=query,
                            top_k=top_k,
                            data_types=data_types if len(data_types) < 3 else None
                        )

                        # LLM 回答
                        st.subheader("💡 AI 回答")
                        if llm:
                            with st.spinner("正在生成回答..."):
                                answer = generate_answer(llm, query, context)
                                st.markdown(answer)
                        else:
                            st.warning("未設定 GEMINI_API_KEY，無法生成 AI 回答")

                        # 顯示參考來源
                        if show_sources:
                            st.markdown("---")
                            st.subheader(f"📚 參考來源 ({len(results)} 筆)")

                            for i, r in enumerate(results, 1):
                                # 資料類型標籤
                                type_labels = {
                                    "penalty": "🔴 裁罰案件",
                                    "law_interpretation": "🔵 法令函釋",
                                    "announcement": "🟢 重要公告"
                                }
                                type_label = type_labels.get(r.data_type, r.data_type)

                                with st.expander(
                                    f"{type_label} | 相關度: {r.score:.2%}",
                                    expanded=(i <= 2)
                                ):
                                    # 元資料
                                    cols = st.columns(3)
                                    if r.metadata.get("date"):
                                        cols[0].markdown(f"**日期:** {r.metadata['date']}")
                                    if r.metadata.get("title"):
                                        cols[1].markdown(f"**標題:** {r.metadata['title'][:30]}...")
                                    if r.metadata.get("doc_number"):
                                        cols[2].markdown(f"**文號:** {r.metadata['doc_number']}")

                                    # 內容
                                    st.markdown("**內容:**")
                                    st.text(r.text[:500] + "..." if len(r.text) > 500 else r.text)

                                    st.caption(f"文件 ID: {r.doc_id} | Chunk ID: {r.chunk_id}")

                except Exception as e:
                    st.error(f"搜尋時發生錯誤：{str(e)}")
                    st.exception(e)

# 範例查詢
st.markdown("---")
st.subheader("💡 範例查詢")

example_queries = [
    "保險公司違反洗錢防制規定會受到什麼處罰？",
    "證券交易法第171條的相關函釋有哪些？",
    "銀行違反個資法的裁罰案例",
    "金融機構內部控制缺失的處分標準",
]

cols = st.columns(2)
for i, eq in enumerate(example_queries):
    if cols[i % 2].button(eq, key=f"example_{i}"):
        st.session_state["auto_search"] = True
        st.rerun()
