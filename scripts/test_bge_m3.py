"""
BGE-M3 本地測試腳本
測試 FlagEmbedding 的 BGE-M3 模型是否正常運作
"""
import time
from FlagEmbedding import BGEM3FlagModel

def main():
    print("=" * 60)
    print("BGE-M3 本地測試")
    print("=" * 60)

    # 測試文本（中文金融法規相關）
    test_sentences = [
        "金融監督管理委員會對違反證券交易法之公司處以罰鍰",
        "保險業務員招攬保險時應善盡告知義務",
        "銀行辦理消費者貸款應遵守公平待客原則",
        "上市公司內部人於重大訊息公開前不得買賣股票",
    ]

    query = "金管會如何處罰違規的金融機構？"

    # 載入模型
    print("\n[1] 載入 BGE-M3 模型...")
    start_time = time.time()
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        use_fp16=True,  # 使用半精度加速
    )
    load_time = time.time() - start_time
    print(f"    模型載入完成，耗時: {load_time:.2f} 秒")

    # 生成文件 embedding
    print("\n[2] 生成文件 embedding...")
    start_time = time.time()
    doc_embeddings = model.encode(
        test_sentences,
        batch_size=4,
        max_length=512,
    )['dense_vecs']
    encode_time = time.time() - start_time
    print(f"    生成 {len(test_sentences)} 筆文件 embedding")
    print(f"    向量維度: {doc_embeddings.shape[1]}")
    print(f"    耗時: {encode_time:.2f} 秒")

    # 生成查詢 embedding
    print("\n[3] 生成查詢 embedding...")
    start_time = time.time()
    query_embedding = model.encode(
        [query],
        max_length=512,
    )['dense_vecs'][0]
    query_time = time.time() - start_time
    print(f"    查詢: {query}")
    print(f"    耗時: {query_time:.4f} 秒")

    # 計算相似度
    print("\n[4] 計算相似度排名...")
    import numpy as np

    similarities = np.dot(doc_embeddings, query_embedding)
    ranked_indices = np.argsort(similarities)[::-1]

    print(f"\n查詢: {query}")
    print("-" * 60)
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"  {rank}. [相似度: {similarities[idx]:.4f}]")
        print(f"     {test_sentences[idx]}")

    print("\n" + "=" * 60)
    print("測試完成！BGE-M3 模型運作正常")
    print("=" * 60)

if __name__ == "__main__":
    main()
