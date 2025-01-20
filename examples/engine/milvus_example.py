from trustrag.modules.retrieval.embedding import SentenceTransformerEmbedding
from trustrag.modules.engine.milvus import MilvusEngine
if __name__ == '__main__':
    # 初始化 MilvusEngine
    local_embedding_generator = SentenceTransformerEmbedding(model_name_or_path="all-MiniLM-L6-v2", device="cpu")
    milvus_engine = MilvusEngine(
        collection_name="my_collection",
        embedding_generator=local_embedding_generator,
        milvus_client_params={"uri": "http://localhost:19530", "token": "root:Milvus"},
        vector_size=1536
    )

    # 定义过滤条件
    conditions = [
        {"key": "color", "value": "red", "operator": "like"},  # color like "red"
        {"key": "likes", "value": 50, "operator": ">"}  # likes > 50
    ]

    # 构建过滤表达式
    filter_expr = milvus_engine.build_filter(conditions)
    print("Filter Expression:", filter_expr)

    # 执行搜索
    results = milvus_engine.search(
        text="Find similar vectors",
        query_filter=filter_expr,
        limit=5
    )

    # 打印结果
    for result in results:
        print(result)