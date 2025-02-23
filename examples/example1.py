import pickle
import pandas as pd
from tqdm import tqdm
import torch

from trustrag.modules.document.chunk import TextChunker
from trustrag.modules.document.txt_parser import TextParser
from trustrag.modules.document.utils import PROJECT_BASE
from trustrag.modules.generator.llm import GLM4Chat
from trustrag.modules.reranker.bge_reranker import BgeRerankerConfig, BgeReranker
from trustrag.modules.retrieval.bm25s_retriever import BM25RetrieverConfig
from trustrag.modules.retrieval.dense_retriever import DenseRetrieverConfig
from trustrag.modules.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverConfig

print("now PROJECT_BASE: ", PROJECT_BASE)


# 文档切片
def generate_chunks():
    tp = TextParser() # 代表txt格式解析
    tc = TextChunker()
    paragraphs = tp.parse(r'/home/TrustRAG/examples-datas/作文数据集0-100.txt', encoding="utf-8")
    print(len(paragraphs))
    chunks = []
    for content in tqdm(paragraphs):
        # chunk = tc.chunk_sentences([content], chunk_size=1024)
        chunk = tc.split_sentences(content)
        chunks.append(chunk)

    with open(f'{PROJECT_BASE}/output/chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)

# generate_chunks()

with open('/home/TrustRAG/trustrag/output/chunks.pkl','rb') as f:
    data = pickle.load(f)
    print(data)



# 构建检索器
# BM25 and Dense Retriever configurations
bm25_config = BM25RetrieverConfig(
    method='lucene',
    index_path='indexs/description_bm25.index',
    k1=1.6,
    b=0.7
)
bm25_config.validate()
print(bm25_config.log_config())
dense_config = DenseRetrieverConfig(
    model_name_or_path = embedding_model_path,
    dim=1024,
    index_path='indexs/dense_cache'
)
config_info = dense_config.log_config()
print(config_info)

# Hybrid Retriever configuration 由于分数框架不在同一维度，建议可以合并
hybrid_config = HybridRetrieverConfig(
    bm25_config=bm25_config,
    dense_config=dense_config,
    bm25_weight=0.7,  # bm25检索结果权重
    dense_weight=0.3  # dense检索结果权重
)
hybrid_retriever = HybridRetriever(config=hybrid_config)

# # 构建索引
# hybrid_retriever.build_from_texts(corpus)
# # 保存索引
# hybrid_retriever.save_index()

# # 加载索引
# hybrid_retriever.load_index()

# # 检索测试
# query = "支付宝"
# results = hybrid_retriever.retrieve(query, top_k=10)
# print(len(results))
# # Output results
# for result in results:
#     print(f"Text: {result['text']}, Score: {result['score']}")

# # 排序模型
# reranker_config = BgeRerankerConfig(
#     model_name_or_path=reranker_model_path
# )
# bge_reranker = BgeReranker(reranker_config)

# # 
# glm4_chat = GLM4Chat(llm_model_path)