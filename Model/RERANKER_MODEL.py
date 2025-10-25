import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder

# 从环境变量加载模型名称，默认为D:/MODEL/bge-reranker-large
reranker_model_name = os.getenv("RERANKER_MODEL_NAME", "D:/MODEL/bge-reranker-large")

# 初始化CrossEncoder模型
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    cross_encoder = CrossEncoder(reranker_model_name, max_length=512, device=device)
    print(f"✅ 重排序模型初始化成功: {reranker_model_name}")
    print(f"🔧 使用设备: {device}")
except Exception as e:
    print(f"❌ 重排序模型初始化失败: {str(e)}")
    # 创建一个简单的备用实现
    class SimpleReranker:
        def predict(self, pairs):
            # 简单的模拟实现 - 在实际应用中会使用真实模型
            return [0.5 for _ in pairs]
    cross_encoder = SimpleReranker()
    print("⚠️ 使用备用重排序实现")

# 重排序功能实现
def rerank_documents(query: str, documents: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float]]:
    """
    对文档列表根据与查询的相关性进行重排序
    
    Args:
        query: 查询文本
        documents: 文档列表，每个文档是包含'text'字段的字典
    
    Returns:
        排序后的文档列表，每个元素是(文档, 相关性分数)的元组
    """
    if not documents:
        return []
    
    try:
        # 准备查询-文档对
        query_doc_pairs = [(query, doc['text']) for doc in documents]
        
        # 使用CrossEncoder计算相关性分数
        scores = cross_encoder.predict(query_doc_pairs)
        
        # 将文档和分数组合并按分数降序排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 只返回前3个最相关的文档
        return scored_docs[:3]
        
    except Exception as e:
        print(f"⚠️ 重排序过程出错: {str(e)}")
        # 失败时返回原始文档（带默认分数）
        return [(doc, 0.5) for doc in documents[:3]]