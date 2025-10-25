import os
import torch
import logging
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("embedding_model")

# 设备检测 - 优先使用CUDA
device = os.getenv("DEVICE", "cuda").lower()
if device == "cuda" and torch.cuda.is_available():
    device = "cuda"
elif device == "cuda" and not torch.cuda.is_available():
    device = "cpu"
    logger.warning("CUDA不可用，将使用CPU")

# 优先从EMBEDDING_MODEL_NAME环境变量获取模型路径
local_model_path = os.getenv("EMBEDDING_MODEL_NAME", "D:/MODEL/bce-embedding-base_v1")
logger.info(f"使用嵌入模型: {local_model_path}, 设备: {device}")

# 初始化全局嵌入模型
global_embeddings = HuggingFaceEmbeddings(
    model_name=local_model_path,
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)


def generate_embedding(text: str) -> List[float]:
    """
    生成文本的嵌入向量
    
    Args:
        text: 输入文本
        
    Returns:
        嵌入向量列表
    """
    try:
        embedding = global_embeddings.embed_query(text)
        logger.info(f"成功生成文本嵌入，维度: {len(embedding)}")
        return embedding
    except Exception as e:
        logger.error(f"生成嵌入失败: {str(e)}")
        raise


def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    计算两个嵌入向量的余弦相似度
    
    Args:
        embedding1: 第一个嵌入向量
        embedding2: 第二个嵌入向量
        
    Returns:
        相似度分数
    """
    try:
        from numpy import dot
        from numpy.linalg import norm
        
        if norm(embedding1) > 0 and norm(embedding2) > 0:
            similarity = dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))
            return float(similarity)
        else:
            return 0.0
    except Exception as e:
        logger.error(f"计算相似度失败: {str(e)}")
        raise
