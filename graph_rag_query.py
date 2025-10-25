#!/usr/bin/env python3
# coding: utf-8
"""
GraphRAG 查询工具
演示如何使用 Chroma 向量数据库和关系数据进行语义检索和图结构推理
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dotenv import load_dotenv

# 导入统一的嵌入模型
from Model.EMBEDDING_MODEL import global_embeddings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='graph_rag_query.log'
)
logger = logging.getLogger("graph_rag_query")

# 加载环境变量
load_dotenv()


class GraphRAGQueryEngine:
    """GraphRAG 查询引擎，结合向量检索和图结构推理"""
    
    def __init__(self):
        """初始化查询引擎"""
        # 向量数据库配置
        from config import config
        self.vector_db_path = config.VECTOR_DB_PATH
        os.makedirs(self.vector_db_path, exist_ok=True)
        
        # 关系数据文件路径
        self.relations_file_path = "relations_data.json"
        
        # 使用统一的全局嵌入模型
        self.embeddings = global_embeddings
        
        # 加载关系数据
        self.relations_by_source = self._load_relations_data()
        
        # 加载 Chroma 集合
        self.vectorstore = self._load_chroma_collection()
        
        logger.info(f"GraphRAGQueryEngine 初始化成功（使用统一本地嵌入模型，向量数据库: Chroma）")
    
    def _load_relations_data(self) -> Dict[int, List[Dict[str, Any]]]:
        """加载关系数据（优化大文件加载）"""
        try:
            import ijson
            relations = {}
            
            logger.info(f"开始加载关系数据文件: {self.relations_file_path}")
            with open(self.relations_file_path, 'rb') as f:
                # 使用ijson流式读取大JSON文件
                objects = ijson.kvitems(f, '')
                for node_id_str, rels in objects:
                    node_id = int(node_id_str)
                    relations[node_id] = rels
            
            logger.info(f"成功加载关系数据，共 {len(relations)} 个节点的关系")
            return relations
        except FileNotFoundError:
            logger.error(f"关系数据文件未找到: {self.relations_file_path}")
            return {}
    
    def _load_chroma_collection(self) -> Chroma:
        """加载 Chroma 集合"""
        logger.info("加载 Chroma 集合...")
        
        try:
            # 尝试从已有路径加载集合
            vectorstore = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embeddings,
                collection_name="medical_graph"
            )
            
            # 验证集合是否有数据
            if vectorstore._collection.count() == 0:
                logger.warning(f"Chroma 集合为空，请先运行 neo4j_to_chroma.py")
            
            logger.info(f"成功加载 Chroma 集合")
            return vectorstore
        except Exception as e:
            logger.error(f"加载 Chroma 集合失败: {str(e)}")
            raise Exception(f"Chroma 集合加载失败，请先运行 neo4j_to_chroma.py")
    
    def vector_search(self, query_text: str, top_k: int = 3, node_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        在 Chroma 中进行向量相似度检索
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            node_type: 可选的节点类型过滤
        
        Returns:
            检索结果列表，包含节点信息和相似度
        """
        logger.info(f"执行向量检索：'{query_text}'，top_k={top_k}")
        
        # 执行检索
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query=query_text,
            k=top_k
        )
        
        # 处理检索结果
        search_results = []
        for doc, score in results:
            # 从元数据中提取信息
            node_id = doc.metadata.get("node_id")
            node_type_val = doc.metadata.get("node_type")
            properties = doc.metadata.get("properties", {})
            
            # 应用节点类型过滤
            if node_type and node_type_val != node_type:
                continue
            
            result = {
                "node_id": node_id,
                "node_type": node_type_val,
                "properties": properties,
                "distance": 1 - score,  # 将相似度转换为距离（1-相似度）
                "similarity": score  # Chroma直接提供相似度
            }
            search_results.append(result)
        
        # 按距离排序并限制数量
        search_results = sorted(search_results, key=lambda x: x["distance"])[:top_k]
        
        logger.info(f"向量检索完成，返回 {len(search_results)} 条结果")
        return search_results
    
    def expand_with_graph_relations(self, node_ids: List[int], relation_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        根据图关系扩展节点信息
        
        Args:
            node_ids: 源节点 ID 列表
            relation_types: 可选的关系类型过滤
        
        Returns:
            扩展后的关系信息列表
        """
        logger.info(f"执行图关系扩展：节点数={len(node_ids)}")
        
        expanded_relations = []
        
        for node_id in node_ids:
            if node_id in self.relations_by_source:
                relations = self.relations_by_source[node_id]
                
                for rel in relations:
                    # 如果指定了关系类型，进行过滤
                    if relation_types and rel["relation_type"] not in relation_types:
                        continue
                    
                    # 构建扩展关系信息
                    expanded_rel = {
                        "source_id": rel["source_id"],
                        "target_id": rel["target_id"],
                        "relation_type": rel["relation_type"],
                        "properties": rel["properties"]
                    }
                    expanded_relations.append(expanded_rel)
        
        logger.info(f"图关系扩展完成，返回 {len(expanded_relations)} 条关系")
        return expanded_relations
    
    def get_target_node_info(self, target_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        根据目标节点 ID 获取节点信息
        
        Args:
            target_ids: 目标节点 ID 列表
        
        Returns:
            节点 ID 到节点信息的映射
        """
        if not target_ids:
            return {}
        
        # 使用 Chroma 的向量检索来查找特定节点
        node_map = {}
        
        # 针对每个目标ID构建查询
        for node_id in target_ids:
            # 使用元数据过滤查询特定node_id
            results = self.vectorstore.similarity_search(
                query="",  # 空查询
                k=1,  # 只需要一个结果
                filter={"node_id": node_id}  # 精确匹配node_id
            )
            
            if results:
                doc = results[0]
                node_map[node_id] = {
                    "node_type": doc.metadata.get("node_type"),
                    "properties": doc.metadata.get("properties", {})
                }
        
        return node_map
    
    def query(self, question: str, top_k: int = 3, relation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        执行完整的 GraphRAG 查询
        1. 向量检索相关节点
        2. 基于节点 ID 扩展图关系
        3. 获取目标节点信息
        4. 整合结果
        
        Args:
            question: 用户问题
            top_k: 向量检索返回的节点数量
            relation_types: 要扩展的关系类型列表，如 ["RECOMMEND_DRUG", "HAS_SYMPTOM"]
        
        Returns:
            整合后的查询结果
        """
        logger.info(f"执行 GraphRAG 查询：'{question}'")
        
        # 步骤1：向量检索
        search_results = self.vector_search(question, top_k=top_k)
        
        # 提取检索到的节点 ID
        retrieved_node_ids = [result["node_id"] for result in search_results]
        
        # 步骤2：扩展图关系
        expanded_relations = self.expand_with_graph_relations(retrieved_node_ids, relation_types)
        
        # 提取目标节点 ID
        target_ids = list(set([rel["target_id"] for rel in expanded_relations]))
        
        # 步骤3：获取目标节点信息
        target_node_map = self.get_target_node_info(target_ids)
        
        # 步骤4：整合结果
        final_results = {
            "question": question,
            "retrieved_nodes": search_results,
            "relations": [
                {
                    "source_id": rel["source_id"],
                    "source_info": next((r for r in search_results if r["node_id"] == rel["source_id"]), None),
                    "relation_type": rel["relation_type"],
                    "target_id": rel["target_id"],
                    "target_info": target_node_map.get(rel["target_id"]),
                    "properties": rel["properties"]
                }
                for rel in expanded_relations
            ]
        }
        
        logger.info(f"GraphRAG 查询完成，发现 {len(expanded_relations)} 条相关关系")
        return final_results
    
    def format_answer(self, results: Dict[str, Any]) -> str:
        """
        格式化查询结果为可读文本
        
        Args:
            results: GraphRAG 查询结果
        
        Returns:
            格式化后的文本回答
        """
        parts = [f"问题: {results['question']}", "", "基于知识图谱的回答:", ""]
        
        # 添加检索到的节点信息
        if results['retrieved_nodes']:
            parts.append("🔍 检索到的相关实体:")
            for node in results['retrieved_nodes']:
                node_name = node['properties'].get('name', '未命名')
                node_type = node['node_type']
                similarity = node['similarity']
                parts.append(f"  - [{node_type}] {node_name} (相关度: {similarity:.3f})")
            parts.append("")
        
        # 添加关系信息
        if results['relations']:
            parts.append("📊 知识图谱关系:")
            
            # 按关系类型分组
            relations_by_type = {}
            for rel in results['relations']:
                rel_type = rel['relation_type']
                if rel_type not in relations_by_type:
                    relations_by_type[rel_type] = []
                relations_by_type[rel_type].append(rel)
            
            # 为每种关系类型生成描述
            for rel_type, rels in relations_by_type.items():
                # 映射关系类型到中文描述
                rel_type_zh = {
                    "HAS_SYMPTOM": "症状",
                    "RECOMMEND_DRUG": "推荐药物",
                    "NOT_EAT": "忌口",
                    "DO_EAT": "宜吃食物",
                    "ACOMPANY": "并发症"
                }.get(rel_type, rel_type)
                
                parts.append(f"\n  {rel_type_zh}:")
                
                # 按源节点分组
                source_relations = {}
                for rel in rels:
                    source_id = rel['source_id']
                    if source_id not in source_relations:
                        source_relations[source_id] = []
                    source_relations[source_id].append(rel)
                
                # 为每个源节点生成描述
                for source_id, source_rels in source_relations.items():
                    source_node = results['retrieved_nodes'][0]  # 简化处理
                    source_name = source_node['properties'].get('name', '未命名')
                    
                    # 收集目标节点名称
                    target_names = []
                    for rel in source_rels:
                        if rel['target_info']:
                            target_name = rel['target_info']['properties'].get('name', '未知')
                            target_names.append(target_name)
                    
                    if target_names:
                        parts.append(f"    {source_name} → {rel_type_zh}: {', '.join(target_names)}")
        
        # 如果没有找到相关信息
        if not results['retrieved_nodes'] and not results['relations']:
            parts.append("没有找到相关信息，请尝试其他问题或检查知识图谱数据。")
        
        return "\n".join(parts)
    
    def __del__(self):
        """清理资源"""
        # Chroma 不需要手动释放集合资源
        logger.info("GraphRAGQueryEngine 资源已释放")


def main():
    """演示 GraphRAG 查询引擎的使用"""
    print("🤖 GraphRAG 查询引擎演示")
    print("-------------------------")
    
    try:
        # 创建查询引擎实例
        engine = GraphRAGQueryEngine()
        
        # 示例查询1：查询感冒的症状
        print("\n📝 示例1：查询感冒的症状")
        results1 = engine.query(
            question="感冒有哪些症状？",
            top_k=2,
            relation_types=["HAS_SYMPTOM"]
        )
        print(engine.format_answer(results1))
        
        # 示例查询2：查询感冒推荐药物
        print("\n📝 示例2：查询感冒推荐药物")
        results2 = engine.query(
            question="感冒应该吃什么药？",
            top_k=2,
            relation_types=["RECOMMEND_DRUG"]
        )
        print(engine.format_answer(results2))
        
        # 示例查询3：综合查询
        print("\n📝 示例3：综合查询（症状和忌口）")
        results3 = engine.query(
            question="高血压的注意事项",
            top_k=2,
            relation_types=["HAS_SYMPTOM", "NOT_EAT"]
        )
        print(engine.format_answer(results3))
        
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()