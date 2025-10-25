#!/usr/bin/env python3
# coding: utf-8
"""
Neo4J 知识图谱转 Chroma 向量数据库工具
将 Neo4J 中的知识图谱数据导入到 Chroma 向量数据库中进行语义检索
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from dotenv import load_dotenv

# 导入统一的嵌入模型
# 移除CPU限制，优先使用GPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from Model.EMBEDDING_MODEL import global_embeddings as embeddings
# 导入Neo4J单例
from neo4j_singleton import Neo4jSingleton

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='neo4j_to_chroma.log'
)
logger = logging.getLogger("neo4j_to_chroma")

# 加载环境变量
load_dotenv()

# 配置参数
CHROMA_INSERT_BATCH = 100  # 批量插入大小
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./chroma_db")
COLLECTION_NAME = "medical_graph"
RELATIONS_FILE_PATH = "relations_data.json"


class Neo4jToChromaConverter:
    """Neo4J 知识图谱转 Chroma 向量数据库转换器"""
    
    def __init__(self):
        """初始化转换器"""
        # 使用统一的全局嵌入模型
        self.embeddings = embeddings
        
        # 确保向量数据库目录存在
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        # 初始化Chroma向量数据库
        self.vectorstore = Chroma(
            persist_directory=VECTOR_DB_PATH,
            embedding_function=self.embeddings,
            collection_name=COLLECTION_NAME
        )
        
        logger.info(f"Neo4jToChromaConverter 初始化成功（使用统一本地嵌入模型）")
    
    def extract_all_nodes(self) -> List[Dict[str, Any]]:
        """
        从Neo4J提取所有节点
        
        Returns:
            节点列表，每个节点包含id、类型和属性
        """
        logger.info("开始从Neo4J提取所有节点...")
        
        query = """
        MATCH (n)
        RETURN id(n) as node_id, labels(n) as node_types, properties(n) as properties
        """
        
        try:
            # 获取Neo4J驱动实例并创建会话
            driver = Neo4jSingleton.get_instance()
            with driver.session() as session:
                result = session.run(query)
                nodes = []
                
                for record in tqdm(result, desc="提取节点"):
                    # 获取主要标签（第一个标签）作为节点类型
                    node_types = record["node_types"]
                    node_type = node_types[0] if node_types else "Unknown"
                    
                    nodes.append({
                        "node_id": record["node_id"],
                        "node_type": node_type,
                        "properties": record["properties"]
                    })
                
            logger.info(f"成功提取 {len(nodes)} 个节点")
            return nodes
        except Exception as e:
            logger.error(f"提取节点失败: {str(e)}")
            raise
    
    def extract_all_relations(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        从Neo4J提取所有关系
        
        Returns:
            以源节点ID为键，关系列表为值的字典
        """
        logger.info("开始从Neo4J提取所有关系...")
        
        query = """
        MATCH (s)-[r]->(t)
        RETURN id(s) as source_id, id(t) as target_id, type(r) as relation_type, properties(r) as properties
        """
        
        try:
            # 获取Neo4J驱动实例并创建会话
            driver = Neo4jSingleton.get_instance()
            with driver.session() as session:
                result = session.run(query)
                relations_by_source = {}
                
                for record in tqdm(result, desc="提取关系"):
                    source_id = record["source_id"]
                    relation = {
                        "source_id": source_id,
                        "target_id": record["target_id"],
                        "relation_type": record["relation_type"],
                        "properties": record["properties"]
                    }
                    
                    if source_id not in relations_by_source:
                        relations_by_source[source_id] = []
                    relations_by_source[source_id].append(relation)
                
            logger.info(f"成功提取 {sum(len(rels) for rels in relations_by_source.values())} 条关系")
            return relations_by_source
        except Exception as e:
            logger.error(f"提取关系失败: {str(e)}")
            raise
    
    def build_document_from_node(self, node: Dict[str, Any]) -> Document:
        """
        将节点转换为Document对象，并过滤复杂元数据结构
        
        Args:
            node: 节点信息
            
        Returns:
            Document对象
        """
        # 构建文档内容
        node_type = node["node_type"]
        properties = node["properties"]
        node_id = node["node_id"]
        
        # 根据节点类型构建不同的内容
        if node_type == "Disease":
            name = properties.get("name", "未命名疾病")
            # 使用Neo4J中实际的字段名称
            description = properties.get("desc", "")
            cause = properties.get("cause", "")
            cure_way = properties.get("cure_way", "")
            cured_prob = properties.get("cured_prob", "")
            cure_department = properties.get("cure_department", "")
            cost_money = properties.get("cost_money", "")
            yibao_status = properties.get("yibao_status", "")
            
            content = f"疾病: {name}\n"
            if description:
                content += f"描述: {description}\n"
            if cause:
                content += f"病因: {cause}\n"
            if cure_way:
                content += f"治疗方式: {cure_way}\n"
            if cured_prob:
                content += f"治愈率: {cured_prob}\n"
            if cure_department:
                content += f"相关科室: {cure_department}\n"
            if cost_money:
                content += f"治疗费用: {cost_money}\n"
            if yibao_status:
                content += f"医保状态: {yibao_status}"
        elif node_type == "Symptom":
            name = properties.get("name", "未命名症状")
            content = f"症状: {name}"
        elif node_type == "Drug":
            name = properties.get("name", "未命名药物")
            content = f"药物: {name}"
        elif node_type == "Food":
            name = properties.get("name", "未命名食物")
            content = f"食物: {name}"
        else:
            name = properties.get("name", "未命名实体")
            content = f"{node_type}: {name}"
        
        # 添加其他属性，但先简化复杂结构
        for key, value in properties.items():
            if key not in ["name", "description"]:
                # 简化显示复杂结构
                if isinstance(value, (list, dict)):
                    if isinstance(value, list) and value:
                        # 列表只显示前两个元素并加上省略号
                        preview = ", ".join(str(item) for item in value[:2])
                        content += f"\n{key}: [{preview}...](共{len(value)}项)"
                    else:
                        content += f"\n{key}: [复杂数据结构]"
                else:
                    content += f"\n{key}: {value}"
        
        # 创建metadata，只包含简单类型的值
        metadata = {
            "node_id": node_id,
            "node_type": node_type,
            "name": properties.get("name", "")
        }
        
        # 手动过滤并添加简单类型的属性
        for key, value in properties.items():
            if key not in ["name", "node_id", "node_type"]:
                # 只添加符合Chroma要求的简单类型
                if isinstance(value, (str, int, float, bool, type(None))):
                    metadata[key] = value
                # 对于列表或字典，我们可以将其转换为字符串表示
                elif isinstance(value, list):
                    # 只取列表的前3个元素并转为字符串
                    if len(value) <= 3:
                        metadata[key] = str(value[:3])
                    else:
                        metadata[key] = str(value[:3]) + "..."
                elif isinstance(value, dict):
                    # 只取字典的前3个键值对并转为字符串
                    dict_items = list(value.items())[:3]
                    metadata[key] = str(dict_items) + ("..." if len(value) > 3 else "")
        
        # 创建Document对象
        document = Document(
            page_content=content,
            metadata=metadata
        )
        
        return document
    
    def import_nodes_to_chroma(self, nodes: List[Dict[str, Any]]):
        """
        将节点导入到Chroma向量数据库
        
        Args:
            nodes: 节点列表
        """
        logger.info(f"开始将 {len(nodes)} 个节点导入到Chroma...")
        
        # 清空现有集合
        self.vectorstore._collection.delete(where={"node_type": {"$ne": ""}})
        logger.info("已清空现有Chroma集合")
        
        # 批量处理节点
        for i in tqdm(range(0, len(nodes), CHROMA_INSERT_BATCH), desc="导入节点到Chroma"):
            batch = nodes[i:i + CHROMA_INSERT_BATCH]
            
            # 构建文档
            documents = [self.build_document_from_node(node) for node in batch]
            
            # 添加到Chroma
            try:
                self.vectorstore.add_documents(documents)
            except Exception as e:
                logger.error(f"批量导入失败（索引 {i}-{i+len(batch)-1}）: {str(e)}")
                # 尝试逐个导入失败的批次
                for doc in documents:
                    try:
                        self.vectorstore.add_documents([doc])
                    except Exception as e_single:
                        logger.error(f"单个文档导入失败（节点ID: {doc.metadata['node_id']}）: {str(e_single)}")
        
        # 持久化
        self.vectorstore.persist()
        logger.info(f"节点导入完成，Chroma集合中共有 {self.vectorstore._collection.count()} 个文档")
    
    def save_relations_data(self, relations_by_source: Dict[int, List[Dict[str, Any]]]):
        """
        保存关系数据到JSON文件
        
        Args:
            relations_by_source: 关系数据
        """
        logger.info(f"开始保存关系数据到 {RELATIONS_FILE_PATH}...")
        
        try:
            # 将整数键转换为字符串以便JSON序列化
            relations_json = {str(node_id): rels for node_id, rels in relations_by_source.items()}
            
            with open(RELATIONS_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(relations_json, f, ensure_ascii=False, indent=2)
            
            logger.info(f"关系数据保存成功")
        except Exception as e:
            logger.error(f"保存关系数据失败: {str(e)}")
            raise
    
    def convert(self):
        """
        执行完整的转换流程
        1. 提取节点
        2. 提取关系
        3. 导入节点到Chroma
        4. 保存关系数据
        """
        try:
            logger.info("开始Neo4J到Chroma的转换流程...")
            
            # 步骤1: 提取节点
            nodes = self.extract_all_nodes()
            
            # 步骤2: 提取关系
            relations_by_source = self.extract_all_relations()
            
            # 步骤3: 导入节点到Chroma
            self.import_nodes_to_chroma(nodes)
            
            # 步骤4: 保存关系数据
            self.save_relations_data(relations_by_source)
            
            logger.info("Neo4J到Chroma的转换流程完成！")
            print("\n✅ 转换完成！")
            print(f"📊 导入节点数量: {len(nodes)}")
            print(f"🔗 导入关系数量: {sum(len(rels) for rels in relations_by_source.values())}")
            print(f"💾 向量数据库路径: {VECTOR_DB_PATH}")
            print(f"📄 关系数据文件: {RELATIONS_FILE_PATH}")
            
        except Exception as e:
            logger.error(f"转换过程失败: {str(e)}")
            raise
    
    def verify_import(self):
        """
        验证导入结果
        """
        logger.info("开始验证导入结果...")
        
        # 检查Chroma集合
        collection_count = self.vectorstore._collection.count()
        logger.info(f"Chroma集合中文档数量: {collection_count}")
        
        # 检查关系数据文件
        if os.path.exists(RELATIONS_FILE_PATH):
            with open(RELATIONS_FILE_PATH, 'r', encoding='utf-8') as f:
                relations = json.load(f)
            logger.info(f"关系数据文件中节点关系数量: {len(relations)}")
        
        # 执行简单的向量检索测试
        test_queries = ["感冒", "头痛", "高血压"]
        for query in test_queries:
            results = self.vectorstore.similarity_search(query, k=3)
            logger.info(f"测试查询 '{query}' 返回 {len(results)} 个结果")
        
        logger.info("导入验证完成")
        print("\n✅ 验证完成！")
        print(f"📊 Chroma集合中文档数量: {collection_count}")


def main():
    """
    主函数
    """
    print("🤖 Neo4J 知识图谱转 Chroma 向量数据库工具")
    print("=========================================")
    print(f"📁 向量数据库路径: {VECTOR_DB_PATH}")
    print(f"📊 集合名称: {COLLECTION_NAME}")
    print(f"🔗 关系数据文件: {RELATIONS_FILE_PATH}")
    print(f"🚀 使用统一本地嵌入模型")
    print()
    
    try:
        # 创建转换器实例
        converter = Neo4jToChromaConverter()
        
        # 执行转换
        converter.convert()
        
        # 验证导入
        converter.verify_import()
        
    except KeyboardInterrupt:
        print("\n❌ 操作被用户中断")
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 确保Neo4J驱动关闭
        logger.info("清理资源，关闭连接")
        # 使用Neo4jSingleton的close_instance方法关闭连接
        Neo4jSingleton.close_instance()


if __name__ == "__main__":
    main()