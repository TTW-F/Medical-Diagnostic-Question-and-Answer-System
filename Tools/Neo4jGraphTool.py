import logging
import re
import sys
import os
from typing import List, Dict, Optional


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.tools import Tool

try:
    from Model.EMBEDDING_MODEL import global_embeddings
    EMBEDDING_AVAILABLE = True
except ImportError:
    logging.warning("无法导入Model.EMBEDDING_MODEL")
    EMBEDDING_AVAILABLE = False

# 配置医疗场景日志
logging.basicConfig(
    filename="medical_kg_queries.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("medical_kg")

try:
    from Tools.GraphRAGTool import GraphRAGMedicalTool
    GRAPHRAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"GraphRAGTool导入失败: {str(e)}")
    GRAPHRAG_AVAILABLE = False

try:
    from neo4j import exceptions as neo4j_exceptions
    from neo4j_singleton import Neo4jSingleton
    NEO4J_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Neo4j相关模块导入失败: {str(e)}")
    NEO4J_AVAILABLE = False


class MedicalKGQueryTool:
    def __init__(self):
        self.graph_rag_tool = None
        self.driver = None
        self.use_graphrag = False
        
        # 初始化GraphRAG（如果可用）
        if GRAPHRAG_AVAILABLE and EMBEDDING_AVAILABLE:
            try:
                # 只使用统一的全局嵌入模型（本地部署）
                self.embeddings = global_embeddings
                self.graph_rag_tool = GraphRAGMedicalTool()
                self.use_graphrag = True
                logger.info("✅ 成功初始化GraphRAG医疗工具（使用统一本地嵌入模型）")
            except Exception as e:
                logger.warning(f"GraphRAG初始化失败: {str(e)}，将尝试使用Neo4j")
                self.use_graphrag = False
        elif GRAPHRAG_AVAILABLE:
            logger.warning("无法初始化GraphRAG工具，因为本地嵌入模型不可用")
            self.use_graphrag = False
        
        # 如果GraphRAG不可用，尝试使用Neo4j
        if not self.use_graphrag and NEO4J_AVAILABLE:
            try:
                self.driver = Neo4jSingleton.get_instance()
                logger.info("✅ 成功初始化Neo4j连接")
            except Exception as e:
                logger.error(f"Neo4j连接失败: {str(e)}")
                self.driver = None
        
        # 保留别名映射用于兼容性
        self.symptom_aliases = {
            "发烧": "发热",
            "拉肚子": "腹泻",
            "嗓子疼": "咽痛",
            "肚子疼": "腹痛",
            "头疼": "头痛"
        }
        # 疾病别名映射
        self.disease_aliases = {
            "肺气肿": ["阻塞性肺气肿", "肺积气"],
            "心梗": ["心肌梗死", "急性心肌梗死"],
            "脑梗": ["脑梗死", "缺血性脑卒中"]
        }
        
        if self.use_graphrag:
            logger.info("✅ 医疗知识图谱工具初始化完成，使用GraphRAG增强查询能力（基于Chroma向量数据库和统一本地嵌入模型）")
        elif self.driver:
            logger.info("✅ 医疗知识图谱工具初始化完成，使用Neo4j备用模式")
        else:
            logger.warning("⚠️ 医疗知识图谱工具初始化完成，但所有后端都不可用")

    # ------------------------------
    # 通用工具方法：名称标准化
    # ------------------------------
    def _normalize_symptom(self, symptom: str) -> str:
        """标准化症状名称（处理别名、冗余修饰词）"""
        # 如果GraphRAG可用，使用其方法
        if self.use_graphrag and self.graph_rag_tool:
            return self.graph_rag_tool._normalize_symptom(symptom)
        
        # 备用实现
        normalized = symptom.strip().replace("，", ",").replace("。", ".")
        # 移除常见冗余修饰词
        stop_words = ["有点", "轻微", "严重", "很", "非常", "感觉"]
        for word in stop_words:
            normalized = normalized.replace(word, "")
        # 替换已知别名
        return self.symptom_aliases.get(normalized, normalized)

    def _normalize_disease(self, disease: str) -> str:
        """标准化疾病名称（处理别名）"""
        # 如果GraphRAG可用，使用其方法
        if self.use_graphrag and self.graph_rag_tool:
            return self.graph_rag_tool._normalize_disease(disease)
        
        # 备用实现
        normalized = disease.strip().replace("，", ",").replace("。", ".")
        # 处理别名映射（如"阻塞性肺气肿"→"肺气肿"）
        for formal_name, aliases in self.disease_aliases.items():
            if normalized in aliases:
                return formal_name
        return normalized

    # ------------------------------
    # 功能1：根据症状查询可能的疾病（Symptom → Disease）
    # 使用GraphRAG增强版本
    # ------------------------------
    def query_symptom_related_diseases(self, symptom: str) -> List[Dict]:
        """根据症状查询可能的关联疾病（如“咳嗽”→ 可能的疾病列表）"""
        # 如果GraphRAG可用，使用其方法
        if self.use_graphrag and self.graph_rag_tool:
            try:
                return self.graph_rag_tool.query_symptom_related_diseases(symptom)
            except Exception as e:
                logger.error(f"GraphRAG症状查询失败: {str(e)}")
        
        # 如果Neo4j可用，使用备用实现
        if self.driver:
            if not symptom or not isinstance(symptom, str) or len(symptom.strip()) == 0:
                error_msg = "输入错误：请提供有效的症状名称（如'咳嗽'、'头痛'）"
                logger.warning(f"无效症状查询：{symptom} - {error_msg}")
                return [{"error": error_msg}]

            # 处理多症状（逗号分割）
            raw_symptoms = [s.strip() for s in re.split(r'[,、，]', symptom) if s.strip()]
            normalized_symptoms = [self._normalize_symptom(s) for s in raw_symptoms]
            logger.info(f"标准化症状列表：{raw_symptoms} → {normalized_symptoms}")

            try:
                with self.driver.session() as session:
                    # 症状查疾病Cypher查询
                    SYMPTOM_TO_DISEASE_TEMPLATE = """
                    MATCH (d:Disease)-[:has_symptom]->(s:Symptom)
                    WHERE s.name = $symptom OR s.name =~ '(?i).*' + $symptom + '.*'
                    RETURN 
                        d.name AS disease_name,
                        d.cured_prob AS cure_probability,
                        d.desc AS disease_description,
                        d.cure_department AS related_department,
                        collect(DISTINCT s.name) AS matched_symptoms
                    ORDER BY d.cured_prob DESC LIMIT 5
                    """
                    # 合并多症状查询条件
                    symptoms_str = "|".join(normalized_symptoms)
                    result = session.run(
                        SYMPTOM_TO_DISEASE_TEMPLATE,
                        symptom=symptoms_str
                    )
                    diseases = [dict(record) for record in result]

                    if not diseases:
                        msg = f"未查询到与「{symptom}」相关的疾病。建议：1. 检查症状名称；2. 尝试其他表述（如'发烧'→'发热'）；3. 咨询医师。"
                        logger.info(f"症状查询无结果：{symptom}")
                        return [{"message": msg}]

                    # 结果格式化
                    for disease in diseases:
                        if "cure_probability" in disease and disease["cure_probability"] is not None:
                            try:
                                prob_str = str(disease["cure_probability"]).replace("%", "").strip()
                                prob = float(prob_str)
                                if prob <= 1:
                                    disease["cure_probability_str"] = f"{prob * 100:.1f}%"
                                else:
                                    disease["cure_probability_str"] = f"{prob:.1f}%"
                            except (ValueError, TypeError):
                                disease["cure_probability_str"] = f"{disease['cure_probability']}（格式异常）"
                        disease["disclaimer"] = "注：结果仅供参考，不构成诊断建议，请遵医嘱。"

                    logger.info(f"症状查询成功：{symptom} → 返回{len(diseases)}条结果")
                    return diseases
            except neo4j_exceptions.Neo4jError as e:
                error_msg = f"Neo4j症状查询失败：{str(e)}"
                logger.error(error_msg, exc_info=True)
        

        return [{"error": "医疗知识图谱查询服务暂时不可用，请稍后重试"}]


    def query_disease_detail(self, disease: str) -> List[Dict]:
        """查询疾病的基本信息（如病因、治疗方式、治愈率等）"""
        # 如果GraphRAG可用，使用其方法
        if self.use_graphrag and self.graph_rag_tool:
            try:
                return self.graph_rag_tool.query_disease_detail(disease)
            except Exception as e:
                logger.error(f"GraphRAG疾病详情查询失败: {str(e)}")
        
        # 如果Neo4j可用，使用备用实现
        if self.driver:
            if not disease or not isinstance(disease, str) or len(disease.strip()) == 0:
                error_msg = "输入错误：请提供有效的疾病名称（如'肺气肿'）"
                logger.warning(f"无效疾病查询：{disease} - {error_msg}")
                return [{"error": error_msg}]

            normalized_disease = self._normalize_disease(disease)
            if normalized_disease != disease:
                logger.info(f"疾病名称标准化：{disease} → {normalized_disease}")

            try:
                with self.driver.session() as session:
                    # 疾病详情查询的Cypher查询
                    DISEASE_DETAIL_TEMPLATE = """
                    MATCH (d:Disease)
                    WHERE d.name = $disease OR d.name =~ '(?i).*' + $disease + '.*'
                    RETURN 
                        d.name AS disease_name,
                        d.desc AS description,
                        d.cause AS cause,
                        d.cure_way AS treatment,
                        d.cured_prob AS cure_probability,
                        d.cure_department AS related_department,
                        d.cost_money AS treatment_cost,
                        d.yibao_status AS insurance_status
                    LIMIT 1
                    """
                    result = session.run(
                        DISEASE_DETAIL_TEMPLATE,
                        disease=normalized_disease
                    )
                    details = [dict(record) for record in result]

                    if not details:
                        msg = f"未查询到「{disease}」的信息。建议：1. 检查疾病名称是否正确；2. 尝试别名（如'心梗'→'心肌梗死'）。"
                        logger.info(f"疾病查询无结果：{disease}")
                        return [{"message": msg}]

                    # 结果格式化
                    for item in details:
                        if "cure_probability" in item and item["cure_probability"] is not None:
                            try:
                                prob = float(item["cure_probability"])
                                item["cure_probability_str"] = f"{prob * 100:.1f}%"
                            except (ValueError, TypeError):
                                item["cure_probability_str"] = f"{item['cure_probability']}（格式异常）"
                        item["disclaimer"] = "注：医疗信息仅供参考，具体治疗请遵医嘱。"

                    logger.info(f"疾病查询成功：{disease} → 返回基本信息")
                    return details
            except neo4j_exceptions.Neo4jError as e:
                error_msg = f"Neo4j疾病查询失败：{str(e)}"
                logger.error(error_msg, exc_info=True)
        
        # 所有后端都不可用时的错误处理
        return [{"error": "医疗知识图谱查询服务暂时不可用，请稍后重试"}]

    # ------------------------------
    # 功能3：查询疾病的症状（如肺气肿的典型症状）
    # 使用GraphRAG增强版本
    # ------------------------------
    def query_disease_symptoms(self, disease: str) -> List[Dict]:
        """查询疾病的典型症状"""
        # 如果GraphRAG可用，使用其方法
        if self.use_graphrag and self.graph_rag_tool:
            try:
                return self.graph_rag_tool.query_disease_symptoms(disease)
            except Exception as e:
                logger.error(f"GraphRAG疾病症状查询失败: {str(e)}")
        
        # 如果Neo4j可用，使用备用实现
        if self.driver:
            if not disease or not isinstance(disease, str) or len(disease.strip()) == 0:
                error_msg = "输入错误：请提供有效的疾病名称（如'肺气肿'）"
                logger.warning(f"无效疾病症状查询：{disease} - {error_msg}")
                return [{"error": error_msg}]

            normalized_disease = self._normalize_disease(disease)
            if normalized_disease != disease:
                logger.info(f"疾病名称标准化：{disease} → {normalized_disease}")

            try:
                with self.driver.session() as session:
                    # 疾病症状查询的Cypher查询
                    DISEASE_SYMPTOMS_TEMPLATE = """
                    MATCH (d:Disease)-[:has_symptom]->(s:Symptom)
                    WHERE d.name = $disease OR d.name =~ '(?i).*' + $disease + '.*'
                    RETURN 
                        s.name AS symptom_name,
                        s.desc AS symptom_description
                    ORDER BY s.name
                    """
                    result = session.run(
                        DISEASE_SYMPTOMS_TEMPLATE,
                        disease=normalized_disease
                    )
                    symptoms = [dict(record) for record in result]

                    if not symptoms:
                        msg = f"未查询到「{disease}」的症状信息。建议：1. 检查疾病名称是否正确；2. 尝试别名（如'心梗'→'心肌梗死'）。"
                        logger.info(f"疾病症状查询无结果：{disease}")
                        return [{"message": msg}]

                    logger.info(f"疾病症状查询成功：{disease} → 返回{len(symptoms)}条症状")
                    return symptoms
            except neo4j_exceptions.Neo4jError as e:
                error_msg = f"Neo4j疾病症状查询失败：{str(e)}"
                logger.error(error_msg, exc_info=True)
        
        # 所有后端都不可用时的错误处理
        return [{"error": "医疗知识图谱查询服务暂时不可用，请稍后重试"}]
    

    def query_drug_recommendations(self, disease: str) -> List[Dict]:
        """查询疾病的推荐药物"""
        # 如果GraphRAG可用，使用其方法
        if self.use_graphrag and self.graph_rag_tool:
            try:
                return self.graph_rag_tool.query_drug_recommendations(disease)
            except Exception as e:
                logger.error(f"GraphRAG药物推荐查询失败: {str(e)}")
        
        # Neo4j备用实现
        if self.driver:
            if not disease or not isinstance(disease, str) or len(disease.strip()) == 0:
                return [{"error": "输入错误：请提供有效的疾病名称"}]

            normalized_disease = self._normalize_disease(disease)
            
            try:
                with self.driver.session() as session:
                    # 药物推荐查询的Cypher查询
                    DRUG_RECOMMENDATION_TEMPLATE = """
                    MATCH (d:Disease)-[:recommend_drug]->(drug:Drug)
                    WHERE d.name = $disease OR d.name =~ '(?i).*' + $disease + '.*'
                    RETURN 
                        drug.name AS drug_name,
                        drug.desc AS drug_description,
                        drug.usage AS usage
                    LIMIT 10
                    """
                    result = session.run(
                        DRUG_RECOMMENDATION_TEMPLATE,
                        disease=normalized_disease
                    )
                    drugs = [dict(record) for record in result]

                    if not drugs:
                        return [{"message": f"未查询到「{disease}」的推荐药物信息"}]

                    logger.info(f"药物推荐查询成功：{disease} → 返回{len(drugs)}条药物信息")
                    return drugs
            except neo4j_exceptions.Neo4jError as e:
                logger.error(f"Neo4j药物推荐查询失败：{str(e)}")
        
        # 所有后端都不可用时的错误处理
        return [{"error": "医疗知识图谱查询服务暂时不可用，请稍后重试"}]
    

    def query_dietary_advice(self, disease: str) -> Dict[str, List[Dict]]:
        """查询疾病的饮食建议（宜吃和忌吃）"""
        # 如果GraphRAG可用，使用其方法
        if self.use_graphrag and self.graph_rag_tool:
            try:
                return self.graph_rag_tool.query_dietary_advice(disease)
            except Exception as e:
                logger.error(f"GraphRAG饮食建议查询失败: {str(e)}")
        

        if self.driver:
            if not disease or not isinstance(disease, str) or len(disease.strip()) == 0:
                return {"error": ["输入错误：请提供有效的疾病名称"]}

            normalized_disease = self._normalize_disease(disease)
            result = {"宜吃": [], "忌吃": []}
            
            try:
                with self.driver.session() as session:
                    # 查询宜吃食物
                    RECOMMENDED_FOOD_TEMPLATE = """
                    MATCH (d:Disease)-[:recommend_food]->(food:Food)
                    WHERE d.name = $disease OR d.name =~ '(?i).*' + $disease + '.*'
                    RETURN 
                        food.name AS food_name,
                        food.desc AS food_description
                    LIMIT 10
                    """
                    recommended_result = session.run(
                        RECOMMENDED_FOOD_TEMPLATE,
                        disease=normalized_disease
                    )
                    result["宜吃"] = [dict(record) for record in recommended_result]

                    # 查询忌吃食物
                    NOT_RECOMMENDED_FOOD_TEMPLATE = """
                    MATCH (d:Disease)-[:avoid_food]->(food:Food)
                    WHERE d.name = $disease OR d.name =~ '(?i).*' + $disease + '.*'
                    RETURN 
                        food.name AS food_name,
                        food.desc AS food_description
                    LIMIT 10
                    """
                    not_recommended_result = session.run(
                        NOT_RECOMMENDED_FOOD_TEMPLATE,
                        disease=normalized_disease
                    )
                    result["忌吃"] = [dict(record) for record in not_recommended_result]

                    if not result["宜吃"] and not result["忌吃"]:
                        result = {"message": [f"未查询到「{disease}」的饮食建议信息"]}

                    logger.info(f"饮食建议查询成功：{disease} → 宜吃{len(result.get('宜吃', []))}种，忌吃{len(result.get('忌吃', []))}种")
                    return result
            except neo4j_exceptions.Neo4jError as e:
                logger.error(f"Neo4j饮食建议查询失败：{str(e)}")
                return {"error": ["饮食建议查询服务暂时不可用"]}
        
        # 所有后端都不可用时的错误处理
        return {"error": ["医疗知识图谱查询服务暂时不可用，请稍后重试"]}
    

    def general_graphrag_query(self, question: str, top_k: int = 3, relation_types=None) -> Dict:
        """执行通用的GraphRAG查询"""
        # 这个功能仅在GraphRAG可用时才能工作
        if self.use_graphrag and self.graph_rag_tool:
            try:
                return self.graph_rag_tool.general_graphrag_query(question, top_k, relation_types)
            except Exception as e:
                logger.error(f"GraphRAG通用查询失败: {str(e)}")
                return {"error": "通用查询服务暂时不可用"}
        

        return {"error": "通用查询功能仅在GraphRAG可用时才能使用"}


    def get_langchain_tools(self) -> List[Tool]:
        """生成多个LangChain Tool，对应不同功能，包含GraphRAG增强功能"""
        return [
            # 工具1：症状查疾病
            Tool(
                name="SymptomToDisease",
                func=self.query_symptom_related_diseases,
                description="""
                根据症状查询可能的疾病，输入为症状名称（多个症状用逗号分隔，如"咳嗽,发热"）。
                适用于用户问“咳嗽可能是什么病”“发烧、头痛是什么原因”等场景。
                """,
                return_direct=False
            ),
            # 工具2：疾病查详情
            Tool(
                name="DiseaseDetail",
                func=self.query_disease_detail,
                description="""
                查询疾病的基本信息（病因、治疗方式、治愈率等），输入为疾病名称（如"肺气肿"）。
                适用于用户问“什么是肺气肿”“肺气肿怎么治疗”等场景。
                """,
                return_direct=False
            ),
            # 工具3：疾病查症状
            Tool(
                name="DiseaseSymptoms",
                func=self.query_disease_symptoms,
                description="""
                查询某疾病的典型症状，输入为疾病名称（如"肺气肿"）。
                适用于用户问“肺气肿有什么症状”“心肌梗死会疼吗”等场景。
                """,
                return_direct=False
            ),
            # 工具4：疾病查药物推荐
            Tool(
                name="DiseaseDrugs",
                func=self.query_drug_recommendations,
                description="""
                查询疾病的推荐药物，输入为疾病名称（如"感冒"、"高血压"）。
                适用于用户问“感冒吃什么药”“高血压用什么药物治疗”等场景。
                """,
                return_direct=False
            ),
            # 工具5：疾病查饮食建议
            Tool(
                name="DiseaseDietaryAdvice",
                func=self.query_dietary_advice,
                description="""
                查询疾病的饮食建议（宜吃和忌吃食物），输入为疾病名称（如"糖尿病"、"高血压"）。
                适用于用户问“糖尿病不能吃什么”“高血压患者的饮食注意事项”等场景。
                """,
                return_direct=False
            ),
            # 工具6：通用GraphRAG查询
            Tool(
                name="GeneralMedicalQuery",
                func=self.general_graphrag_query,
                description="""
                执行通用的医疗知识图谱查询，基于GraphRAG技术获取结构化知识。
                参数：question-查询问题（必需），top_k-返回结果数量（可选），relation_types-关系类型（可选）。
                适用于复杂或非常规的医疗知识查询。
                """,
                return_direct=False
            )
        ]


def get_medical_kg_tools():
    """获取医疗知识图谱相关的LangChain工具列表（增强版，支持GraphRAG）"""
    kg_tool = MedicalKGQueryTool()

    tools = [
        Tool(
            name="查询症状相关疾病",
            func=kg_tool.query_symptom_related_diseases,
            description="根据症状查询可能的关联疾病。参数应为症状名称（如'咳嗽'、'头痛'、'发热'）。"
        ),
        Tool(
            name="查询疾病详情",
            func=kg_tool.query_disease_detail,
            description="查询疾病的基本信息，包括病因、治疗方式、治愈率等。参数应为疾病名称（如'肺气肿'、'高血压'）。"
        ),
        Tool(
            name="查询疾病症状",
            func=kg_tool.query_disease_symptoms,
            description="查询某疾病的典型症状。参数应为疾病名称（如'肺气肿'、'高血压'）。"
        ),
        # 新增GraphRAG增强功能工具
        Tool(
            name="查询疾病推荐药物",
            func=kg_tool.query_drug_recommendations,
            description="查询疾病推荐药物。参数应为疾病名称（如'感冒'、'高血压'）。"
        ),
        Tool(
            name="查询疾病饮食建议",
            func=kg_tool.query_dietary_advice,
            description="查询疾病饮食建议（宜吃和忌吃食物）。参数应为疾病名称（如'糖尿病'、'高血压'）。"
        ),
        Tool(
            name="医疗知识图谱查询",
            func=kg_tool.general_graphrag_query,
            description="执行通用的医疗知识图谱查询，基于GraphRAG技术获取结构化知识。参数：question-查询问题，top_k-返回结果数量(可选)，relation_types-关系类型(可选)。"
        )
    ]

    return tools


medical_kg_tool = MedicalKGQueryTool()
langchain_medical_tools = medical_kg_tool.get_langchain_tools()


