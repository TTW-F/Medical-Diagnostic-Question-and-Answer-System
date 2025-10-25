import os
import langchain_core
import requests
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")


# ------------------------------
# 1. 药物禁忌检查工具
# ------------------------------
class DrugContraindicationInput(BaseModel):
    """检查药物禁忌的输入参数（LangChain 1.0+ 要求显式定义 Pydantic 模型）"""
    drug_name: str = Field(
        ...,  # 表示必填
        description="药物名称（通用名或商品名，如 'ibuprofen' 或 '布洛芬'，需准确拼写）"
    )
    patient_age: int = Field(
        ...,
        description="患者年龄（整数，例如 5 表示 5 岁，65 表示 65 岁）"
    )
    patient_diseases: List[str] = Field(
        ...,
        description="患者基础疾病列表，如 ['高血压', '哮喘']，无疾病则传空列表 []"
    )
    patient_allergies: List[str] = Field(
        ...,
        description="患者过敏史列表，如 ['青霉素', '阿司匹林']，无过敏则传空列表 []"
    )


@tool(
    # LangChain 1.0+ 中，name 和 description 是推荐参数（增强工具可解释性）
    name="check_drug_contraindication",
    description="通过 FDA 数据库检查药物对特定患者的禁忌，包括年龄限制、基础疾病禁忌和过敏禁忌",
    args_schema=DrugContraindicationInput  # 绑定输入模型（1.0+ 强制参数校验）
)
def check_drug_contraindication(
        drug_name: str,
        patient_age: int,
        patient_diseases: List[str],
        patient_allergies: List[str]
) -> Dict[str, str]:
    """核心逻辑：调用 FDA API 检查药物禁忌并返回结构化结果"""
    base_url = "https://api.fda.gov/drug/label.json"
    params = {
        "search": f"openfda.generic_name:{drug_name} OR openfda.brand_name:{drug_name}",
        "limit": 1,
        "api_key": OPENFDA_API_KEY
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # 自动处理 HTTP 错误（如 404、500）
        data = response.json()

        if not data.get("results"):
            return {"status": "error", "message": f"未查询到药物 '{drug_name}' 的 FDA 记录，请检查名称拼写"}

        drug_data = data["results"][0]
        contraindications = drug_data.get("contraindications", ["无明确禁忌说明"])
        age_restrictions = drug_data.get("pediatric_use", ["无年龄限制说明"])
        allergy_info = drug_data.get("allergies", ["无过敏提示"])

        # 分析禁忌匹配结果
        issues = []
        # 年龄禁忌检查
        if any("儿童禁用" in str(restriction) for restriction in age_restrictions) and patient_age < 12:
            issues.append(f"年龄禁忌：{drug_name} 禁用于 12 岁以下儿童")
        # 基础疾病禁忌检查
        for disease in patient_diseases:
            if any(disease.lower() in str(ct).lower() for ct in contraindications):
                issues.append(f"疾病禁忌：患者患有 {disease}，药物明确标注相关禁忌")
        # 过敏禁忌检查
        for allergy in patient_allergies:
            if any(allergy.lower() in str(ai).lower() for ai in allergy_info):
                issues.append(f"过敏禁忌：患者对 {allergy} 过敏，药物含相关成分")

        return {
            "status": "success",
            "drug_name": drug_name,
            "patient_info": f"年龄 {patient_age} 岁，基础疾病 {patient_diseases}，过敏史 {patient_allergies}",
            "contraindication_results": "；".join(issues) if issues else "未发现明确禁忌",
            "fda_reference": f"https://pubmed.ncbi.nlm.nih.gov/?term={drug_name}+contraindications"
        }

    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"API 调用失败：{str(e)}"}


# ------------------------------
# 2. 诊疗指南检索工具（基于 PubMed）
# ------------------------------
class GuidelineRetrievalInput(BaseModel):
    """检索诊疗指南的输入参数"""
    disease_name: str = Field(..., description="疾病名称（如 '2型糖尿病' '肺癌'，需准确）")
    query_dimension: str = Field(
        ...,
        description="检索维度，如 '诊断标准' '治疗方案' '鉴别诊断'（需明确具体方向）"
    )
    max_results: int = Field(
        3,
        description="最大返回结果数（1-10，默认 3，避免结果过多）"
    )


@tool(
    name="retrieve_clinical_guidelines",
    description="通过 PubMed 检索权威诊疗指南，获取疾病的诊断/治疗依据",
    args_schema=GuidelineRetrievalInput
)
def retrieve_clinical_guidelines(
        disease_name: str,
        query_dimension: str,
        max_results: int = 3
) -> Dict[str, List[Dict]]:
    """核心逻辑：调用 PubMed 检索诊疗指南"""
    from langchain_community.document_loaders import PubMedLoader  # 延迟导入（1.0+ 推荐）

    # 构建检索关键词（PubMed 优先支持英文）
    query = f"{disease_name} AND {query_dimension} AND clinical practice guideline"
    loader = PubMedLoader(query=query, load_max_docs=max_results)

    try:
        documents = loader.load()
        guidelines = []
        for doc in documents:
            guidelines.append({
                "title": doc.metadata.get("title", "未知标题"),
                "pmid": doc.metadata.get("pmid", "未知ID"),
                "publication_date": doc.metadata.get("publication_date", "未知日期"),
                "summary": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{doc.metadata.get('pmid', '')}/"
            })
        return {
            "status": "success",
            "disease": disease_name,
            "dimension": query_dimension,
            "guidelines": guidelines or ["未检索到相关指南"]
        }
    except Exception as e:
        return {"status": "error", "message": f"指南检索失败：{str(e)}"}


# ------------------------------
# 工具列表（供 Agent 调用）
# ------------------------------
MEDICAL_TOOLS = [
    check_drug_contraindication,
    retrieve_clinical_guidelines
]

# 转换为 OpenAI 函数格式（LangChain 1.0+ 与 LLM 交互的标准格式）
MEDICAL_FUNCTIONS = [convert_to_openai_function(tool) for tool in MEDICAL_TOOLS]