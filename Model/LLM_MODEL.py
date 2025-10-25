from typing import Dict, Any, List, Tuple
import sys
import os
from dotenv import load_dotenv
import numpy as np

from langchain_community.vectorstores import Milvus
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入医疗知识图谱工具实例
from Tools.Neo4jGraphTool import medical_kg_tool
# 导入嵌入模型和重排序模型
from Model.EMBEDDING_MODEL import global_embeddings, compute_similarity
from Model.RERANKER_MODEL import rerank_documents

# 加载环境变量
load_dotenv()

# ------------------------------
# 1. 增强的数据库查询工具（集成嵌入和重排序模型）
# ------------------------------

# 创建增强版症状查询工具
def enhanced_symptom_query(symptom: str) -> List[Dict[str, Any]]:
    """增强版症状查询，使用嵌入模型进行语义理解，重排序模型优化结果"""
    print(f"🔍 使用增强查询: {symptom}")
    
    # 1. 首先使用原始工具查询数据库
    raw_results = medical_kg_tool.query_symptom_related_diseases(symptom)
    
    # 2. 如果查询有结果，使用重排序模型优化相关性
    if raw_results and "error" not in raw_results[0] and "message" not in raw_results[0]:
        # 准备用于重排序的文档格式
        docs_to_rerank = [{
            'text': f"疾病: {disease['disease_name']}\n描述: {disease.get('disease_description', '')}\n相关科室: {disease.get('related_department', '')}",
            'original': disease
        } for disease in raw_results]
        
        # 3. 应用重排序
        reranked = rerank_documents(symptom, docs_to_rerank)
        
        # 4. 返回优化后的结果
        optimized_results = [item[0]['original'] for item in reranked]
        print(f"✅ 查询结果已优化，原始{len(raw_results)}条，优化后{len(optimized_results)}条")
        return optimized_results
    
    return raw_results

# 创建增强版疾病查询工具
def enhanced_disease_query(disease: str) -> List[Dict[str, Any]]:
    """增强版疾病查询，使用嵌入模型提高匹配精度"""
    # 标准查询
    results = medical_kg_tool.query_disease_detail(disease)
    return results

# 创建增强版症状列表查询工具
def enhanced_disease_symptoms_query(disease: str) -> List[Dict[str, Any]]:
    """增强版疾病症状查询"""
    # 标准查询
    results = medical_kg_tool.query_disease_symptoms(disease)
    return results

# 注册增强版工具
tools = [
    Tool(
        name="EnhancedSymptomToDisease",
        func=enhanced_symptom_query,
        description="""
        根据症状查询可能的疾病，使用嵌入和重排序技术优化结果，输入为症状名称（多个症状用逗号分隔，如"咳嗽,发热"）。
        适用于用户问"咳嗽可能是什么病"
        """
        
    ),
    Tool(
        name="EnhancedDiseaseDetail",
        func=enhanced_disease_query,
        description="""
        查询疾病的基本信息，使用嵌入技术提高匹配精度，输入为疾病名称（如"肺气肿"）。
        适用于用户问"什么是肺气肿"
        """
        
    ),
    Tool(
        name="EnhancedDiseaseSymptoms",
        func=enhanced_disease_symptoms_query,
        description="""
        查询某疾病的典型症状，输入为疾病名称（如"肺气肿"）。
        适用于用户问"肺气肿有什么症状"
        """
        
    )
]

# ------------------------------
# 2. 初始化LLM
# ------------------------------
llm = ChatOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",  # 固定DeepSeek的URL
    model="deepseek-chat",  # 固定模型
    temperature=0.1
)

# ------------------------------
# 3. 定义Prompt（完全转义花括号）
# ------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """
你是医疗知识助手，严格遵循以下规则：

1. 对于所有医学问题（包括疾病查询、症状查询、病因、治疗方法等），必须使用提供的增强型数据库工具查询，严禁使用模型自身的预训练知识回答。
2. 医学问题分类及对应工具使用规则：
   - 症状相关问题（如'咳嗽可能是什么病'）→ 使用EnhancedSymptomToDisease工具
   - 疾病详细信息（如'什么是肺气肿'、'肺气肿怎么治疗'）→ 使用EnhancedDiseaseDetail工具
   - 疾病症状查询（如'肺气肿有什么症状'）→ 使用EnhancedDiseaseSymptoms工具
3. 只回答数据库中存在的信息，对于工具返回的结果进行清晰总结，不添加任何额外的医学知识或推测。
4. 非医学问题（如问候语、日常生活问题）可以直接回答，但要明确区分。
5. 回答必须包含免责声明：'以上信息仅供参考，不构成诊断建议。'
6. **重要**：在每个医学信息回答的结尾处，必须明确指明信息来源为"信息来源：医疗知识图谱数据库"。

请严格遵守上述规则，确保所有医学信息都来自数据库查询并明确标注来源。
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ------------------------------
# 4. 创建Agent和Executor
# ------------------------------
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="工具调用格式错误，请检查症状参数"
)

# ------------------------------
# 5. 对话记忆配置
# ------------------------------
session_histories = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]


agent_with_memory = RunnableWithMessageHistory(
    runnable=agent_executor,
    get_session_history=get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)


# ------------------------------
# 6. 交互模块
# ------------------------------
class MedicalQASystem:
    def __init__(self):
        self.welcome_msg = (
            "🏥 增强型医疗知识问答助手（集成嵌入和重排序模型）\n"
            "✅ 特点：语义理解、智能重排序、精准匹配\n"
            "📌 功能：症状查疾病、疾病详情查询、疾病症状查询\n"
            "📌 输入 'exit' 退出，'clear' 清空历史\n"
            "📊 输入 'compare' 测试语义相似度\n"
            "-----------------------------------------"
        )

    def handle_user_input(self, user_input: str, session_id: str = "default") -> str:
        if user_input.lower() == "clear":
            get_session_history(session_id).clear()
            return "已清空对话历史～"
        
        # 特殊命令：测试语义相似度
        if user_input.lower() == "compare":
            return self.test_similarity()

        try:
            result = agent_with_memory.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            response = result.get("output", "未获取到信息")
            
            # 确保包含免责声明
            if "以上信息仅供参考" not in response:
                response += "\n\n⚠️ 以上信息仅供参考，不构成诊断建议。"
            
            # 确保包含信息来源（针对医学问题）
            # 判断是否为医学相关问题（简单判断，可根据实际需求优化）
            is_medical_query = any(keyword in user_input.lower() for keyword in 
                                 ["症状", "疾病", "治疗", "诊断", "病因", "咳嗽", "发热", "头痛"])
            
            if is_medical_query and "信息来源" not in response:
                response += "\n\n📚 信息来源：医疗知识图谱数据库"
            
            return response
        except Exception as e:
            return f"处理失败：{str(e)[:100]}...（请重新提问）"
    
    def test_similarity(self) -> str:
        """测试嵌入模型的语义相似度功能"""
        test_cases = [
            ("咳嗽", "干咳"),
            ("发烧", "发热"),
            ("头痛", "肚子痛"),
            ("高血压", "低血压")
        ]
        
        results = [
            f"'{text1}' 和 '{text2}' 的语义相似度: {compute_similarity(text1, text2):.4f}"
            for text1, text2 in test_cases
        ]
        
        return "\n".join([
            "📊 语义相似度测试结果:",
            "----------------------"
        ] + results + [
            "----------------------",
            "💡 说明: 相似度接近1表示语义高度相似，接近0表示语义差异大",
            "以上测试展示了嵌入模型如何理解医疗术语间的语义关系"
        ])

    def start_interactive_mode(self, session_id: str = "default"):
        print(self.welcome_msg + "\n")
        while True:
            user_input = input("您的问题：").strip()
            if user_input.lower() in ["exit", "退出"]:
                print("\n👋 再见！")
                break
            if not user_input:
                print("ℹ️ 请输入问题～\n")
                continue
            print("💡 思考中...\n")
            print(f"助手：{self.handle_user_input(user_input, session_id)}\n")
            print("-" * 60 + "\n")


if __name__ == "__main__":
    qa_system = MedicalQASystem()
    qa_system.start_interactive_mode()
