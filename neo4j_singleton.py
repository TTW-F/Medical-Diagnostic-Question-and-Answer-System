import os
from neo4j import GraphDatabase, basic_auth, exceptions
import dotenv
from typing import Optional, Dict, List

# 加载环境变量
dotenv.load_dotenv()

class Neo4jSingleton:
    _instance: Optional[GraphDatabase.driver] = None

    @classmethod
    def get_instance(cls) -> GraphDatabase.driver:
        """单例模式创建 Neo4j 驱动，确保仅初始化一次"""
        if cls._instance is None:
            # 从环境变量读取配置（避免硬编码）
            uri = os.getenv("DB_URI")
            user = os.getenv("DB_ROOT")
            password = os.getenv("DB_PASSWORD")

            # 初始化驱动并验证连接
            try:
                cls._instance = GraphDatabase.driver(
                    uri,
                    auth=basic_auth(user, password),
                    max_connection_lifetime=30  # 连接超时时间（秒）
                )
                # 验证连接有效性
                cls._instance.verify_connectivity()
                print("✅ Neo4j 连接初始化成功")
            except exceptions.AuthError:
                raise ConnectionError("Neo4j 认证失败：用户名或密码错误")
            except exceptions.ServiceUnavailable:
                raise ConnectionError("Neo4j 服务不可用：检查地址、端口或服务状态")
            except Exception as e:
                raise ConnectionError(f"Neo4j 连接异常：{str(e)}")
        return cls._instance

    @classmethod
    def close_instance(cls):
        """程序退出时关闭连接池"""
        if cls._instance:
            cls._instance.close()
            print("✅ Neo4j 连接已关闭")

    @classmethod
    def run_query(cls, query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        执行 Cypher 查询并返回格式化结果
        Args:
            query: Cypher 语句（推荐用参数化查询避免注入）
            parameters: 查询参数（如 {"symptoms": ["咳嗽", "发烧"]}）
        Returns:
            字典列表，每个字典对应一条查询结果
        """
        driver = cls.get_instance()
        parameters = parameters or {}
        try:
            with driver.session() as session:
                # 执行查询并转换结果格式
                result = session.run(query, parameters)
                return [record.data() for record in result]
        except Exception as e:
            print(f"❌ Cypher 查询失败：{str(e)}")
            return [{"error": f"查询失败：{str(e)}"}]