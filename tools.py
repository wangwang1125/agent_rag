# 工具模块，包含身体异常分析系统使用的工具函数
import json
import os
import re
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank

# 配置嵌入模型
EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key="sk-51d30a5436ca433b8ff81e624a23dcac",
)
Settings.embed_model = EMBED_MODEL

class MedicalAnalysis:
    """身体异常分析工具类"""
    
    @staticmethod
    def initialize_structured_data(structured_data):
        """直接初始化结构化身体数据，跳过agent分析流程，仿照UserDataAnalysisAssistant的输出格式"""
        result = {
            "user_info": {},
            "body_composition": {},
            "posture_metrics": {},
            "posture_conclusion": {},
            "girth_info": {},
            "data_categories": [],
        }
        try:
            # 解析结构化数据
            if isinstance(structured_data, str):
                if structured_data.strip().startswith('{'):
                    data = json.loads(structured_data)
                else:
                    # 如果是文本描述，创建一个默认结构
                    data = {"raw_text": structured_data}
            else:
                data = structured_data
            
            user_info = {}
            body_composition = {}
            posture_metrics = {}
            girth_info = {}
            posture_conclusion = {}
            
            # 解析用户基本信息
            if "user_info" in data:
                user_info["身高"] = data["user_info"].get("height", "")
                user_info["年龄"] = data["user_info"].get("age", "")
                user_info["性别"] = "男性" if data["user_info"].get("sex") == 1 else "女性" if data["user_info"].get("sex") == 0 or data["user_info"].get("sex") == 2 else "未知"

            # 解析体成分数据
            if "mass_info" in data:
                mass_info = data["mass_info"]
                for key, value in mass_info.items():
                    # 提取关键指标到extracted_metrics
                    if key == "BMI":
                        body_composition["BMI"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "PBF":
                        body_composition["体脂率"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "FFM":
                        body_composition["去脂体重"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "WT":
                        body_composition["体重"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "TBW":
                        body_composition["去脂体重"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "BMR":
                        body_composition["基础代谢率"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "WHR":
                        body_composition["腰臀比"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "SM":
                        body_composition["肌肉量"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "PROTEIN":
                        body_composition["蛋白质"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "ICW":
                        body_composition["细胞内液"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "ECW":
                        body_composition["细胞外液"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    elif key == "VAG":
                        body_composition["总水分"] = {"value": value["v"], "low": value.get("l", ""), "medium": value.get("m", ""), "high": value.get("h", "")}
                    
                #根据身高和体重计算去脂体重指数
                if "身高" in user_info and "去脂体重" in body_composition:
                    body_composition["去脂体重指数"] = body_composition["去脂体重"]["value"] / (user_info["身高"]/100) ** 2

        
            # 解析体态测量数据
            if "eval_info" in data:
                eval_info = data["eval_info"]
                posture_metrics = {
                    "左腿型测量值": eval_info.get("left_leg_xo", 0),
                    "右腿型测量值": eval_info.get("right_leg_xo", 0),
                    "骨盆前移测量值": eval_info.get("pelvis_forward", 0),
                    "左膝关节测量值": eval_info.get("left_knee_check", 0),
                    "右膝关节测量值": eval_info.get("right_knee_check", 0)
                }
                        
            # 解析体态结论数据
            if "eval_conclusion" in data:
                eval_conclusion = data["eval_conclusion"]
                
                # 定义中文映射，便于理解
                conclusion_mapping = {
                    "high_low_shoulder": "高低肩结论",
                    "head_forward": "头前引结论", 
                    "head_slant": "头侧歪结论",
                    "round_shoulder_left": "左圆肩结论",
                    "round_shoulder_right": "右圆肩结论"
                }
                
                for eng_key, chn_key in conclusion_mapping.items():
                    if eng_key in eval_conclusion:
                        conclusion_data = eval_conclusion[eng_key]
                        if isinstance(conclusion_data, dict):
                            # 提取结论键值
                            conclusion_key = conclusion_data.get("conclusion_key", "")
                            
                            # 将结论添加到posture_conclusion
                            if conclusion_key:
                                posture_conclusion[chn_key] = conclusion_key
                            
                # 处理不同类型体态结论的键值，转换为更易理解的中文描述
                for key, value in posture_conclusion.items():
                    if key.endswith("结论") and isinstance(value, str):
                        value_lower = value.lower()
                        
                        # 高低肩、头侧歪：偏左/偏右/正常
                        if key in ["高低肩结论", "头侧歪结论"]:
                            if "normal" in value_lower:
                                posture_conclusion[key] = "正常"
                            elif "left" in value_lower:
                                posture_conclusion[key] = "偏左"
                            elif "right" in value_lower:
                                posture_conclusion[key] = "偏右"
                            else:
                                # 保持原值，可能包含其他有用信息
                                posture_conclusion[key] = value
                        elif key in ["左圆肩结论", "右圆肩结论"]:
                            if "normal" in value_lower:
                                posture_conclusion[key] = "正常"
                            elif "left" in value_lower:
                                posture_conclusion[key] = "存在左圆肩"
                            elif "right" in value_lower:
                                posture_conclusion[key] = "存在右圆肩"
                            else:
                                # 保持原值，可能包含其他有用信息
                                posture_conclusion[key] = value
                        # 头前引：有异常/无异常
                        elif key in ["头前引结论"]:
                            if "normal" in value_lower:
                                posture_conclusion[key] = "无异常"
                            elif "headforward.head" in value_lower or "head" in value_lower:
                                posture_conclusion[key] = "存在异常"
                            else:
                                posture_conclusion[key] = value
                        # 其他体态结论的通用处理
                        else:
                            if "normal" in value_lower:
                                posture_conclusion[key] = "正常"
                            else:
                                # 保持原值，让用户看到原始结论键
                                posture_conclusion[key] = value

            # 解析围度信息
            if "girth_info" in data:
                for key, value in data["girth_info"].items():
                    if value:
                        girth_info[f"围度_{key}"] = value
            
            result["user_info"] = user_info
            result["body_composition"] = body_composition
            result["posture_metrics"] = posture_metrics
            result["posture_conclusion"] = posture_conclusion
            result["girth_info"] = girth_info
            
            # 数据分类
            data_categories = []
            if body_composition:
                data_categories.append("体成分数据")
            if posture_metrics:
                data_categories.append("体态测量数据")
            if posture_conclusion:
                data_categories.append("体态结论数据")
            if any(k.startswith("围度_") for k in girth_info.keys()):
                data_categories.append("围度数据")
            if any(k in user_info for k in ["身高", "年龄", "性别"]):
                data_categories.append("基本信息")
            
            result["data_categories"] = data_categories
            
            
        except Exception as e:
            result["error"] = f"初始化结构化数据时出错：{str(e)}"
            result["analysis_summary"] = "数据初始化失败"
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @staticmethod
    def query_medical_knowledge(query_text, knowledge_base_name=None):
        """查询身体异常判断决策树知识库"""
        DB_PATH = "VectorStore"
        
        try:
            # 如果没有指定知识库，尝试找到一个可用的
            if not knowledge_base_name or knowledge_base_name == "None":
                available_dbs = os.listdir(DB_PATH) if os.path.exists(DB_PATH) else []
                if available_dbs:
                    knowledge_base_name = available_dbs[0]
                else:
                    return json.dumps({"error": "未找到可用的决策树知识库"}, ensure_ascii=False)
            
            # 检查知识库是否存在
            print(f"查询知识库：{knowledge_base_name}")
            db_path = os.path.join(DB_PATH, knowledge_base_name)
            if not os.path.exists(db_path):
                return json.dumps({"error": f"决策树知识库 {knowledge_base_name} 不存在"}, ensure_ascii=False)
            
            
            # 配置重排序
            dashscope_rerank = DashScopeRerank(
                top_n=10,  # 增加重排序数量以获取更多决策规则
                return_documents=True,
                api_key="sk-51d30a5436ca433b8ff81e624a23dcac"
            )
            
            # 加载知识库
            storage_context = StorageContext.from_defaults(persist_dir=db_path)
            index = load_index_from_storage(storage_context)
            
            # 检索相关文档
            retriever_engine = index.as_retriever(similarity_top_k=25)
            retrieve_chunk = retriever_engine.retrieve(query_text)
            
            # 重排序
            try:
                results = dashscope_rerank.postprocess_nodes(retrieve_chunk, query_str=query_text)
                print(f"决策树重排序成功，获得{len(results)}个结果")
            except Exception as e:
                print(f"决策树重排序失败: {e}，使用原始检索结果")
                results = retrieve_chunk[:10]
            
            # 整理检索结果 - 专注于决策规则
            decision_rules = []
            for i, result in enumerate(results):
                score = result.score
                # 对于决策树，降低阈值以获取更多规则
                if score >= 0.05 or i < 5:
                    # 将中文括号【】替换为英文括号[]，便于后续解析
                    clean_text = result.text.replace('【', '[').replace('】', ']')
                    clean_content = clean_text[:600] + "..." if len(clean_text) > 600 else clean_text
                    
                    decision_rules.append({
                        "rule_id": i + 1,
                        "content": clean_content,
                        "confidence_score": round(score, 3),
                        "full_content": clean_text
                    })
            
            return json.dumps({
                "query": query_text,
                "original_query": query_text,
                "knowledge_base": knowledge_base_name,
                "retrieved_chunks": decision_rules,
                "total_rules": len(decision_rules),
                "query_type": "身体异常决策树查询"
            }, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return json.dumps({
                "error": f"查询决策树知识库时出错：{str(e)}",
                "query": query_text
            }, ensure_ascii=False)