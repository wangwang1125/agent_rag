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
    def analyze_symptoms(user_data):
        """分析用户提供的身体数据"""
        # 对用户提供的身体数据进行解析和标准化
        result = {
            "body_data_analysis": {
                "input_data": user_data,
                "extracted_metrics": {},
                "data_categories": [],
                "analysis_summary": f"接收到身体数据：{user_data}。正在进行数据解析和分析..."
            }
        }
        
        # 解析身体数据
        data_lower = user_data.lower()
        extracted_metrics = {}
        
        # 提取性别信息
        if "男性" in data_lower or "男" in data_lower:
            extracted_metrics["性别"] = "男性"
        elif "女性" in data_lower or "女" in data_lower:
            extracted_metrics["性别"] = "女性"
        
        # 提取BMI值
        bmi_pattern = r"bmi[\s=：]*(\d+\.?\d*)"
        bmi_match = re.search(bmi_pattern, data_lower)
        if bmi_match:
            extracted_metrics["BMI"] = float(bmi_match.group(1))
        
        # 提取去脂体重指数
        ffmi_patterns = [
            r"去脂体重指数[\s=：为]*(\d+\.?\d*)",
            r"ffmi[\s=：为]*(\d+\.?\d*)",
            r"瘦体重指数[\s=：为]*(\d+\.?\d*)"
        ]
        for pattern in ffmi_patterns:
            ffmi_match = re.search(pattern, data_lower)
            if ffmi_match:
                extracted_metrics["去脂体重指数"] = float(ffmi_match.group(1))
                break
        
        # 提取体重
        weight_pattern = r"体重[\s=：]*(\d+\.?\d*)"
        weight_match = re.search(weight_pattern, data_lower)
        if weight_match:
            extracted_metrics["体重"] = float(weight_match.group(1))
        
        # 提取身高
        height_patterns = [
            r"身高[\s=：]*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s*cm",
            r"(\d+\.?\d*)\s*厘米"
        ]
        for pattern in height_patterns:
            height_match = re.search(pattern, data_lower)
            if height_match:
                extracted_metrics["身高"] = float(height_match.group(1))
                break
        
        # 提取年龄
        age_pattern = r"年龄[\s=：]*(\d+)"
        age_match = re.search(age_pattern, data_lower)
        if age_match:
            extracted_metrics["年龄"] = int(age_match.group(1))
        
        result["body_data_analysis"]["extracted_metrics"] = extracted_metrics
        
        # 数据分类
        data_categories = []
        if "BMI" in extracted_metrics:
            data_categories.append("体重指数数据")
        if "去脂体重指数" in extracted_metrics:
            data_categories.append("体成分数据")
        if "性别" in extracted_metrics:
            data_categories.append("基本信息")
        if "年龄" in extracted_metrics:
            data_categories.append("年龄信息")
        
        result["body_data_analysis"]["data_categories"] = data_categories
        result["body_data_analysis"]["analysis_summary"] = f"成功解析出{len(extracted_metrics)}项身体指标：{list(extracted_metrics.keys())}"
        
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
            db_path = os.path.join(DB_PATH, knowledge_base_name)
            if not os.path.exists(db_path):
                return json.dumps({"error": f"决策树知识库 {knowledge_base_name} 不存在"}, ensure_ascii=False)
            
            # 优化查询文本 - 提取关键身体指标词
            optimized_query = MedicalAnalysis._optimize_body_query(query_text)
            
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
            retrieve_chunk = retriever_engine.retrieve(optimized_query)
            
            # 重排序
            try:
                results = dashscope_rerank.postprocess_nodes(retrieve_chunk, query_str=optimized_query)
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
                    decision_rules.append({
                        "rule_id": i + 1,
                        "content": result.text[:600] + "..." if len(result.text) > 600 else result.text,
                        "confidence_score": round(score, 3),
                        "full_content": result.text
                    })
            
            return json.dumps({
                "query": optimized_query,
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
    
    @staticmethod
    def _optimize_body_query(query_text):
        """优化查询文本，提取关键身体指标词汇"""
        # 身体指标关键词映射
        body_keywords = {
            "BMI": ["bmi", "体重指数", "体质指数"],
            "去脂体重指数": ["去脂体重指数", "ffmi", "瘦体重指数", "肌肉指数"],
            "体重": ["体重", "重量", "kg"],
            "身高": ["身高", "cm", "厘米"],
            "体脂率": ["体脂率"],
            "男性": ["男性", "男", "male"],
            "女性": ["女性", "女", "female"]
        }
        
        # 提取查询文本中的关键指标
        extracted_terms = []
        query_lower = query_text.lower()
        
        for main_term, variations in body_keywords.items():
            if any(var in query_lower for var in variations):
                extracted_terms.append(main_term)
        
        # 提取数值信息
        numbers = re.findall(r'\d+\.?\d*', query_text)
        
        # 构建优化查询
        if extracted_terms:
            optimized = " ".join(extracted_terms) + " " + query_text
            if numbers:
                optimized += " " + " ".join(numbers)
        else:
            optimized = query_text
        
        return optimized

class HealthAssessment:
    """身体异常评估工具类"""
    
    @staticmethod
    def comprehensive_analysis(user_data, decision_rules):
        """基于用户身体数据和决策树规则进行综合异常分析"""
        result = {
            "abnormality_assessment": {
                "user_data_input": user_data,
                "decision_rules_applied": decision_rules,
                "extracted_user_metrics": {},
                "applied_decision_rules": [],
                "decision_process": [],
                "abnormality_findings": [],
                "risk_level": "未知",
                "specific_conclusions": [],
                "recommendations": [],
                "disclaimer": "此分析结果严格基于知识库决策树规则，仅供参考，建议咨询专业的健康顾问或医生获取准确诊断"
            }
        }
        
        try:
            # 解析用户数据
            user_data_parsed = json.loads(user_data) if isinstance(user_data, str) else user_data
            extracted_metrics = {}
            
            if isinstance(user_data_parsed, dict) and "body_data_analysis" in user_data_parsed:
                extracted_metrics = user_data_parsed["body_data_analysis"].get("extracted_metrics", {})
            
            result["abnormality_assessment"]["extracted_user_metrics"] = extracted_metrics
            
            # 解析决策规则
            decision_data = json.loads(decision_rules) if isinstance(decision_rules, str) else decision_rules
            knowledge_rules = []
            
            if isinstance(decision_data, dict) and "retrieved_chunks" in decision_data:
                knowledge_rules = decision_data["retrieved_chunks"]
            
            # 解析知识库中的结构化决策规则
            parsed_rules = []
            for rule in knowledge_rules:
                content = rule.get("full_content", rule.get("content", ""))
                parsed_rule = HealthAssessment._parse_decision_rule(content)
                if parsed_rule:
                    parsed_rule["confidence_score"] = rule.get("confidence_score", 0.5)
                    parsed_rule["rule_id"] = rule.get("rule_id", "unknown")
                    parsed_rules.append(parsed_rule)
            
            # 按置信度排序规则
            parsed_rules.sort(key=lambda x: x["confidence_score"], reverse=True)
            
            # 应用决策树规则进行判断
            applied_rules = []
            decision_process = []
            abnormality_findings = []
            specific_conclusions = []
            recommendations = []
            
            for rule in parsed_rules:
                # 检查规则是否适用于当前用户数据
                is_applicable, match_result = HealthAssessment._check_rule_applicability(extracted_metrics, rule)
                
                if is_applicable:
                    applied_rules.append(rule)
                    
                    # 记录决策过程
                    process_step = {
                        "rule_id": rule["rule_id"],
                        "condition": rule.get("condition", ""),
                        "user_values": match_result,
                        "result": rule.get("conclusion", ""),
                        "confidence": rule["confidence_score"]
                    }
                    decision_process.append(process_step)
                    
                    # 添加异常发现
                    if rule.get("conclusion"):
                        abnormality_findings.append(rule["conclusion"])
                        specific_conclusions.append(
                            f"根据决策规则：{rule.get('condition', '')} -> 结论：{rule['conclusion']} "
                            f"(用户数据匹配：{match_result})"
                        )
                    
                    # 添加建议
                    if rule.get("solution"):
                        recommendations.append(f"针对{rule['conclusion']}：{rule['solution']}")
                    
                    if rule.get("recommendation"):
                        recommendations.append(rule["recommendation"])
            
            # 如果没有匹配的规则，表示知识库中没有相关的决策规则
            if not applied_rules:
                decision_process.append({
                    "rule_id": "无匹配规则",
                    "condition": "未在知识库中找到适用的决策规则",
                    "user_values": extracted_metrics,
                    "result": "无法基于知识库规则进行判断",
                    "confidence": 0.0
                })
                
                specific_conclusions.append("知识库中未找到适用于当前用户数据的决策规则，无法进行异常分析")
                risk_level = "无法评估"
            else:
                # 完全基于知识库规则的判断结果
                # 根据知识库中规则的优先级来确定风险等级
                if applied_rules:
                    # 获取最高优先级（数字越小优先级越高）
                    priorities = [rule.get("priority", 5) for rule in applied_rules if rule.get("priority")]
                    if priorities:
                        highest_priority = min(priorities)
                        if highest_priority == 1:
                            risk_level = "高风险"
                        elif highest_priority == 2:
                            risk_level = "中高风险"
                        elif highest_priority == 3:
                            risk_level = "中风险"
                        else:
                            risk_level = "低风险"
                    else:
                        # 如果没有优先级信息，根据异常发现数量判断
                        if len(abnormality_findings) >= 2:
                            risk_level = "高风险"
                        elif len(abnormality_findings) == 1:
                            risk_level = "中风险"
                        else:
                            risk_level = "低风险"
                else:
                    risk_level = "低风险"
                
                # 如果没有异常发现，表示指标正常
                if not abnormality_findings:
                    specific_conclusions.append("基于知识库决策规则分析，当前用户数据未匹配到任何异常条件，各项指标在正常范围内")
                    risk_level = "低风险"
            
            result["abnormality_assessment"]["applied_decision_rules"] = applied_rules
            result["abnormality_assessment"]["decision_process"] = decision_process
            result["abnormality_assessment"]["abnormality_findings"] = abnormality_findings
            result["abnormality_assessment"]["risk_level"] = risk_level
            result["abnormality_assessment"]["specific_conclusions"] = specific_conclusions
            result["abnormality_assessment"]["recommendations"] = list(set(recommendations))
            
        except Exception as e:
            result["abnormality_assessment"]["error"] = f"分析过程中出现错误：{str(e)}"
            result["abnormality_assessment"]["specific_conclusions"] = ["分析过程中出现错误，请检查输入数据格式"]
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _parse_decision_rule(rule_content):
        """解析知识库中的决策规则"""
        try:
            # 解析结构化的决策规则
            parsed = {}
            
            # 提取异常结论
            print("rule_content",rule_content)
            conclusion_match = re.search(r"异常结论[:：]([^,，]+)", rule_content)
            if conclusion_match:
                parsed["conclusion"] = conclusion_match.group(1).strip()
            
            # 提取判断流程/条件
            condition_patterns = [
                r"异常判断流程[:：]([^,，]+)",
                r"判断条件[^：:]*[:：]([^,，]+)",
                r"BMI\s*[<>≤≥]\s*[\d.]+[^,，]*"
            ]
            
            for pattern in condition_patterns:
                condition_match = re.search(pattern, rule_content)
                if condition_match:
                    parsed["condition"] = condition_match.group(1).strip() if ":" in pattern else condition_match.group(0).strip()
                    break
            
            # 提取关键解决点
            solution_match = re.search(r"关键解决点[^：:]*[:：]([^,，]+)", rule_content)
            if solution_match:
                parsed["solution"] = solution_match.group(1).strip()
            
            # 提取建议
            recommendation_match = re.search(r"建议[^：:]*[:：]([^,，]+)", rule_content)
            if recommendation_match:
                parsed["recommendation"] = recommendation_match.group(1).strip()
            
            # 提取优先级
            priority_match = re.search(r"优先级[:：](\d+)", rule_content)
            if priority_match:
                parsed["priority"] = int(priority_match.group(1))
            
            return parsed if parsed else None
            
        except Exception as e:
            print(f"解析决策规则失败: {e}")
            return None
    
    @staticmethod
    def _check_rule_applicability(user_metrics, rule):
        """检查决策规则是否适用于用户数据"""
        try:
            condition = rule.get("condition", "")
            if not condition:
                return False, {}
            
            user_bmi = user_metrics.get("BMI")
            user_ffmi = user_metrics.get("去脂体重指数")
            user_gender = user_metrics.get("性别", "")
            
            match_result = {}
            
            # 检查BMI条件
            bmi_conditions = re.findall(r"BMI\s*([<>≤≥])\s*([\d.]+)", condition)
            for operator, threshold in bmi_conditions:
                if user_bmi is not None:
                    threshold = float(threshold)
                    match_result[f"BMI{operator}{threshold}"] = user_bmi
                    
                    if operator in ["<", "＜"]:
                        if not (user_bmi < threshold):
                            return False, match_result
                    elif operator in [">", "＞"]:
                        if not (user_bmi > threshold):
                            return False, match_result
                    elif operator in ["≤", "<=", "≦"]:
                        if not (user_bmi <= threshold):
                            return False, match_result
                    elif operator in ["≥", ">=", "≧"]:
                        if not (user_bmi >= threshold):
                            return False, match_result
                else:
                    return False, match_result
            
            # 检查去脂体重指数条件
            ffmi_patterns = [
                r"(男性|女性)去脂体重指数\s*([<>≤≥])\s*([\d.]+)",
                r"去脂体重指数[^）]*([<>≤≥])\s*([\d.]+)"
            ]
            
            for pattern in ffmi_patterns:
                ffmi_matches = re.findall(pattern, condition)
                for match in ffmi_matches:
                    if len(match) == 3:  # 包含性别的匹配
                        gender, operator, threshold = match
                        if user_gender and gender in user_gender and user_ffmi is not None:
                            threshold = float(threshold)
                            match_result[f"{gender}去脂体重指数{operator}{threshold}"] = user_ffmi
                            
                            if operator in ["<", "＜"]:
                                if not (user_ffmi < threshold):
                                    return False, match_result
                            elif operator in [">", "＞"]:
                                if not (user_ffmi > threshold):
                                    return False, match_result
                            elif operator in ["≤", "<=", "≦"]:
                                if not (user_ffmi <= threshold):
                                    return False, match_result
                            elif operator in ["≥", ">=", "≧"]:
                                if not (user_ffmi >= threshold):
                                    return False, match_result
                    elif len(match) == 2:  # 不包含性别的匹配
                        operator, threshold = match
                        if user_ffmi is not None:
                            threshold = float(threshold)
                            match_result[f"去脂体重指数{operator}{threshold}"] = user_ffmi
                            
                            if operator in ["<", "＜"]:
                                if not (user_ffmi < threshold):
                                    return False, match_result
                            elif operator in ["≥", ">=", "≧"]:
                                if not (user_ffmi >= threshold):
                                    return False, match_result
            
            # 如果所有条件都满足，返回True
            return len(match_result) > 0, match_result
            
        except Exception as e:
            print(f"检查规则适用性失败: {e}")
            return False, {} 