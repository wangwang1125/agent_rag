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
        """分析用户提供的结构化身体数据"""
        result = {
            "body_data_analysis": {
                "input_data": user_data,
                "extracted_metrics": {},
                "body_composition": {},
                "posture_metrics": {},
                "data_categories": [],
                "analysis_summary": "正在分析结构化身体数据..."
            }
        }
        
        try:
            # 解析结构化数据
            if isinstance(user_data, str):
                if user_data.strip().startswith('{'):
                    print("处理结构化数据")
                    data = json.loads(user_data)
                else:
                    # 处理文本形式的数据
                    print("处理文本数据")
                    return MedicalAnalysis._analyze_text_data(user_data)
            else:
                data = user_data
            
            extracted_metrics = {}
            body_composition = {}
            posture_metrics = {}
            
            # 解析用户基本信息
            if "user_info" in data:
                user_info = data["user_info"]
                extracted_metrics["身高"] = user_info.get("height", "")
                extracted_metrics["年龄"] = user_info.get("age", "")
                extracted_metrics["性别"] = "男性" if user_info.get("sex") == 1 else "女性" if user_info.get("sex") == 0 or user_info.get("sex") == 2 else "未知"
            
            # 解析体成分数据
            if "mass_info" in data:
                mass_info = data["mass_info"]
                for key, value in mass_info.items():
                    if isinstance(value, dict) and "v" in value:
                        body_composition[key] = {
                            "value": value["v"],
                            "low": value.get("l", ""),
                            "medium": value.get("m", ""),
                            "high": value.get("h", ""),
                            "status": value.get("status", "")
                        }
                        
                        # 提取关键指标到extracted_metrics
                        if key == "BMI":
                            extracted_metrics["BMI"] = value["v"]
                        elif key == "PBF":
                            extracted_metrics["体脂率"] = value["v"]
                        elif key == "FFM":
                            extracted_metrics["去脂体重"] = value["v"]
                        elif key == "WT":
                            extracted_metrics["体重"] = value["v"]
            
            # 解析体态数据
            if "eval_info" in data:
                eval_info = data["eval_info"]
                posture_metrics = {
                    "高低肩": eval_info.get("high_low_shoulder", 0),
                    "头前倾": eval_info.get("head_forward", 0),
                    "头倾斜": eval_info.get("head_slant", 0),
                    "左圆肩": eval_info.get("round_shoulder_left", 0),
                    "右圆肩": eval_info.get("round_shoulder_right", 0),
                    "左腿型": eval_info.get("left_leg_xo", 0),
                    "右腿型": eval_info.get("right_leg_xo", 0)
                }
                
                # 添加到提取指标中
                for key, value in posture_metrics.items():
                    if value and value != 0:
                        extracted_metrics[key] = value
            
            # 解析围度信息
            if "girth_info" in data:
                girth_info = data["girth_info"]
                for key, value in girth_info.items():
                    if value:
                        extracted_metrics[f"围度_{key}"] = value
            
            result["body_data_analysis"]["extracted_metrics"] = extracted_metrics
            result["body_data_analysis"]["body_composition"] = body_composition
            result["body_data_analysis"]["posture_metrics"] = posture_metrics
            
            # 数据分类
            data_categories = []
            if body_composition:
                data_categories.append("体成分数据")
            if posture_metrics:
                data_categories.append("体态数据")
            if "girth_info" in data:
                data_categories.append("围度数据")
            if "user_info" in data:
                data_categories.append("基本信息")
            
            result["body_data_analysis"]["data_categories"] = data_categories
            result["body_data_analysis"]["analysis_summary"] = (
                f"成功解析结构化身体数据，包含：{len(body_composition)}项体成分指标，"
                f"{len([k for k, v in posture_metrics.items() if v != 0])}项体态指标，"
                f"共{len(extracted_metrics)}项关键指标"
            )
            
        except Exception as e:
            result["body_data_analysis"]["error"] = f"解析结构化数据时出错：{str(e)}"
            result["body_data_analysis"]["analysis_summary"] = "数据解析失败"
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    @staticmethod
    def _analyze_text_data(user_data):
        """分析文本形式的身体数据（保持原有逻辑）"""
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
        
        result["body_data_analysis"]["extracted_metrics"] = extracted_metrics
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
    def body_composition_analysis(user_data, decision_rules):
        """基于用户体成分数据和决策树规则进行体成分异常分析"""
        return HealthAssessment._analyze_specific_type(user_data, decision_rules, "体成分异常", 1)
    
    @staticmethod
    def posture_analysis(user_data, decision_rules):
        """基于用户体态数据和决策树规则进行体态异常分析，最多生成4个异常"""
        return HealthAssessment._analyze_specific_type(user_data, decision_rules, "体态异常", 4)
    
    @staticmethod
    def _analyze_specific_type(user_data, decision_rules, analysis_type, max_count):
        """通用的异常分析方法，严格按照知识库优先级排序"""
        result = {
            f"{analysis_type}_assessment": {
                "analysis_type": analysis_type,
                "max_abnormalities": max_count,
                "user_data_input": user_data,
                "decision_rules_applied": decision_rules,
                "extracted_user_metrics": {},
                "parsed_knowledge_rules": [],
                "decision_process_steps": [],
                "abnormality_findings": [],
                "priority_sorted_results": [],
                "final_conclusions": [],
                "detailed_analysis": "",
                "disclaimer": f"此{analysis_type}分析结果严格基于知识库决策树规则，仅供参考，建议咨询专业的健康顾问或医生获取准确诊断"
            }
        }
        
        try:
            # 解析用户数据
            user_data_parsed = json.loads(user_data) if isinstance(user_data, str) else user_data
            extracted_metrics = {}
            
            if isinstance(user_data_parsed, dict) and "body_data_analysis" in user_data_parsed:
                all_metrics = user_data_parsed["body_data_analysis"].get("extracted_metrics", {})
                
                # 根据分析类型筛选相关指标
                if analysis_type == "体成分异常":
                    # 体成分相关指标
                    composition_keys = ["BMI", "体脂率", "去脂体重", "体重", "身高", "年龄", "性别"]
                    extracted_metrics = {k: v for k, v in all_metrics.items() if k in composition_keys}
                elif analysis_type == "体态异常":
                    # 体态相关指标
                    posture_keys = ["高低肩", "头前倾", "头倾斜", "左圆肩", "右圆肩", "左腿X型", "右腿X型"]
                    extracted_metrics = {k: v for k, v in all_metrics.items() if k in posture_keys or "围度" in k}
                    # 也包含基本信息用于体态分析
                    basic_keys = ["身高", "年龄", "性别"]
                    for key in basic_keys:
                        if key in all_metrics:
                            extracted_metrics[key] = all_metrics[key]
            
            result[f"{analysis_type}_assessment"]["extracted_user_metrics"] = extracted_metrics
            
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
                    # 完善规则信息
                    parsed_rule["confidence_score"] = rule.get("confidence_score", 0.5)
                    parsed_rule["rule_id"] = rule.get("rule_id", f"rule_{len(parsed_rules)+1}")
                    parsed_rule["original_content"] = content
                    
                    # 根据分析类型筛选规则
                    rule_content_lower = content.lower()
                    is_relevant = False
                    
                    if analysis_type == "体成分异常":
                        if any(word in rule_content_lower for word in ["bmi", "体脂", "体重", "肥胖", "消瘦", "体成分"]):
                            is_relevant = True
                    elif analysis_type == "体态异常":
                        if any(word in rule_content_lower for word in ["高低肩", "头前倾", "圆肩", "体态", "姿势", "腿型", "头倾斜"]):
                            is_relevant = True
                    
                    if is_relevant:
                        parsed_rules.append(parsed_rule)
            
            # 严格按照知识库优先级排序（优先级数字越小越重要）
            def get_sort_key(rule):
                priority = rule.get("priority", 999)  # 默认最低优先级
                confidence = rule.get("confidence_score", 0.0)
                return (priority, -confidence)  # 先按优先级升序，再按置信度降序
            
            parsed_rules.sort(key=get_sort_key)
            result[f"{analysis_type}_assessment"]["parsed_knowledge_rules"] = parsed_rules
            
            # 应用决策树规则进行判断
            matched_rules = []
            decision_steps = []
            abnormality_findings = []
            priority_results = []
            
            # 限制异常发现的数量
            abnormality_count = 0
            
            for rule in parsed_rules:
                if abnormality_count >= max_count:
                    break
                
                # 检查规则是否适用于当前用户数据
                is_applicable, match_result = HealthAssessment._check_rule_applicability(extracted_metrics, rule)
                
                # 记录决策步骤（无论是否匹配）
                decision_step = {
                    "step": len(decision_steps) + 1,
                    "rule_id": rule["rule_id"],
                    "priority": rule.get("priority", 999),
                    "condition": rule.get("condition", ""),
                    "user_data_values": match_result,
                    "is_match": is_applicable,
                    "conclusion": rule.get("conclusion", "") if is_applicable else "条件不匹配，跳过此规则",
                    "confidence_score": rule["confidence_score"],
                    "detailed_process": ""
                }
                
                if is_applicable:
                    # 构建详细的判断过程描述
                    process_detail = f"【决策流程{len(decision_steps) + 1}】\n"
                    process_detail += f"规则ID: {rule['rule_id']} (优先级: {rule.get('priority', '未设置')})\n"
                    process_detail += f"判断条件: {rule.get('condition', '')}\n"
                    process_detail += f"用户数据匹配情况:\n"
                    
                    for condition_key, user_value in match_result.items():
                        process_detail += f"  - {condition_key}: 用户值={user_value} ✓匹配\n"
                    
                    process_detail += f"匹配结果: 满足条件\n"
                    process_detail += f"异常结论: {rule.get('conclusion', '')}\n"
                    
                    if rule.get("solution"):
                        process_detail += f"解决方案: {rule['solution']}\n"
                    
                    process_detail += f"置信度: {rule['confidence_score']:.2f}\n"
                    
                    decision_step["detailed_process"] = process_detail
                    matched_rules.append(rule)
                    abnormality_count += 1
                    
                    # 添加异常发现
                    if rule.get("conclusion"):
                        finding = {
                            "priority": rule.get("priority", 999),
                            "conclusion": rule["conclusion"],
                            "condition_matched": rule.get("condition", ""),
                            "user_values": match_result,
                            "confidence": rule["confidence_score"],
                            "solution": rule.get("solution", ""),
                            "rule_id": rule["rule_id"]
                        }
                        abnormality_findings.append(finding)
                        priority_results.append(finding)
                else:
                    # 不匹配的情况
                    decision_step["detailed_process"] = f"【决策流程{len(decision_steps) + 1}】\n"
                    decision_step["detailed_process"] += f"规则ID: {rule['rule_id']}\n"
                    decision_step["detailed_process"] += f"判断条件: {rule.get('condition', '')}\n"
                    decision_step["detailed_process"] += f"用户数据: {extracted_metrics}\n"
                    decision_step["detailed_process"] += f"匹配结果: 不满足条件，跳过\n"
                
                decision_steps.append(decision_step)
            
            # 按优先级排序最终结果
            priority_results.sort(key=lambda x: (x["priority"], -x["confidence"]))
            
            # 生成详细分析报告
            detailed_analysis = f"\n=== {analysis_type}详细分析报告 ===\n\n"
            detailed_analysis += f"用户关键指标: {extracted_metrics}\n\n"
            detailed_analysis += f"知识库规则总数: {len(parsed_rules)}\n"
            detailed_analysis += f"匹配规则数量: {len(matched_rules)}\n"
            detailed_analysis += f"发现异常数量: {len(abnormality_findings)}\n\n"
            
            if matched_rules:
                detailed_analysis += "=== 按优先级排序的异常分析 ===\n\n"
                for i, finding in enumerate(priority_results, 1):
                    detailed_analysis += f"{i}. 【优先级 {finding['priority']}】{finding['conclusion']}\n"
                    detailed_analysis += f"   判断依据: {finding['condition_matched']}\n"
                    detailed_analysis += f"   数据匹配: {finding['user_values']}\n"
                    detailed_analysis += f"   置信度: {finding['confidence']:.2f}\n"
                    if finding['solution']:
                        detailed_analysis += f"   建议措施: {finding['solution']}\n"
                    detailed_analysis += "\n"
                
                detailed_analysis += "=== 完整决策流程 ===\n\n"
                for step in decision_steps:
                    if step["is_match"]:
                        detailed_analysis += step["detailed_process"] + "\n"
            else:
                detailed_analysis += "=== 分析结果 ===\n\n"
                detailed_analysis += f"经过{len(parsed_rules)}条决策规则的严格判断，用户的{analysis_type}数据未匹配到任何异常条件。\n"
                detailed_analysis += "当前各项指标在知识库定义的正常范围内。\n\n"
                
                if parsed_rules:
                    detailed_analysis += "=== 检查的决策规则 ===\n\n"
                    for i, rule in enumerate(parsed_rules[:5], 1):  # 显示前5条规则
                        detailed_analysis += f"{i}. 规则ID: {rule['rule_id']}\n"
                        detailed_analysis += f"   条件: {rule.get('condition', '')}\n"
                        detailed_analysis += f"   结论: {rule.get('conclusion', '')}\n"
                        detailed_analysis += f"   用户数据不匹配此条件\n\n"
            print("detailed_analysis",detailed_analysis)
            
            # 生成最终结论
            final_conclusions = []
            if priority_results:
                final_conclusions.append(f"基于知识库决策树规则，发现{len(priority_results)}项{analysis_type}：")
                for finding in priority_results:
                    final_conclusions.append(f"- {finding['conclusion']} (优先级{finding['priority']})")
            else:
                final_conclusions.append(f"基于知识库决策树规则分析，未发现{analysis_type}。")
                final_conclusions.append("用户相关指标均在正常范围内。")
            
            result[f"{analysis_type}_assessment"]["decision_process_steps"] = decision_steps
            result[f"{analysis_type}_assessment"]["abnormality_findings"] = abnormality_findings
            result[f"{analysis_type}_assessment"]["priority_sorted_results"] = priority_results
            result[f"{analysis_type}_assessment"]["final_conclusions"] = final_conclusions
            result[f"{analysis_type}_assessment"]["detailed_analysis"] = detailed_analysis
            
        except Exception as e:
            error_msg = f"分析过程中出现错误：{str(e)}"
            result[f"{analysis_type}_assessment"]["error"] = error_msg
            result[f"{analysis_type}_assessment"]["final_conclusions"] = [f"{analysis_type}分析失败: {error_msg}"]
            result[f"{analysis_type}_assessment"]["detailed_analysis"] = f"错误详情: {error_msg}"
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    
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
        """解析知识库中的决策规则，提取异常结论、判断流程和优先级"""
        try:
            # 解析结构化的决策规则
            parsed = {}
            
            # 提取异常结论 - 支持多种格式
            conclusion_patterns = [
                r"异常结论[:：]\s*([^\n,，]+)",
                r"结论[:：]\s*([^\n,，]+)",
                r"诊断[:：]\s*([^\n,，]+)",
                r"异常[:：]\s*([^\n,，]+)"
            ]
            
            for pattern in conclusion_patterns:
                conclusion_match = re.search(pattern, rule_content)
                if conclusion_match:
                    parsed["conclusion"] = conclusion_match.group(1).strip()
                    break
            
            # 如果没有找到异常结论，尝试从内容中推断
            if "conclusion" not in parsed:
                # 查找可能的异常词汇
                abnormal_keywords = ["肥胖", "消瘦", "过重", "偏瘦", "高低肩", "头前倾", "圆肩", "X型腿", "O型腿"]
                for keyword in abnormal_keywords:
                    if keyword in rule_content:
                        parsed["conclusion"] = keyword
                        break
            
            # 提取判断流程/条件 - 支持多种格式
            condition_patterns = [
                r"异常判断流程[:：]\s*([^\n]+)",
                r"判断流程[:：]\s*([^\n]+)",
                r"判断条件[^：:]*[:：]\s*([^\n]+)",
                r"条件[:：]\s*([^\n]+)",
                r"如果[:：]\s*([^\n]+)",
                r"当[:：]\s*([^\n]+)",
                # BMI相关条件
                r"BMI\s*[<>≤≥]\s*[\d.]+[^\n]*",
                # 体脂率相关条件  
                r"体脂率?\s*[<>≤≥]\s*[\d.]+[%％]?[^\n]*",
                # 去脂体重指数相关条件
                r"去脂体重指数\s*[<>≤≥]\s*[\d.]+[^\n]*",
                r"[男女]性去脂体重指数\s*[<>≤≥]\s*[\d.]+[^\n]*"
            ]
            
            for pattern in condition_patterns:
                condition_match = re.search(pattern, rule_content)
                if condition_match:
                    if ":" in pattern or "：" in pattern:
                        parsed["condition"] = condition_match.group(1).strip()
                    else:
                        parsed["condition"] = condition_match.group(0).strip()
                    break
            
            # 提取关键解决点/解决方案
            solution_patterns = [
                r"关键解决点[^：:]*[:：]\s*([^\n,，]+)",
                r"解决方案[^：:]*[:：]\s*([^\n,，]+)",
                r"解决[^：:]*[:：]\s*([^\n,，]+)",
                r"治疗[^：:]*[:：]\s*([^\n,，]+)"
            ]
            
            for pattern in solution_patterns:
                solution_match = re.search(pattern, rule_content)
                if solution_match:
                    parsed["solution"] = solution_match.group(1).strip()
                    break
            
            # 提取建议
            recommendation_patterns = [
                r"建议[^：:]*[:：]\s*([^\n,，]+)",
                r"推荐[^：:]*[:：]\s*([^\n,，]+)",
                r"注意[^：:]*[:：]\s*([^\n,，]+)"
            ]
            
            for pattern in recommendation_patterns:
                recommendation_match = re.search(pattern, rule_content)
                if recommendation_match:
                    parsed["recommendation"] = recommendation_match.group(1).strip()
                    break
            
            # 提取优先级 - 支持多种格式
            priority_patterns = [
                r"优先级[:：]\s*(\d+)",
                r"级别[:：]\s*(\d+)",
                r"等级[:：]\s*(\d+)",
                r"priority[:：]\s*(\d+)",
                r"level[:：]\s*(\d+)"
            ]
            
            for pattern in priority_patterns:
                priority_match = re.search(pattern, rule_content, re.IGNORECASE)
                if priority_match:
                    parsed["priority"] = int(priority_match.group(1))
                    break
            
            # 如果没有找到优先级，尝试从关键词推断
            if "priority" not in parsed:
                high_priority_keywords = ["严重", "危险", "急需", "立即", "紧急"]
                medium_priority_keywords = ["注意", "建议", "需要", "应该"]
                
                content_lower = rule_content.lower()
                for keyword in high_priority_keywords:
                    if keyword in content_lower:
                        parsed["priority"] = 1  # 高优先级
                        break
                else:
                    for keyword in medium_priority_keywords:
                        if keyword in content_lower:
                            parsed["priority"] = 3  # 中优先级
                            break
                    else:
                        parsed["priority"] = 5  # 默认低优先级
            
            # 提取数值阈值（用于后续条件匹配）
            thresholds = {}
            
            # BMI阈值
            bmi_matches = re.findall(r"BMI\s*([<>≤≥])\s*([\d.]+)", rule_content)
            for operator, value in bmi_matches:
                thresholds[f"BMI_{operator}"] = float(value)
            
            # 体脂率阈值
            body_fat_matches = re.findall(r"体脂率?\s*([<>≤≥])\s*([\d.]+)", rule_content)
            for operator, value in body_fat_matches:
                thresholds[f"体脂率_{operator}"] = float(value)
            
            # 去脂体重指数阈值
            ffmi_matches = re.findall(r"去脂体重指数\s*([<>≤≥])\s*([\d.]+)", rule_content)
            for operator, value in ffmi_matches:
                thresholds[f"去脂体重指数_{operator}"] = float(value)
            
            if thresholds:
                parsed["thresholds"] = thresholds
            
            # 提取适用性别
            if "男性" in rule_content:
                parsed["applicable_gender"] = "男性"
            elif "女性" in rule_content:
                parsed["applicable_gender"] = "女性"
            
            # 记录原始内容用于调试
            parsed["raw_content"] = rule_content
            
            print(f"解析决策规则结果: {parsed}")  # 调试信息
            
            # 只有在至少有结论或条件时才返回解析结果
            if "conclusion" in parsed or "condition" in parsed:
                return parsed
            else:
                print(f"未能解析出有效规则，原始内容：{rule_content[:100]}...")
                return None
            
        except Exception as e:
            print(f"解析决策规则失败: {e}, 内容: {rule_content[:100]}...")
            return None
    
    @staticmethod
    def _check_rule_applicability(user_metrics, rule):
        """检查决策规则是否适用于用户数据，支持体成分和体态指标"""
        try:
            condition = rule.get("condition", "")
            if not condition:
                return False, {}
            
            # 获取用户指标
            user_bmi = user_metrics.get("BMI")
            user_body_fat = user_metrics.get("体脂率")
            user_ffmi = user_metrics.get("去脂体重指数", user_metrics.get("去脂体重"))
            user_weight = user_metrics.get("体重")
            user_height = user_metrics.get("身高")
            user_gender = user_metrics.get("性别", "")
            user_age = user_metrics.get("年龄")
            
            # 体态相关指标
            user_shoulder_imbalance = user_metrics.get("高低肩", 0)
            user_head_forward = user_metrics.get("头前倾", 0)
            user_head_tilt = user_metrics.get("头倾斜", 0)
            user_left_round_shoulder = user_metrics.get("左圆肩", 0)
            user_right_round_shoulder = user_metrics.get("右圆肩", 0)
            user_left_leg_xo = user_metrics.get("左腿X型", 0)
            user_right_leg_xo = user_metrics.get("右腿X型", 0)
            
            match_result = {}
            all_conditions_met = True
            
            # 检查性别适用性
            applicable_gender = rule.get("applicable_gender")
            if applicable_gender and user_gender:
                if applicable_gender not in user_gender:
                    return False, {"gender_mismatch": f"规则适用于{applicable_gender}，用户为{user_gender}"}
            
            # 检查BMI条件
            bmi_conditions = re.findall(r"BMI\s*([<>≤≥])\s*([\d.]+)", condition)
            for operator, threshold in bmi_conditions:
                if user_bmi is not None:
                    threshold = float(threshold)
                    match_result[f"BMI{operator}{threshold}"] = user_bmi
                    
                    condition_met = False
                    if operator in ["<", "＜"]:
                        condition_met = user_bmi < threshold
                    elif operator in [">", "＞"]:
                        condition_met = user_bmi > threshold
                    elif operator in ["≤", "<=", "≦"]:
                        condition_met = user_bmi <= threshold
                    elif operator in ["≥", ">=", "≧"]:
                        condition_met = user_bmi >= threshold
                    
                    if not condition_met:
                        all_conditions_met = False
                        match_result[f"BMI{operator}{threshold}_result"] = "不满足"
                    else:
                        match_result[f"BMI{operator}{threshold}_result"] = "满足"
                else:
                    all_conditions_met = False
                    match_result[f"BMI{operator}{threshold}"] = "用户无BMI数据"
            
            # 检查体脂率条件
            body_fat_conditions = re.findall(r"体脂率?\s*([<>≤≥])\s*([\d.]+)", condition)
            for operator, threshold in body_fat_conditions:
                if user_body_fat is not None:
                    threshold = float(threshold)
                    match_result[f"体脂率{operator}{threshold}"] = user_body_fat
                    
                    condition_met = False
                    if operator in ["<", "＜"]:
                        condition_met = user_body_fat < threshold
                    elif operator in [">", "＞"]:
                        condition_met = user_body_fat > threshold
                    elif operator in ["≤", "<=", "≦"]:
                        condition_met = user_body_fat <= threshold
                    elif operator in ["≥", ">=", "≧"]:
                        condition_met = user_body_fat >= threshold
                    
                    if not condition_met:
                        all_conditions_met = False
                        match_result[f"体脂率{operator}{threshold}_result"] = "不满足"
                    else:
                        match_result[f"体脂率{operator}{threshold}_result"] = "满足"
                else:
                    all_conditions_met = False
                    match_result[f"体脂率{operator}{threshold}"] = "用户无体脂率数据"
            
            # 检查去脂体重指数条件
            ffmi_patterns = [
                r"([男女]性)?去脂体重指数\s*([<>≤≥])\s*([\d.]+)",
                r"去脂体重[^指数]*([<>≤≥])\s*([\d.]+)"
            ]
            
            for pattern in ffmi_patterns:
                ffmi_matches = re.findall(pattern, condition)
                for match in ffmi_matches:
                    if len(match) == 3:  # 包含性别的匹配
                        gender, operator, threshold = match
                        if gender and user_gender and gender in user_gender and user_ffmi is not None:
                            threshold = float(threshold)
                            match_result[f"{gender}去脂体重指数{operator}{threshold}"] = user_ffmi
                            
                            condition_met = HealthAssessment._evaluate_numeric_condition(user_ffmi, operator, threshold)
                            if not condition_met:
                                all_conditions_met = False
                                match_result[f"{gender}去脂体重指数{operator}{threshold}_result"] = "不满足"
                            else:
                                match_result[f"{gender}去脂体重指数{operator}{threshold}_result"] = "满足"
                        elif not user_ffmi:
                            all_conditions_met = False
                            match_result[f"去脂体重指数{operator}{threshold}"] = "用户无去脂体重指数数据"
                    elif len(match) == 2:  # 不包含性别的匹配
                        operator, threshold = match
                        if user_ffmi is not None:
                            threshold = float(threshold)
                            match_result[f"去脂体重指数{operator}{threshold}"] = user_ffmi
                            
                            condition_met = HealthAssessment._evaluate_numeric_condition(user_ffmi, operator, threshold)
                            if not condition_met:
                                all_conditions_met = False
                                match_result[f"去脂体重指数{operator}{threshold}_result"] = "不满足"
                            else:
                                match_result[f"去脂体重指数{operator}{threshold}_result"] = "满足"
                        else:
                            all_conditions_met = False
                            match_result[f"去脂体重指数{operator}{threshold}"] = "用户无去脂体重指数数据"
            
            # 检查体态指标条件
            posture_patterns = [
                (r"高低肩\s*([<>≤≥])\s*([\d.]+)", user_shoulder_imbalance, "高低肩"),
                (r"头前倾\s*([<>≤≥])\s*([\d.]+)", user_head_forward, "头前倾"),
                (r"头倾斜\s*([<>≤≥])\s*([\d.]+)", user_head_tilt, "头倾斜"),
                (r"左圆肩\s*([<>≤≥])\s*([\d.]+)", user_left_round_shoulder, "左圆肩"),
                (r"右圆肩\s*([<>≤≥])\s*([\d.]+)", user_right_round_shoulder, "右圆肩"),
                (r"左腿X型\s*([<>≤≥])\s*([\d.]+)", user_left_leg_xo, "左腿X型"),
                (r"右腿X型\s*([<>≤≥])\s*([\d.]+)", user_right_leg_xo, "右腿X型")
            ]
            
            for pattern, user_value, metric_name in posture_patterns:
                matches = re.findall(pattern, condition)
                for operator, threshold in matches:
                    if user_value is not None and user_value != 0:
                        threshold = float(threshold)
                        match_result[f"{metric_name}{operator}{threshold}"] = user_value
                        
                        condition_met = HealthAssessment._evaluate_numeric_condition(user_value, operator, threshold)
                        if not condition_met:
                            all_conditions_met = False
                            match_result[f"{metric_name}{operator}{threshold}_result"] = "不满足"
                        else:
                            match_result[f"{metric_name}{operator}{threshold}_result"] = "满足"
                    else:
                        # 对于体态指标，如果用户值为0或None，可能表示正常
                        if threshold == 0:  # 如果阈值是0，则用户值0表示满足条件
                            match_result[f"{metric_name}{operator}{threshold}"] = user_value or 0
                            match_result[f"{metric_name}{operator}{threshold}_result"] = "满足"
                        else:
                            all_conditions_met = False
                            match_result[f"{metric_name}{operator}{threshold}"] = f"用户无{metric_name}数据"
            
            # 检查关键词匹配（针对没有具体数值条件的规则）
            if not match_result:  # 如果没有找到数值条件，尝试关键词匹配
                condition_lower = condition.lower()
                
                # 体成分关键词匹配
                if any(word in condition_lower for word in ["肥胖", "过重"]) and user_bmi and user_bmi >= 25:
                    match_result["肥胖"] = f"BMI={user_bmi}"
                    all_conditions_met = True
                elif any(word in condition_lower for word in ["消瘦", "偏瘦"]) and user_bmi and user_bmi < 18.5:
                    match_result["消瘦"] = f"BMI={user_bmi}"
                    all_conditions_met = True
                
                # 体态关键词匹配
                posture_keyword_checks = [
                    ("高低肩", user_shoulder_imbalance, 2.0),
                    ("头前倾", user_head_forward, 1.0),
                    ("圆肩", max(user_left_round_shoulder or 0, user_right_round_shoulder or 0), 10.0)
                ]
                
                for keyword, value, threshold in posture_keyword_checks:
                    if keyword in condition_lower and value and value > threshold:
                        match_result[keyword] = value
                        all_conditions_met = True
            
            return all_conditions_met and len(match_result) > 0, match_result
            
        except Exception as e:
            print(f"检查规则适用性失败: {e}")
            return False, {"error": str(e)}
    
    @staticmethod
    def _evaluate_numeric_condition(user_value, operator, threshold):
        """评估数值条件是否满足"""
        if operator in ["<", "＜"]:
            return user_value < threshold
        elif operator in [">", "＞"]:
            return user_value > threshold
        elif operator in ["≤", "<=", "≦"]:
            return user_value <= threshold
        elif operator in ["≥", ">=", "≧"]:
            return user_value >= threshold
        return False 