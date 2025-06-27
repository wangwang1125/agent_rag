import os
from openai import OpenAI
from llama_index.core import StorageContext,load_index_from_storage,Settings
from llama_index.embeddings.dashscope import (
    DashScopeEmbedding,
    DashScopeTextEmbeddingModels,
    DashScopeTextEmbeddingType,
)
from llama_index.postprocessor.dashscope_rerank import DashScopeRerank
from create_kb import *

# 导入多智能体相关模块
from dashscope import Assistants, Messages, Runs, Threads
import dashscope
import json
import ast
from tools import MedicalAnalysis, HealthAssessment

# 设置DashScope API密钥
dashscope.api_key = "sk-51d30a5436ca433b8ff81e624a23dcac"

DB_PATH = "VectorStore"
TMP_NAME = "tmp_abcd"
EMBED_MODEL = DashScopeEmbedding(
    model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2,
    text_type=DashScopeTextEmbeddingType.TEXT_TYPE_DOCUMENT,
    api_key="sk-51d30a5436ca433b8ff81e624a23dcac",
)
# 若使用本地嵌入模型，请取消以下注释：
# from langchain_community.embeddings import ModelScopeEmbeddings
# from llama_index.embeddings.langchain import LangchainEmbedding
# embeddings = ModelScopeEmbeddings(model_id="modelscope/iic/nlp_gte_sentence-embedding_chinese-large")
# EMBED_MODEL = LangchainEmbedding(embeddings)

# 设置嵌入模型
Settings.embed_model = EMBED_MODEL

# ==================== 多智能体定义 ====================

    # 决策级别的agent，决定使用哪些agent，以及它们的运行顺序
PlannerAssistant = Assistants.create(
    model="qwen-plus",
    name='身体异常分析流程编排机器人',
    description='你是身体异常分析团队的leader，你需要根据用户提供的结构化身体数据，决定要以怎样的顺序去使用这些assistant',
    instructions="""你的团队中有以下assistant：
    UserDataAnalysisAssistant：可以分析用户提供的结构化身体数据（包括体成分、体态、围度等），提取关键指标；
    KnowledgeQueryAssistant：可以查询身体异常判断决策树知识库，获取体成分和体态相关的判断规则；
    AbnormalityAnalysisAssistant：可以基于用户数据和决策树规则，进行体成分异常分析（1个）和体态异常分析（最多4个）；
    ChatAssistant：如果用户的问题不是身体数据分析相关，则调用该assistant。
    
    对于结构化身体数据分析问题，推荐的流程是：["UserDataAnalysisAssistant", "KnowledgeQueryAssistant", "AbnormalityAnalysisAssistant"]
    对于非身体数据分析问题：["ChatAssistant"]
    
    你的返回形式必须是一个列表，不能返回其它信息。列表中的元素只能为上述的assistant名称。"""
)

# 功能是回复日常问题。对于日常问题来说，可以使用价格较为低廉的模型作为agent的基座
ChatAssistant = Assistants.create(
    model="qwen-turbo",
    name='回答日常问题的机器人',
    description='一个智能助手，解答用户的问题',
    instructions='请礼貌地回答用户的问题'
)

# 用户数据分析助手
UserDataAnalysisAssistant = Assistants.create(
    model="qwen-plus",
    name='用户身体数据分析机器人',
    description='一个专业的身体数据分析助手，能够分析用户提供的结构化身体指标数据',
    instructions='你是一个专业的身体数据分析助手，专门负责分析用户提供的身体数据。请仔细分析用户提供的结构化身体数据，包括体成分数据(mass_info)、围度信息(girth_info)、体态评估(eval_info)等，提取关键数据信息并识别潜在的异常指标。',
    tools=[
        {
            'type': 'function',
            'function': {
                'name': '结构化身体数据分析',
                'description': '分析用户提供的结构化身体数据，提取体成分和体态关键指标信息',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'user_data': {
                            'type': 'string',
                            'description': '用户提供的结构化身体数据内容，包含mass_info、girth_info、eval_info等'
                        },
                    },
                    'required': ['user_data']},
            }
        }
    ]
)

# 知识库查询助手
KnowledgeQueryAssistant = Assistants.create(
    model="qwen-plus",
    name='身体异常决策树查询机器人',
    description='一个专业的助手，能够查询身体异常判断决策树知识库获取体成分和体态异常相关判断规则',
    instructions='你是一个专业的身体异常分析助手，专门负责查询身体异常判断决策树知识库。根据用户的身体数据，分别查询体成分异常和体态异常的相关判断规则和决策树信息。需要重点关注体成分指标（BMI、体脂率、去脂体重等）和体态指标（高低肩、头前倾、圆肩等）。',
    tools=[
        {
            'type': 'function',
            'function': {
                'name': '体成分异常决策树查询',
                'description': '查询体成分异常判断决策树知识库获取相关的判断规则',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query_text': {
                            'type': 'string',
                            'description': '要查询的体成分指标相关内容，如BMI、体脂率、去脂体重指数等'
                        },
                        'knowledge_base_name': {
                            'type': 'string',
                            'description': '指定的知识库名称，可选参数'
                        }
                    },
                    'required': ['query_text']},
            }
        },
        {
            'type': 'function',
            'function': {
                'name': '体态异常决策树查询',
                'description': '查询体态异常判断决策树知识库获取相关的判断规则',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query_text': {
                            'type': 'string',
                            'description': '要查询的体态指标相关内容，如高低肩、头前倾、圆肩、腿型等'
                        },
                        'knowledge_base_name': {
                            'type': 'string',
                            'description': '指定的知识库名称，可选参数'
                        }
                    },
                    'required': ['query_text']},
            }
        }
    ]
)

# 身体异常分析助手
AbnormalityAnalysisAssistant = Assistants.create(
    model="qwen-plus",
    name='身体异常分析机器人',
    description='一个专业的身体异常分析助手，能够基于用户数据和决策树规则进行体成分和体态异常判断',
    instructions="""你是一个专业的身体异常分析专家，专门负责根据知识库中的决策树规则和用户身体数据进行精确的异常判断。

你必须严格按照以下要求执行：

【核心要求】
1. 必须生成1个体成分异常分析和最多4个体态异常分析
2. 严格按照知识库决策树规则进行判断，不得擅自推测
3. 每个异常判断都必须展示完整的决策流程
4. 按照知识库中的优先级排序异常

【分析流程】
1. 解析用户身体数据，提取关键指标
2. 解析知识库决策树规则，识别判断条件和异常结论
3. 将用户数据与决策树条件进行匹配
4. 展示详细的判断过程：条件→用户数据→匹配结果→结论
5. 按优先级排序并输出最终结论

【输出格式要求】
对于每个异常分析，必须包含：
- 应用的决策规则条件
- 用户数据如何匹配这些条件  
- 具体的匹配过程和计算
- 基于匹配结果得出的结论
- 完整的决策流程展示

【严格限制】
- 只能基于知识库中存在的决策规则进行判断
- 不能根据经验或常识做出判断
- 必须展示从条件判断到结论的完整过程
- 如果知识库中没有相关规则，必须明确说明无法判断

你的分析必须专业、准确、可追溯，确保每个结论都有明确的决策依据。""",
    tools=[
        {
            'type': 'function',
            'function': {
                'name': '体成分异常分析',
                'description': '基于用户体成分数据和决策树规则，进行严格的体成分异常分析和判断，必须生成1个体成分异常分析',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'user_data': {
                            'type': 'string',
                            'description': '用户体成分数据分析结果，包含BMI、体脂率、去脂体重等关键指标'
                        },
                        'decision_rules': {
                            'type': 'string',
                            'description': '从决策树知识库获得的体成分相关判断规则，包含异常结论和判断流程'
                        }
                    },
                    'required': ['user_data', 'decision_rules']},
            }
        },
        {
            'type': 'function',
            'function': {
                'name': '体态异常分析',
                'description': '基于用户体态数据和决策树规则，进行严格的体态异常分析和判断，最多生成4个体态异常分析',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'user_data': {
                            'type': 'string',
                            'description': '用户体态数据分析结果，包含高低肩、头前倾、圆肩等体态指标'
                        },
                        'decision_rules': {
                            'type': 'string',
                            'description': '从决策树知识库获得的体态相关判断规则，包含异常结论和判断流程'
                        }
                    },
                    'required': ['user_data', 'decision_rules']},
            }
        }
    ]
)

# 在Multi Agent场景下，定义一个用于总结的Agent，该Agent会根据用户的问题与之前Agent输出的参考信息，全面、完整地回答用户问题
SummaryAssistant = Assistants.create(
    model="qwen-plus",
    name='身体异常总结机器人',
    description='一个专业的身体异常分析助手，根据用户的身体数据与各个分析阶段的参考信息，提供全面、完整的异常分析结论',
    instructions="""你是一个专业的身体异常分析总结专家，负责基于多智能体分析结果提供最终的综合报告。

你必须严格按照以下要求进行总结：

【核心任务】
1. 总结分析出1个体成分异常和最多4个体态异常
2. 严格按照知识库中的优先级排序异常结论
3. 展示完整的决策判断流程
4. 提供基于决策树规则的专业结论

【总结格式要求】
## 身体异常分析报告

### 一、体成分异常分析
- **异常结论**: [基于决策树规则得出的结论]
- **判断依据**: [具体的决策规则条件]
- **数据匹配**: [用户数据如何满足条件]
- **决策流程**: [从条件到结论的完整过程]
- **优先级**: [知识库中的优先级]

### 二、体态异常分析（按优先级排序）
1. **[优先级1] 异常名称**
   - 判断依据: [决策规则]
   - 数据匹配: [用户数据匹配情况]
   - 决策流程: [判断过程]

2. **[优先级X] 异常名称**
   - 判断依据: [决策规则]
   - 数据匹配: [用户数据匹配情况]
   - 决策流程: [判断过程]

### 三、综合结论
- 基于知识库决策树规则，用户存在X项异常
- 按优先级排序的风险等级
- 建议采取的措施

【严格要求】
- 只能基于知识库决策树规则得出结论
- 必须展示每个异常的完整判断流程
- 严格按照优先级排序（数字越小优先级越高）
- 如果某类异常未发现，需明确说明
- 每个结论都要有明确的决策依据

【免责声明】
此分析结果严格基于知识库决策树规则，仅供参考，建议咨询专业的健康顾问或医生获取准确诊断。"""
)

# 将工具函数的name映射到函数本体
function_mapper = {
    "结构化身体数据分析": MedicalAnalysis.analyze_symptoms,
    "体成分异常决策树查询": MedicalAnalysis.query_medical_knowledge,
    "体态异常决策树查询": MedicalAnalysis.query_medical_knowledge,
    "体成分异常分析": HealthAssessment.body_composition_analysis,
    "体态异常分析": HealthAssessment.posture_analysis
}

# 将Agent的name映射到Agent本体
assistant_mapper = {
    "ChatAssistant": ChatAssistant,
    "UserDataAnalysisAssistant": UserDataAnalysisAssistant,
    "KnowledgeQueryAssistant": KnowledgeQueryAssistant,
    "AbnormalityAnalysisAssistant": AbnormalityAnalysisAssistant
}

# ==================== Agent处理函数 ====================

def get_agent_response(assistant, message='', return_tool_output=False):
    """输入message信息，输出为指定Agent的回复"""
    #print(f"Query: {message}")
    thread = Threads.create()
    message = Messages.create(thread.id, content=message)
    run = Runs.create(thread.id, assistant_id=assistant.id)
    run_status = Runs.wait(run.id, thread_id=thread.id)
    
    tool_output = None  # 存储工具输出
    
    if run_status.status == 'failed':
        print('run failed:')
        return ("抱歉，处理过程中出现错误", tool_output) if return_tool_output else "抱歉，处理过程中出现错误"
    
    if run_status.required_action:
        tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        all_tool_output = ""  # 存储所有工具输出
        
        # 处理多个工具调用
        for tool_call in tool_calls:
            f = tool_call.function
            func_name = f['name']  
            param = json.loads(f['arguments'])
            print("func_name", func_name)
        
            if func_name in function_mapper:
                # 如果是决策树查询，添加知识库参数
                if func_name in ["体成分异常决策树查询", "体态异常决策树查询"] and 'knowledge_base_name' not in param:
                    # 从全局变量或传递的参数中获取知识库名称
                    import inspect
                    frame = inspect.currentframe()
                    try:
                        # 尝试从调用堆栈中获取knowledge_base参数
                        caller_locals = frame.f_back.f_back.f_locals
                        if 'knowledge_base' in caller_locals:
                            param['knowledge_base_name'] = caller_locals['knowledge_base']
                    finally:
                        del frame
                
                try:
                    output = function_mapper[func_name](**param)
                    # 确保输出不为空
                    if output is None:
                        output = '{"error": "工具函数返回空值"}'
                    elif not isinstance(output, str):
                        output = str(output)
                    
                    all_tool_output += f"{func_name}: {output}\n"  # 累积工具输出
                except Exception as e:
                    print(f"工具函数执行失败 {func_name}: {e}")
                    output = f'{{"error": "工具函数执行失败: {str(e)}"}}'
                    all_tool_output += f"{func_name}: {output}\n"
            else:    
                output = '{"error": "未知的工具函数"}'
            
            tool_outputs.append({
                'tool_call_id': tool_call.id,
                'output': output
            })
        
        tool_output = all_tool_output  # 保存所有工具输出
        run = Runs.submit_tool_outputs(run.id,
                                       thread_id=thread.id,
                                       tool_outputs=tool_outputs)
        run_status = Runs.wait(run.id, thread_id=thread.id)
    
    run_status = Runs.get(run.id, thread_id=thread.id)
    msgs = Messages.list(thread.id)
    response = msgs['data'][0]['content'][0]['text']['value']
    if return_tool_output:
        return response, tool_output
    else:
        return response

def get_multi_agent_response_internal(query, knowledge_base=None):
    """获得Multi Agent的回复的内部函数"""
    if len(query) == 0:
        return "请输入您的身体数据或问题", ""
    
    collected_knowledge_chunks = ""  # 收集知识库召回信息
    
    try:
        # 获取Agent的运行顺序
        assistant_order = get_agent_response(PlannerAssistant, query)
        print("assistant_order", assistant_order)
        
        # 安全地解析Assistant顺序
        try:
            if isinstance(assistant_order, str):
                # 尝试不同的解析方法
                assistant_order = assistant_order.strip()
                if assistant_order.startswith('[') and assistant_order.endswith(']'):
                    # 如果是列表格式
                    order_stk = ast.literal_eval(assistant_order)
                elif assistant_order.startswith('{') and assistant_order.endswith('}'):
                    # 如果是JSON格式
                    order_data = json.loads(assistant_order)
                    order_stk = order_data if isinstance(order_data, list) else [assistant_order]
                else:
                    # 尝试从文本中提取列表
                    import re
                    list_match = re.search(r'\[(.*?)\]', assistant_order)
                    if list_match:
                        order_stk = ast.literal_eval('[' + list_match.group(1) + ']')
                    else:
                        # 默认处理
                        order_stk = ["UserDataAnalysisAssistant", "KnowledgeQueryAssistant", "AbnormalityAnalysisAssistant"]
            else:
                order_stk = assistant_order if isinstance(assistant_order, list) else [str(assistant_order)]
        except Exception as e:
            print(f"解析assistant_order失败: {e}, 使用默认顺序")
            order_stk = ["UserDataAnalysisAssistant", "KnowledgeQueryAssistant", "AbnormalityAnalysisAssistant"]
        cur_query = query
        Agent_Message = ""
        
        # 依次运行Agent
        for i in range(len(order_stk)):
            cur_assistant = assistant_mapper[order_stk[i]]
            
            # 如果是决策树查询助手，获取工具输出
            if order_stk[i] == "KnowledgeQueryAssistant":
                response, tool_output = get_agent_response(cur_assistant, cur_query, return_tool_output=True)
                # 解析工具输出中的决策树信息
                if tool_output:
                    try:
                        print(f"工具输出内容: {tool_output}")  # 调试信息
                        # 处理多个工具输出的情况
                        if isinstance(tool_output, str) and tool_output.strip():
                            # 分割多个工具输出
                            tool_outputs = tool_output.strip().split('\n')
                            for output_line in tool_outputs:
                                output_line = output_line.strip()
                                if not output_line:  # 跳过空行
                                    continue
                                    
                                if ':' in output_line and output_line.startswith((
                                    '体成分异常决策树查询:', '体态异常决策树查询:')):
                                    # 解析带前缀的JSON输出
                                    try:
                                        _, json_part = output_line.split(':', 1)
                                        json_part = json_part.strip()
                                        if json_part and json_part.startswith('{'):
                                            kb_data = json.loads(json_part)
                                            if "retrieved_chunks" in kb_data:
                                                chunks = kb_data["retrieved_chunks"]
                                                query_type = kb_data.get("query_type", "决策树查询")
                                                collected_knowledge_chunks += f"### {query_type}结果：\n"
                                                for chunk in chunks[:5]:  # 只显示前5个
                                                    collected_knowledge_chunks += f"## {chunk.get('rule_id', 'N/A')}:\n{chunk.get('content', '')}\n置信度: {chunk.get('confidence_score', 'N/A')}\n\n"
                                    except json.JSONDecodeError as je:
                                        print(f"JSON解析错误 - 前缀格式: {je}")
                                        continue
                                        
                                elif output_line.startswith('{') and output_line.endswith('}'):
                                    # 直接的JSON输出
                                    try:
                                        kb_data = json.loads(output_line)
                                        if "retrieved_chunks" in kb_data:
                                            chunks = kb_data["retrieved_chunks"]
                                            query_type = kb_data.get("query_type", "决策树查询")
                                            collected_knowledge_chunks += f"### {query_type}结果：\n"
                                            for chunk in chunks[:5]:
                                                collected_knowledge_chunks += f"## {chunk.get('rule_id', 'N/A')}:\n{chunk.get('content', '')}\n置信度: {chunk.get('confidence_score', 'N/A')}\n\n"
                                    except json.JSONDecodeError as je:
                                        print(f"JSON解析错误 - 直接格式: {je}")
                                        continue
                                        
                            # 如果没有成功解析任何JSON，但有工具输出，显示原始输出
                            if not collected_knowledge_chunks and tool_output.strip():
                                collected_knowledge_chunks += f"原始工具输出: {tool_output[:500]}...\n"
                                
                    except Exception as e:
                        print(f"解析决策树工具输出失败: {e}")
                        # 如果解析失败，直接显示原始输出
                        if tool_output and tool_output.strip():
                            collected_knowledge_chunks += f"工具输出解析失败，原始内容: {tool_output[:500]}...\n"
                        
                # 如果还没有获取到决策树信息，尝试直接查询体成分和体态异常
                if not collected_knowledge_chunks and knowledge_base:
                    try:
                        from tools import MedicalAnalysis
                        # 查询体成分异常
                        composition_result = MedicalAnalysis.query_medical_knowledge("体成分异常 BMI 体脂率", knowledge_base)
                        composition_data = json.loads(composition_result)
                        if "retrieved_chunks" in composition_data:
                            chunks = composition_data["retrieved_chunks"]
                            collected_knowledge_chunks += "### 体成分异常相关规则：\n"
                            for chunk in chunks[:3]:  # 获取前3个体成分相关
                                collected_knowledge_chunks += f"## {chunk.get('rule_id', 'N/A')}:\n{chunk.get('content', '')}\n置信度: {chunk.get('confidence_score', 'N/A')}\n\n"
                        
                        # 查询体态异常
                        posture_result = MedicalAnalysis.query_medical_knowledge("体态异常 高低肩 头前倾 圆肩", knowledge_base)
                        posture_data = json.loads(posture_result)
                        if "retrieved_chunks" in posture_data:
                            chunks = posture_data["retrieved_chunks"]
                            collected_knowledge_chunks += "### 体态异常相关规则：\n"
                            for chunk in chunks[:4]:  # 获取前4个体态相关
                                collected_knowledge_chunks += f"## {chunk.get('rule_id', 'N/A')}:\n{chunk.get('content', '')}\n置信度: {chunk.get('confidence_score', 'N/A')}\n\n"
                    except Exception as e:
                        print(f"直接查询决策树知识库失败: {e}")
            else:
                response = get_agent_response(cur_assistant, cur_query)
            
            Agent_Message += f"*{order_stk[i]}*的回复为：{response}\n\n"
            
            # 如果当前Agent为最后一个Agent，则将其输出作为Multi Agent的输出
            if i == len(order_stk)-1:
                prompt = f"请参考已知的信息：{Agent_Message}，回答用户的问题：{query}。"
                multi_agent_response = get_agent_response(SummaryAssistant, prompt)
                
                # 确保有召回文本段显示
                if not collected_knowledge_chunks:
                    if "KnowledgeQueryAssistant" in order_stk:
                        collected_knowledge_chunks = "多智能体模式：已完成身体异常决策树查询，但未检索到足够相关的内容。建议提供更详细的身体数据或咨询专业健康顾问。"
                    else:
                        collected_knowledge_chunks = "多智能体模式：此问题未涉及决策树查询，已通过通用问答处理。"
                
                return multi_agent_response, collected_knowledge_chunks
            # 如果当前Agent不是最后一个Agent，则将上一个Agent的输出response添加到下一轮的query中，作为参考信息
            else:
                cur_query = f"你可以参考已知的信息：{response}你要完整地回答用户的问题。问题是：{query}。"
    
    except Exception as e:
        print(f"Multi-agent processing failed: {e}")
        # 兜底策略，如果上述程序运行失败，则直接调用ChatAssistant
        fallback_response = get_agent_response(ChatAssistant, query)
        return fallback_response, "多智能体模式出错，已切换到通用问答模式"

# ==================== 原有RAG函数 ====================

def get_model_response(multi_modal_input,history,model,temperature,max_tokens,history_round,db_name,similarity_threshold,chunk_cnt):
    # prompt = multi_modal_input['text']
    prompt = history[-1][0]
    tmp_files = multi_modal_input['files']
    if os.path.exists(os.path.join("File",TMP_NAME)):
        db_name = TMP_NAME
    else:
        if tmp_files:
            create_tmp_kb(tmp_files)
            db_name = TMP_NAME
    # 获取index
    print(f"prompt:{prompt},tmp_files:{tmp_files},db_name:{db_name}")
    try:
        dashscope_rerank = DashScopeRerank(
            top_n=chunk_cnt,
            return_documents=True,
            api_key="sk-51d30a5436ca433b8ff81e624a23dcac"
        )
        storage_context = StorageContext.from_defaults(
            persist_dir=os.path.join(DB_PATH,db_name)
        )
        index = load_index_from_storage(storage_context)
        print("index获取完成")
        retriever_engine = index.as_retriever(
            similarity_top_k=20,
        )
        # 获取chunk
        retrieve_chunk = retriever_engine.retrieve(prompt)
        print(f"原始chunk为：{retrieve_chunk}")
        try:
            results = dashscope_rerank.postprocess_nodes(retrieve_chunk, query_str=prompt)
            print(f"rerank成功，重排后的chunk为：{results}")
        except:
            results = retrieve_chunk[:chunk_cnt]
            print(f"rerank失败，chunk为：{results}")
        chunk_text = ""
        chunk_show = ""
        for i in range(len(results)):
            if results[i].score >= similarity_threshold:
                chunk_text = chunk_text + f"## {i+1}:\n {results[i].text}\n"
                chunk_show = chunk_show + f"## {i+1}:\n {results[i].text}\nscore: {round(results[i].score,2)}\n"
        print(f"已获取chunk：{chunk_text}")
        prompt_template = f"请参考以下内容：{chunk_text}，以合适的语气回答用户的问题：{prompt}。如果参考内容中有图片链接也请直接返回。"
    except Exception as e:
        print(f"异常信息：{e}")
        prompt_template = prompt
        chunk_show = ""
    history[-1][-1] = ""
    client = OpenAI(
        api_key="sk-51d30a5436ca433b8ff81e624a23dcac",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )                
    system_message = {'role': 'system', 'content': 'You are a helpful assistant.'}
    messages = []
    history_round = min(len(history),history_round)
    for i in range(history_round):
        messages.append({'role': 'user', 'content': history[-history_round+i][0]})
        messages.append({'role': 'assistant', 'content': history[-history_round+i][1]})
    messages.append({'role': 'user', 'content': prompt_template})
    messages = [system_message] + messages
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True
        )
    assistant_response = ""
    for chunk in completion:
        assistant_response += chunk.choices[0].delta.content
        history[-1][-1] = assistant_response
        yield history,chunk_show

# ==================== 统一响应函数 ====================

def get_unified_response(multi_modal_input, history, mode, model, temperature, max_tokens, history_round, knowledge_base, similarity_threshold, chunk_cnt):
    """统一的响应函数，支持RAG和多智能体两种模式"""
    prompt = history[-1][0] if history else ""
    
    if mode == "multi_agent":
        # 多智能体模式
        try:
            response, knowledge_chunks = get_multi_agent_response_internal(prompt, knowledge_base)
            history[-1][-1] = response
            yield history, knowledge_chunks
        except Exception as e:
            print(f"多智能体模式失败，降级到RAG模式: {e}")
            # 降级到RAG模式
            yield from get_model_response(multi_modal_input, history, model, temperature, max_tokens, history_round, knowledge_base, similarity_threshold, chunk_cnt)
    else:
        # RAG模式
        yield from get_model_response(multi_modal_input, history, model, temperature, max_tokens, history_round, knowledge_base, similarity_threshold, chunk_cnt)
        
def test_body_analysis():
    """测试身体异常分析的多智能体流程"""
    # 示例用户数据
    user_body_data = {
        "mass_info":{
            "WT":{"l":25.8, "m":30.4, "h":35, "v":48.7, "status":3},
            "FFM":{"l":28.7, "m":31.9, "h":35.1, "v":4.9, "status":1},
            "TBW":{"l":37.285292814, "m":41.412983381000004, "h":45.540673948000006, "v":34.110146224000005, "status":1},
            "BMI":{"l":18.5, "m":22, "h":24, "v":23.4, "status":2},
            "PBF":{"l":10, "m":15, "h":20, "v":33.7, "status":3},
            "BMR":{"l":1432.4, "m":1591.6, "h":1750.8, "v":1413.1, "status":1},
            "WHR":{"l":0.8, "m":0.85, "h":0.9, "v":0.88, "status":2},
            "SM":{"l":28.440241599000004, "m":31.842184374000002, "h":35.289486386, "v":25.764046616, "status":1},
            "PROTEIN":{"l":9.979032140000001, "m":11.113013065, "h":12.24699399, "v":9.162565874, "status":1},
            "ICW":{"l":23.133210870000003, "m":25.718687379000002, "h":28.304163888, "v":21.092045205, "status":1},
            "ECW":{"l":14.152081944, "m":15.739655239000003, "h":17.327228534000003, "v":13.471693389, "status":1},
            "VAG":{"l":0.9, "h":10, "m":0, "v":9, "status":2}
        },
        "girth_info":{
            "left_calf":40.9, "right_calf":41.4, "neck":37.083999999999996,
            "waist":87.884, "hip":103.37800000000001, "left_upper_arm":28.194, "right_upper_arm":28.448
        },
        "eval_info":{
            "id":20985, "scan_id":"37123456789136-950b07ba-875a-41b7-8850-271bf5b79764",
            "high_low_shoulder":3.302, "head_slant":0, "head_forward":1.5,
            "left_leg_xo":179.5, "right_leg_xo":174.3, "pelvis_forward":176.5,
            "left_knee_check":183.6, "right_knee_check":175.9,
            "round_shoulder_left":7.7, "round_shoulder_right":19.6, "create_time":None
        },
        "eval_conclusion":{
            "head_slant":{"conclusion_key":"body-product.bsEval.headSlant.normal", "val":0},
            "high_low_shoulder":{"conclusion_key":"body-product.bsEval.highLowShoudler.left", "val":3.302},
            "head_forward":{"conclusion_key":"body-product.bsEval.headForward.head", "val":1.5},
            "round_shoulder_left":{"conclusion_key":"body-product.bsEval.roundShoulder.left", "val":7.7},
            "round_shoulder_right":{"conclusion_key":"body-product.bsEval.roundShoulder.right", "val":19.6},
            "is_new_math_tt":1
        },
        "shoulder_info":{
            "id":2192, "scan_id":"37123456789136-950b07ba-875a-41b7-8850-271bf5b79764",
            "result":1, "left_abduction":173.2, "right_abduction":181.1,
            "left_adduction":174.1, "right_adduction":171, "create_time":1724313842
        },
        "user_info":{"height":174, "age":30, "sex":1},
        "isForeign":"0", "unit":"imperial", "lang": "zh-CN"
    }
    
    # 转换为字符串格式供分析使用
    query = f"请分析以下身体数据并生成1个体成分异常和最多4个体态异常分析：{json.dumps(user_body_data, ensure_ascii=False)}"
    
    # 调用多智能体分析
    try:
        response, knowledge_chunks = get_multi_agent_response_internal(query, "medical_kb")
        print("=== 多智能体分析结果 ===")
        print(f"分析结果：{response}")
        print("\n=== 知识库召回信息 ===")
        print(knowledge_chunks)
        return response, knowledge_chunks
    except Exception as e:
        print(f"测试失败：{e}")
        return None, None

if __name__ == "__main__":
    # 运行测试
    test_body_analysis()