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
import json
import ast
from tools import MedicalAnalysis, HealthAssessment

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
    description='你是身体异常分析团队的leader，你需要根据用户提供的身体数据，决定要以怎样的顺序去使用这些assistant',
    instructions="""你的团队中有以下assistant：
    UserDataAnalysisAssistant：可以分析用户提供的身体数据（如BMI、去脂体重指数等），提取关键指标；
    KnowledgeQueryAssistant：可以查询身体异常判断决策树知识库，获取相关的判断规则；
    AbnormalityAnalysisAssistant：可以基于用户数据和决策树规则，进行身体异常分析和判断；
    ChatAssistant：如果用户的问题不是身体数据分析相关，则调用该assistant。
    
    对于身体数据分析问题，推荐的流程是：["UserDataAnalysisAssistant", "KnowledgeQueryAssistant", "AbnormalityAnalysisAssistant"]
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
    description='一个专业的身体数据分析助手，能够分析用户提供的身体指标数据',
    instructions='你是一个专业的身体数据分析助手，专门负责分析用户提供的身体数据。请仔细分析用户提供的身体指标（如性别、BMI、去脂体重指数等），提取关键数据信息。',
    tools=[
        {
            'type': 'function',
            'function': {
                'name': '身体数据分析',
                'description': '分析用户提供的身体数据，提取关键指标信息',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'user_data': {
                            'type': 'string',
                            'description': '用户提供的身体数据内容'
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
    description='一个专业的助手，能够查询身体异常判断决策树知识库获取相关判断规则',
    instructions='你是一个专业的身体异常分析助手，专门负责查询身体异常判断决策树知识库。根据用户的身体数据，查询相关的判断规则和决策树信息。',
    tools=[
        {
            'type': 'function',
            'function': {
                'name': '身体异常决策树查询',
                'description': '查询身体异常判断决策树知识库获取相关的判断规则',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query_text': {
                            'type': 'string',
                            'description': '要查询的身体指标相关内容，通常是BMI、去脂体重指数等身体数据'
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
    description='一个专业的身体异常分析助手，能够基于用户数据和决策树规则进行异常判断',
    instructions='你是一个专业的身体异常分析助手，专门负责综合分析。请严格按照从决策树知识库获取的判断规则，结合用户的身体数据进行异常分析。你必须：1) 明确展示应用的决策规则条件；2) 说明用户数据如何匹配这些条件；3) 根据匹配结果得出具体的异常结论；4) 在回复中详细展示整个决策流程。',
    tools=[
        {
            'type': 'function',
            'function': {
                'name': '身体异常综合分析',
                'description': '基于用户身体数据和决策树规则，进行身体异常分析和判断',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'user_data': {
                            'type': 'string',
                            'description': '用户身体数据分析结果'
                        },
                        'decision_rules': {
                            'type': 'string',
                            'description': '从决策树知识库获得的相关判断规则'
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
    instructions='你是一个专业的身体异常分析助手，负责最终的总结分析。请根据用户的身体数据以及数据分析、决策树查询、异常判断等各个阶段的信息，提供全面、专业的身体异常分析结论。请注意：你的分析结果仅供参考，建议用户咨询专业的健康顾问或医生。'
)

# 将工具函数的name映射到函数本体
function_mapper = {
    "身体数据分析": MedicalAnalysis.analyze_symptoms,
    "身体异常决策树查询": MedicalAnalysis.query_medical_knowledge,
    "身体异常综合分析": HealthAssessment.comprehensive_analysis
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
        f = run_status.required_action.submit_tool_outputs.tool_calls[0].function
        func_name = f['name']
        param = json.loads(f['arguments'])
        print("func_name",func_name)
    
        if func_name in function_mapper:
            # 如果是身体异常决策树查询，添加知识库参数
            if func_name == "身体异常决策树查询" and 'knowledge_base_name' not in param:
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
            output = function_mapper[func_name](**param)
            tool_output = output  # 保存工具输出
        else:    
            output = ""
        
        tool_outputs = [{
            'output': output
        }]
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
        
        order_stk = ast.literal_eval(assistant_order)
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
                        kb_data = json.loads(tool_output)
                        if "retrieved_chunks" in kb_data:
                            chunks = kb_data["retrieved_chunks"]
                            for chunk in chunks[:5]:  # 只显示前5个
                                collected_knowledge_chunks += f"## {chunk.get('rule_id', 'N/A')}:\n{chunk.get('content', '')}\n置信度: {chunk.get('confidence_score', 'N/A')}\n\n"
                    except Exception as e:
                        print(f"解析决策树工具输出失败: {e}")
                        
                # 如果还没有获取到决策树信息，尝试直接查询
                if not collected_knowledge_chunks and knowledge_base:
                    try:
                        from tools import MedicalAnalysis
                        kb_result = MedicalAnalysis.query_medical_knowledge(query, knowledge_base)
                        kb_data = json.loads(kb_result)
                        if "retrieved_chunks" in kb_data:
                            chunks = kb_data["retrieved_chunks"]
                            for chunk in chunks[:5]:
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