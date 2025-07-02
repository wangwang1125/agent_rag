import os
import logging
import warnings
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
from tools import MedicalAnalysis

# 禁用DashScope和相关库的日志输出
logging.getLogger('dashscope').setLevel(logging.ERROR)
logging.getLogger('openai').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)
logging.getLogger('requests').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# 禁用所有警告
warnings.filterwarnings('ignore')

# 设置根日志级别为WARNING，避免调试信息输出
logging.basicConfig(level=logging.WARNING)

# 全局过滤DashScope调试输出的解决方案
import sys
import contextlib

class DashScopeOutputFilter:
    """全局过滤DashScope JSON调试输出的类"""
    def __init__(self, original_stdout, original_stderr):
        self.original_stdout = original_stdout
        self.original_stderr = original_stderr
        self.suppress_next_newlines = False  # 标记是否需要抑制后续的换行
        
    def write(self, text):
        if text and isinstance(text, str):
            # 过滤DashScope的JSON调试输出
            text_lower = text.lower()
            if (text.strip().startswith('{') and 
                any(keyword in text for keyword in ['assistant_id', 'thread_id', 'run_id', 'object": "thread.run'])):
                self.suppress_next_newlines = True  # 设置标记，抑制后续换行
                return  # 不输出
            if any(keyword in text_lower for keyword in [
                'status_code', 'request_id', 'created_at', 'instructions'
            ]) and text.strip().startswith('{'):
                self.suppress_next_newlines = True  # 设置标记，抑制后续换行
                return  # 不输出
            
            # 如果当前在抑制模式，且文本只包含空白字符（换行符、空格等），则不输出
            if self.suppress_next_newlines:
                if text.strip() == '':  # 只包含空白字符
                    return  # 不输出空白行
                else:
                    # 遇到非空白内容，取消抑制模式
                    self.suppress_next_newlines = False
                    
        # 输出到原始stdout
        self.original_stdout.write(text)
        
    def flush(self):
        self.original_stdout.flush()

# 应用全局过滤器
original_stdout = sys.stdout
original_stderr = sys.stderr
sys.stdout = DashScopeOutputFilter(original_stdout, original_stderr)

# 设置DashScope API密钥
dashscope.api_key = "sk-51d30a5436ca433b8ff81e624a23dcac"

# 进一步控制DashScope输出 - 设置环境变量
os.environ['DASHSCOPE_DEBUG'] = 'false'
os.environ['DASHSCOPE_VERBOSE'] = 'false'
os.environ['OPENAI_LOG_LEVEL'] = 'error'

# 如果DashScope有配置选项，设置为静默模式
try:
    dashscope.api_base = dashscope.api_base  # 保持默认值
    # 尝试设置调试模式为False（如果支持）
    if hasattr(dashscope, 'debug'):
        dashscope.debug = False
    if hasattr(dashscope, 'verbose'):
        dashscope.verbose = False
except:
    pass

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

# ==================== 决策树定义 ====================

decision_tree = """
{
    "体成分": {
        "体重偏低": {
            "优先级": 3,
            "异常判断流程": [
                "BMI < 18.5 且（女性去脂体重指数 ≥ 15 或男性去脂体重指数 ≥ 17）"
            ]
        },
        "消瘦": {
            "优先级": 1,
            "异常判断流程": [
                "BMI < 18.5 且（女性去脂体重指数 < 15 或 男性去脂体重指数 < 17）"
            ]
        },
        "肌肉过少": {
            "优先级": 2,
            "异常判断流程": [
                "BMI > 18.5 且（女性去脂体重指数 < 15 或 男性去脂体重指数 < 17）",
                "BMI ≥ 28（海外标准：≥ 30）且 体脂率（女性 ≥ 30% 或 男性 ≥ 25%）且 双侧小腿围（女性 < 33cm 或 男性 < 34cm）",
                "BMI ≥ 28（海外标准：≥ 30）且 体脂率（女性 ≥ 30% 或 男性 ≥ 25%）且 去脂体重指数（女性 < 15 或 男性 < 17）"
            ]
        },
        "中心性肥胖": {
            "优先级": 2,
            "异常判断流程": [
                "18.5 < BMI < 27.9（海外标准：< 30）且 腰臀比：中国以外地区：男性 > 1.0 或 女性 > 0.9 中国地区：男性 > 0.9 或 女性 > 0.85",
                "18.5 < BMI < 27.9（海外标准：< 30）且 腰围（女性 ≥ 80cm 或 男性 ≥ 85cm，海外男性 ≥ 90cm）且 内脏脂肪等级 ≥ 10"
            ]
        },
        "肥胖": {
            "优先级": 2,
            "异常判断流程": [
                "BMI ≥ 28（海外标准：≥ 30）且 体脂率（女性 ≥ 30% 或 男性 ≥ 25%）"
            ]
        },
        "少肌性肥胖": {
            "优先级": 1,
            "异常判断流程": [
                "BMI ≥ 28（海外标准：≥ 30）且 体脂率（女性 ≥ 30% 或 男性 ≥ 25%）且 双侧小腿围（女性 < 33cm 或 男性 < 34cm）",
                "BMI ≥ 28（海外标准：≥ 30）且 体脂率（女性 ≥ 30% 或 男性 ≥ 25%）且 去脂体重指数（女性 < 15 或 男性 < 17）"
            ]
        },
        "体重正常": {
            "优先级": 5,
            "异常判断流程": [
                "18.5 ≤ BMI < 24且 体脂率（女性 < 30% 且 男性 < 25%）且（腰围：女性 < 80cm 且 男性 < 85cm，海外男性 < 90cm）或 内脏脂肪等级 < 10"
            ]
        },
        "超重": {
            "优先级": 4,
            "异常判断流程": [
                "24 ≤ BMI < 28（海外标准：< 30）且（体脂率：女性 25%-30% 或 男性 20%-25%）或（腰围：女性 77-80cm 或 男性 80-85cm 且 8 ≤ 内脏脂肪等级 < 10）"
            ]
        },
        "肌肉发达": {
            "优先级": 5,
            "异常判断流程": [
                "24 ≤ BMI < 28（海外标准：< 30）且 去脂体重指数（男性 ≥ 20 或 女性 ≥ 18）且 体脂率（女性 < 25% 且 男性 < 20%）且（腰围女性 < 77cm 且 男性 < 80cm或 内脏脂肪等级 < 8或 腰臀比：男性 < 0.9 ，女性 < 0.8（海外：男性 < 0.9 且 女性 < 0.85））"
            ]
        },
        "健康风险增加": {
            "优先级": 3,
            "异常判断流程": [
                "18.5 ≤ BMI < 24且 体脂率（女性 < 30% 且 男性 < 25%）且（腰围：女性 < 80cm 且 男性 < 85cm，海外男性 < 90cm）或 内脏脂肪等级 < 10 且 存在以下任一风险指标：颈围：男性 > 43cm 或 女性 > 41cm或 细胞外液/总水分 ≥ 0.4或 细胞外液/细胞内液 ≤ 0.57",
                "24 ≤ BMI < 28（海外标准：< 30）且（体脂率：女性 25%-30% 或 男性 20%-25%）或（腰围：女性 77-80cm 或 男性 80-85cm且 8 ≤ 内脏脂肪等级 < 10）且 存在以下任一风险指标：颈围：男性 > 43cm 或 女性 > 41cm或 细胞外液/总水分 ≥ 0.4或 细胞外液/细胞内液 ≤ 0.57"
            ]
        }
    },
    "体态": {
        "可能骨盆旋移": {
            "优先级": 3,
            "异常判断流程": [
                "骨盆前移角度≤179或骨盆前移距离≥2cm且双侧膝关节角度差异相差≥5°。同时较大腿型角度一侧的膝关节角度比较小膝关节一侧腿型角度多3"
            ]
        },
        "可能骨盆前移": {
            "优先级": 3,
            "异常判断流程": [
                "骨盆前移角度≤179或骨盆前移距离>2cm且双侧膝关节角度≥181"
            ]
        },
        "可能骨盆前倾": {
            "优先级": 3,
            "异常判断流程": [
                "骨盆前移角度>179或骨盆前移距离≤2cm且双侧膝关节角度>184°且双侧腿型角度<179"
            ]
        },
        "可能骨盆后倾": {
            "优先级": 3,
            "异常判断流程": [
                "骨盆前移角度>179或骨盆前移距离≤2cm且双侧膝关节角度<179°且双侧腿型角度>181"
            ]
        },
        "骨盆旋移且可能诱发肩关节活动受限": {
            "优先级": 2,
            "异常判断流程": [
                "骨盆前移角度≤179或骨盆前移距离≥2cm且双侧膝关节角度差异相差≥5°。同时较大腿型角度一侧的膝关节角度比较小膝关节一侧腿型角度多3且膝关节角度较大一侧肩部前伸上举上举小于<175°"
            ]
        },
        "骨盆旋移且可能诱发头侧歪": {
            "优先级": 2,
            "异常判断流程": [
                "骨盆前移角度≤179或骨盆前移距离≥2cm且双侧膝关节角度差异相差≥5°。同时较大腿型角度一侧的膝关节角度比较小膝关节一侧腿型角度多3且头侧歪朝向为较小膝关节角度一侧"
            ]
        },
        "高低肩是由头侧歪诱发的": {
            "优先级": 5,
            "异常判断流程": [
                ""
            ]
        },
        "（头歪侧/高肩一侧）斜方肌和肩胛提肌紧张": {
            "优先级": 6,
            "异常判断流程": [
                "存在头歪向一侧（头歪侧）且存在高低肩（高肩侧），并且高肩侧与头歪侧相反"
            ]
        },
        "高肩侧 上斜方肌紧张": {
            "优先级": 6,
            "异常判断流程": [
                "存在头前引且存在高低肩（高肩侧）"
            ]
        },
        "头侧歪侧肩胛提肌、头侧歪侧斜角肌、高肩侧上斜方肌紧张、头侧歪的对侧 胸锁乳突肌紧张": {
            "优先级": 6,
            "异常判断流程": [
                "存在头前引、存在头歪向一侧（头歪侧）且存在高低肩（高肩侧），并且高肩侧与头歪侧相反（对侧关系）"
            ]
        },
        "头侧歪侧斜角肌紧张、对侧胸锁乳突肌紧张": {
            "优先级": 6,
            "异常判断流程": [
                "头歪向一侧（头歪侧）、高低肩（高肩侧）、圆肩（圆肩侧）及头前引这四项中至少两项同时存在，且这些症状出现在身体同侧"
            ]
        },
        "且前锯肌紧张、三角肌前束、冈上肌、下斜方肌无力": {
            "优先级": 5,
            "异常判断流程": [
                "存在头侧歪或高低肩且（头歪侧/高肩一侧)斜方肌和肩胛提肌紧张且头歪侧或高肩侧肩部外展上举小于175°"
            ]
        },
        "且三角肌中束和后束无力": {
            "优先级": 5,
            "异常判断流程": [
                "存在头侧歪或高低肩且（头歪侧/高肩一侧)斜方肌和肩胛提肌紧张且头歪侧或高肩一侧前伸上举小于175"
            ]
        },
        "且 前锯肌紧张、三角肌前/中/后束、冈上肌、下斜方肌无力": {
            "优先级": 5,
            "异常判断流程": [
                "前锯肌紧张、三角肌前束、网上肌、下斜方肌无力且三角肌中/后束无力"
            ]
        },
        "可能存在单侧足弓塌陷": {
            "优先级": 2,
            "异常判断流程": [
                "存在高低肩，并且存在头侧歪、头前引或圆肩中至少一项，并且头侧歪朝向高肩侧，并且低肩一侧腿型角度≥180，百高肩一侧腿型角度≤179"
            ]
        },
        "头侧歪和（或）头前引和（或）高低肩是一个问题诱发": {
            "优先级": 5,
            "异常判断流程": [
                "低肩一侧腿型角度≥180，百高肩一侧腿型角度≤179"
            ]
        },
        "头侧歪/高低肩/头前引 均由骨盆旋移诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在高低肩，且存在头歪、头前引或圆肩中至少一项，且（存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或存在头前引且存在高低肩，或存在头前引、存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或头歪向一侧、高低肩、圆肩及头前引这四项中至少两项同时存在且这些症状出现在身体同侧），且存在骨盆前移的可能性"
            ]
        },
        "单圆肩侧胸小肌、胸锁乳突肌、斜角肌、前锯肌紧张": {
            "优先级": 5,
            "异常判断流程": [
                "存在高低肩，且存在头歪、头前引或圆肩中至少一项，且（存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或存在头前引且存在高低肩，或存在头前引、存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或头歪向一侧、高低肩、圆肩及头前引这四项中至少两项同时存在且这些症状出现在身体同侧），且在圆肩侧肩部前屈上举角度 < 178° 和/或 外展上举角度 < 178°"
            ]
        },
        "头前引可能是由骨盆前倾或前移导致": {
            "优先级": 2,
            "异常判断流程": [
                "存在高低肩，且存在头歪、头前引或圆肩中至少一项，且（存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或存在头前引且存在高低肩，或存在头前引、存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或头歪向一侧、高低肩、圆肩及头前引这四项中至少两项同时存在且这些症状出现在身体同侧），且存在骨盆前移的可能性，并且头前引程度超标"
            ]
        },
        "圆肩侧胸小肌、胸大肌紧张": {
            "优先级": 5,
            "异常判断流程": [
                "存在单侧圆肩，且圆肩侧 肩部前伸上举>=175°"
            ]
        },
        "圆肩侧胸小肌、胸大肌紧张三角肌中束、后束力量不足，大圆肌紧张": {
            "优先级": 5,
            "异常判断流程": [
                "存在单侧圆肩，且圆肩侧 肩部前伸上举<175°"
            ]
        },
        "头侧歪侧 胸锁乳突肌、斜角肌紧张": {
            "优先级": 5,
            "异常判断流程": [
                "存在单侧圆肩，且圆肩侧 肩部前伸上举>=175°，且头前引超标或头侧歪超标，且头侧歪和圆肩为同侧",
                "存在单侧圆肩，且圆肩侧 肩部前伸上举<175°，且头前引超标或头侧歪超标，且头侧歪和圆肩为同侧"
            ]
        },
        "头侧歪的 对侧的胸锁乳突肌、同侧斜角肌": {
            "优先级": 5,
            "异常判断流程": [
                "存在单侧圆肩，且圆肩侧 肩部前伸上举>=175°，且头前引超标或头侧歪超标，且头侧歪和圆肩为异侧",
                "存在单侧圆肩，且圆肩侧 肩部前伸上举<175°，且头前引超标或头侧歪超标，且头侧歪和圆肩为异侧"
            ]
        },
        "上交叉综合征": {
            "优先级": 4,
            "异常判断流程": [
                "存在双侧圆肩，且头前引超标"
            ]
        },
        "上下交叉综合征": {
            "优先级": 1,
            "异常判断流程": [
                "存在双侧圆肩，且头前引超标，且可能骨盆前倾"
            ]
        },
        "双侧胸小肌、胸大肌、三角肌前束与菱形肌、下斜方、三角肌后束、冈下肌和小圆肌力量不对称": {
            "优先级": 5,
            "异常判断流程": [
                "不存在圆肩，且头前引不超标"
            ]
        },
        "梨形臀可能由骨盆前倾诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在骨盆前倾，且是梨形臀，且体脂率<35%且BMI<24"
            ]
        },
        "骨盆前倾和膝超伸可能是由足弓较低诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在骨盆前倾，且双膝膝超伸，且腿型角度双侧<180°"
            ]
        },
        "膝超伸可能是由骨盆前倾诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在骨盆前倾，且双膝膝超伸，且腿型角度双侧>180°"
            ]
        },
        "腰椎过度前凸是由骨盆前倾导诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在骨盆前倾，且腰椎前凸过度"
            ]
        },
        "胸椎过度后凸可能是由于骨盆前倾诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在骨盆前倾，且腰椎前凸过度，且胸椎后凸过度"
            ]
        },
        "脊椎曲度在矢状面上过度位移": {
            "优先级": 3,
            "异常判断流程": [
                "存在骨盆前倾，且s1在C7后方超过正常范围(>4cm)",
                "存在骨盆前倾，且双侧圆肩且头前倾，且s1在C7后方超过正常范围(>4cm)"
            ]
        },
        "上下交叉综合征，上交叉综合征可能由骨盆前倾诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在骨盆前倾，且双侧圆肩且头前倾",
                "存在骨盆前倾，且骨盆前移"
            ]
        },
        "骨盆后倾可能由骨盆前移诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在骨盆后倾，且骨盆前移"
            ]
        },
        "腰椎前凸不足由骨盆后倾诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在骨盆后倾，且腰椎前凸不足"
            ]
        },
        "平背姿态可能由骨盆后倾诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在骨盆后倾，且胸椎前凸不足"
            ]
        },
        "颈椎过度前凸由平背诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在骨盆后倾，且胸椎前凸不足，且头前倾数值为负"
            ]
        },
        "可能存在高足弓": {
            "优先级": 3,
            "异常判断流程": [
                "存在骨盆后倾，且双膝膝关节角度均<180°，且腿型角度双侧>180°"
            ]
        },
        "臀型可能是由骨盆后倾诱发": {
            "优先级": 3,
            "异常判断流程": [
                "存在骨盆后倾，且方形臀"
            ]
        },
        "高低肩是由长短腿 诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在长短腿，且存在高低肩，且长短腿(短腿侧肩膀高)或躯干倾斜(躯干倾斜侧肩膀高)或中心不平衡(足底压力低侧肩膀高)"
            ]
        },
        "高低肩是由躯干倾斜诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在躯干倾斜，且存在高低肩，且长短腿(短腿侧肩膀高)或躯干倾斜(躯干倾斜侧肩膀高)或中心不平衡(足底压力低侧肩膀高)"
            ]
        },
        "高低肩是由重心不平衡 诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在重心不平衡(长腿侧足底重量>短腿侧+2kg)，且存在高低肩，且长短腿(短腿侧肩膀高)或躯干倾斜(躯干倾斜侧肩膀高)或中心不平衡(足底压力低侧肩膀高)"
            ]
        },
        "高低肩是由长短腿/重心不平衡 诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在长短腿，且存在重心不平衡(长腿侧足底重量>短腿侧+2kg)，且存在高低肩，且长短腿(短腿侧肩膀高)或躯干倾斜(躯干倾斜侧肩膀高)或中心不平衡(足底压力低侧肩膀高)"
            ]
        },
        "高低肩是由长短腿/躯干倾斜 诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在长短腿，且存在躯干倾斜，且存在高低肩，且长短腿(短腿侧肩膀高)或躯干倾斜(躯干倾斜侧肩膀高)或中心不平衡(足底压力低侧肩膀高)"
            ]
        },
        "高低肩是由躯干倾斜/重心不平衡 诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在躯干倾斜，且存在重心不平衡(长腿侧足底重量>短腿侧+2kg)，且存在高低肩，且长短腿(短腿侧肩膀高)或躯干倾斜(躯干倾斜侧肩膀高)或中心不平衡(足底压力低侧肩膀高)"
            ]
        },
        "高低肩是由长短腿/躯干倾斜/重心不平衡 诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在长短腿，且存在躯干倾斜，且存在重心不平衡(长腿侧足底重量>短腿侧+2kg)，且存在高低肩，且长短腿(短腿侧肩膀高)或躯干倾斜(躯干倾斜侧肩膀高)或中心不平衡(足底压力低侧肩膀高)"
            ]
        },
        "躯干倾斜是由重心不平衡诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在长短腿，且长腿侧足底重量>短腿侧+0.5kg，且躯干向短腿侧倾斜"
            ]
        },
        "重心变化是由长短腿诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在长短腿，且长腿侧足底重量>短腿侧+0.5kg"
            ]
        },
        "脊柱横向移位是由长短腿诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在长短腿，且骶骨中点垂线较颈椎第7椎体(C7)偏向长腿侧"
            ]
        },
        "脊柱过度位移由头前引诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在头前引，且无骨盆前倾，且s1在C7后方超过正常范围(>4cm)"
            ]
        },
        "头前引是由枕后肌群紧张诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在头前引，且颈椎前屈角度不足，但其他正常"
            ]
        },
        "头前引 是由胸锁乳突肌和斜角肌紧张诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在头前引，且颈椎后伸、双侧屈曲 有一个角度不足"
            ]
        },
        "（头侧歪对侧）胸锁乳突肌紧张可能由头侧歪诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在头前引，且存在头侧歪，且颈椎朝向头侧歪侧旋转角度不足、但朝对侧侧屈角度正常"
            ]
        },
        "（头侧歪侧）斜角肌紧张": {
            "优先级": 2,
            "异常判断流程": [
                "存在头侧歪、无头前引，且颈椎向头侧歪对侧侧屈时角度足，但向头侧歪对侧旋转角度正常"
            ]
        },
        "（头侧歪侧）胸锁乳突肌紧张": {
            "优先级": 2,
            "异常判断流程": [
                "存在头侧歪、无头前引，且颈椎向头侧歪对侧侧屈时角度正常，但向头侧歪对侧旋转角度不足"
            ]
        },
        "长短腿/倾斜/足底压力由K型腿诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在腿型呈现K型，且K型腿中腿型角度不正常的腿短或躯干朝腿型角度不正常侧倾斜或正常腿侧足底重量 >短腿侧+2kg"
            ]
        },
        "长短腿/倾斜/足底压力由D型腿诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在腿型呈现D型，且D型腿中腿型角度不正常的腿短或躯干朝腿型角度不正常侧倾斜或正常腿侧足底重量 >短腿侧+2kg"
            ]
        },
        "您的臀型可能是由X型腿诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在倒三角臀，且没有骨盆前后倾，且存在x型腿"
            ]
        }
    },
    "体围": {
        "健康风险低": {
            "优先级": 4,
            "异常判断流程": [
                "腰高比(腰围/身高)<0.5",
                "腰臀比(腰围/臀围)：男性<0.9，女性<0.8"
            ]
        },
        "健康风险增加": {
            "优先级": 3,
            "异常判断流程": [
                "0.5<腰高比(腰围/身高)<0.6",
                "腰臀比(腰围/臀围)：男性0.9~0.99，女性0.8~0.84"
            ]
        },
        "健康风险大幅度增加": {
            "优先级": 2,
            "异常判断流程": [
                "腰高比(腰围/身高)>0.6",
                "腰臀比(腰围/臀围)：男性>=1，女性>=0.85"
            ]
        },
        "中心性肥胖": {
            "优先级": 1,
            "异常判断流程": [
                "腰围：亚洲:男性>90、女性>85。海外：男性≥102，女性>88"
            ]
        },
        "肌肉可能不足": {
            "优先级": 3,
            "异常判断流程": [
                "存在上臂围(左右)：男性<28.5，女性<27，或小腿围(左右)：男性<34，女性<33"
            ]
        },
        "肌肉过少": {
            "优先级": 2,
            "异常判断流程": [
                "存在上臂围(左右)：男性<28.5，女性<27，且小腿围(左右)：男性<34，女性<33"
            ]
        }
    }
}
"""

# ==================== 多智能体定义 ====================
# 功能是回复日常问题。对于日常问题来说，可以使用价格较为低廉的模型作为agent的基座
ChatAssistant = Assistants.create(
    model="qwen-turbo",
    name='回答日常问题的机器人',
    description='一个智能助手，解答用户的问题',
    instructions='请礼貌地回答用户的问题'
)

# UserDataAnalysisAssistant已删除，替换为直接的代码处理

# 第一个AI：身体异常分析助手（严格决策树分析 + 补充分析）
AbnormalityAnalysisAssistant = Assistants.create(
    model="qwen-max-latest",
    name='身体异常分析机器人',
    description='一个专业的身体异常分析助手，严格基于决策树规则进行异常判断，当决策树无法识别异常时再进行补充分析',
    instructions="""你是一个严格按照决策树规则的身体异常分析专家，必须按照以下两步流程进行分析：

【决策树规则】
""" + decision_tree + """

【核心分析流程 - 必须严格遵守】
**决策树严格分析**
1. **绝对禁止推测**：只能输出决策树中明确存在的异常名称
2. **条件必须完全满足**：用户数据必须完全符合决策树条件才能得出异常结论
3. **逐条验证**：必须逐一检查决策树中的每个条件，进行严格数值计算和验证
4. **判断条件**：有多条判断路径，满足其中一条路径即可
5. **逻辑关系严格执行**：
   - "且"关系：所有条件必须同时满足，任何一个条件不满足则整个异常判断为假
   - "或"关系：至少一个条件满足即可
   - 条件组合：严格按照括号和逻辑连接词执行
6. **系统性检查**：必须逐一检查决策树中的所有异常类型，不能遗漏
7. **明确标注**：所有通过决策树识别的异常必须明确标注为"决策树识别"
8. **严格验证原则**：如果任何一个必要条件不满足，绝对不能输出该异常，即使其他条件满足


【输出格式要求】
必须严格按照以下JSON格式输出，不得添加任何额外文字说明：

```json
{
  "analysis_type": "决策树严格分析",
  "systematic_check": [
    {
      "abnormality_name": "异常名称",
      "category": "体成分" | "体态",
      "priority": 优先级数字,
      "decision_tree_condition": "决策树中的完整判断条件",
      "condition_verification": "判断流程",
      "meets_decision_tree": true | false,
      "rejection_reason": "如果不符合，说明具体原因"
    }
  ],
  "identified_abnormalities": {
    "body_composition": [
      {
        "abnormality_name": "异常名称",
        "priority": 优先级数字,
        "identification_source": "决策树识别"
      }
    ],
    "posture": [
      {
        "abnormality_name": "异常名称", 
        "priority": 优先级数字,
        "identification_source": "决策树识别"
      }
    ]
  },
}
```


【严格禁止 - 违反将导致分析失效】
- 在第一步中输出决策树中不存在的异常名称
- **绝对禁止在任何一个必要条件不满足时输出该异常**（即使部分条件满足）
- 绕过条件验证过程直接给出结论
- 对"且"逻辑关系的误解（所有条件必须同时满足）
- 混淆决策树识别和补充分析识别的异常
- 在决策树已识别足够异常时仍进行补充分析
- 在系统性检查中遗漏任何决策树异常类型""",
    tools=[]
)

# 注释掉原有的KnowledgeQueryAssistant，功能合并到SummaryAssistant中
# KnowledgeQueryAssistant = Assistants.create(...)

# 在Multi Agent场景下，定义一个用于总结的Agent，该Agent会根据用户的问题与之前Agent输出的参考信息，全面、完整地回答用户问题
SummaryAssistant = Assistants.create(
    model="qwen-max-latest",
    name='身体异常总结机器人',
    description='一个专业的身体异常分析助手，负责整合异常分析结果和知识库查询结果，生成完整的异常分析综合报告',
    instructions="""你是一个专业的身体异常分析总结专家，负责整合异常分析结果和知识库查询结果，提供最终的综合报告。

【核心任务】
1. 接收异常分析结果（异常结论、判断过程）
2. 接收已查询好的知识库解决方案信息
3. 整合异常分析结果和知识库查询结果，生成包含完整信息的身体异常分析综合报告
4. 按照优先级排序所有异常

【总结格式要求】
## 身体异常完整分析报告

### 一、异常结论汇总
[列出所有检测到的异常，体成分和体态分别按优先级排序]

### 二、体成分异常详细分析
- **异常结论**: [异常名称]
- **优先级**: [决策树中的优先级]
- **判断流程**: [基于专业指标分析（不要显示具体判断阈值），给出判断流程]
- **关键解决点**: [从知识库获得的关键解决点]
- **建议**: [从知识库获得的建议]
- **症状**: [从知识库获得的症状]
- **对身体的影响**: [从知识库获得的影响分析] 

### 三、体态异常详细分析（按优先级排序）
1. **[优先级X] 异常名称**
    - **优先级**: [决策树中的优先级]
    - **判断结果**: [基于专业指标分析（不要显示具体判断阈值），给出判断流程]
    - **关键解决点**: [从知识库获得的关键解决点]
    - **建议**: [从知识库获得的建议]
    - **症状**: [从知识库获得的症状]
    - **对身体的影响**: [从知识库获得的影响分析] 

### 四、综合建议与总结
- 给出一段话的整体身体状况评估与总结

【严格要求】
- 必须整合所有分析结果和提供的知识库信息
- 体成分和体态分别按照优先级排序（数字越小优先级越高）
- 确保每个异常都有完整的信息
- 基于提供的知识库查询结果来补充解决方案、症状、影响等信息""",
    tools=[]
)

# 将工具函数的name映射到函数本体
function_mapper = {
    "初始化身体数据": MedicalAnalysis.initialize_structured_data,
    "异常解决方案查询": MedicalAnalysis.query_medical_knowledge,
}

# 将Agent的name映射到Agent本体
assistant_mapper = {
    "ChatAssistant": ChatAssistant,
    "AbnormalityAnalysisAssistant": AbnormalityAnalysisAssistant
}

# ==================== Agent处理函数 ====================

def get_agent_response(assistant, message='', return_tool_output=False, knowledge_base=None):
    """输入message信息，输出为指定Agent的回复"""
    #print(f"Query: {message}")
    thread = Threads.create()
    message = Messages.create(thread.id, content=message)
    run = Runs.create(thread.id, assistant_id=assistant.id)
    run_status = Runs.wait(run.id, thread_id=thread.id)
    
    all_tool_output = ""  # 存储所有工具输出
    print("run_status:",run_status)
    
    if run_status.status == 'failed':
        print('run failed:')
        return ("抱歉，处理过程中出现错误", all_tool_output) if return_tool_output else "抱歉，处理过程中出现错误"
    
    # 🔥 循环处理多个工具调用
    while run_status.required_action:
        tool_calls = run_status.required_action.submit_tool_outputs.tool_calls
        tool_outputs = []
        
        # 处理多个工具调用
        for tool_call in tool_calls:
            f = tool_call.function
            func_name = f['name'] 
            print(f"f: {f}")
            param = json.loads(f['arguments'])
            print(f"调用工具: {func_name}")
            print(f"工具参数: {param}")
            
            # 特别关注异常解决方案查询工具的调用
            if func_name == "异常解决方案查询":
                print(f"知识库查询文本: {param.get('query_text', '未提供')}")
                print(f"知识库名称: {param.get('knowledge_base_name', '未提供')}")
        
            if func_name in function_mapper:
                # 如果是解决方案查询，添加知识库参数
                if func_name == "异常解决方案查询" and 'knowledge_base_name' not in param:
                    # 使用传递的knowledge_base参数
                    if knowledge_base:
                        param['knowledge_base_name'] = knowledge_base
                        print(f"设置知识库参数: {knowledge_base}")
                    else:
                        print("警告：未提供知识库参数，将使用默认知识库")
                
                try:
                    output = function_mapper[func_name](**param)
                    # 确保输出不为空
                    if output is None:
                        output = '{"error": "工具函数返回空值"}'
                    elif not isinstance(output, str):
                        output = str(output)
                    
                    all_tool_output += f"{func_name}: {output}\n"  # 累积工具输出
                    print(f"工具 {func_name} 执行成功")
                except Exception as e:
                    print(f"工具函数执行失败 {func_name}: {e}")
                    output = f'{{"error": "工具函数执行失败: {str(e)}"}}'
                    all_tool_output += f"{func_name}: {output}\n"
            else:    
                output = '{"error": "未知的工具函数"}'
                print(f"未知工具函数: {func_name}")
            
            tool_outputs.append({
                'tool_call_id': tool_call.id,
                'output': output
            })
        
        # 提交工具输出
        run = Runs.submit_tool_outputs(run.id,
                                       thread_id=thread.id,
                                       tool_outputs=tool_outputs)
        run_status = Runs.wait(run.id, thread_id=thread.id)
        print(f"工具调用完成，状态: {run_status.status}")
    
    # 获取最终响应
    run_status = Runs.get(run.id, thread_id=thread.id)
    msgs = Messages.list(thread.id)
    response = msgs['data'][0]['content'][0]['text']['value']
    
    if return_tool_output:
        return response, all_tool_output
    else:
        return response

def get_multi_agent_response_internal(query, knowledge_base=None):
    """获得Multi Agent的回复的内部函数"""
    if len(query) == 0:
        return "请输入您的身体数据或问题", ""
    
    collected_knowledge_chunks = ""  # 收集知识库召回信息
    
    try:
        order_stk = ["AbnormalityAnalysisAssistant"]
        
        # 提取用户身体数据（从原始query中）
        user_body_data = ""
        if "请分析以下身体数据" in query:
            # 提取JSON数据部分
            import re
            json_match = re.search(r'：(\{.*\})$', query)
            if json_match:
                user_body_data = json_match.group(1)
            else:
                user_body_data = query
        else:
            user_body_data = query
        
        Agent_Message = ""
        previous_responses = {}  # 存储各个Agent的响应
        
        # 直接初始化结构化身体数据，跳过UserDataAnalysisAssistant
        print("直接初始化结构化身体数据...")
        user_analysis = MedicalAnalysis.initialize_structured_data(user_body_data)
        previous_responses["UserDataAnalysisAssistant"] = user_analysis
        Agent_Message += f"*直接数据初始化*的结果为：{user_analysis}\n\n"

        
        # 依次运行Agent
        for i in range(len(order_stk)):
            assistant_name = order_stk[i]
            cur_assistant = assistant_mapper[assistant_name]
            
            # 为不同的Assistant定制专门的查询内容
            if assistant_name == "AbnormalityAnalysisAssistant":
                # 异常分析Assistant，直接使用决策树和用户数据分析异常
                user_analysis = previous_responses.get("UserDataAnalysisAssistant", "")
                cur_query = f"请基于以下用户身体数据，严格按照决策树规则分析体成分异常和体态异常，并按优先级排序。\n\n用户身体数据：{user_analysis}\n\n请详细显示每个异常的判断过程和数据匹配情况。"
            else:
                # 其他Assistant保持原始查询
                cur_query = query
            
            print(f"{assistant_name}助手开始工作，专门任务：{cur_query}")
            
            # 调用Assistant
            response = get_agent_response(cur_assistant, cur_query, knowledge_base=knowledge_base)
            
            # 存储当前Assistant的响应
            previous_responses[assistant_name] = response
            Agent_Message += f"*{assistant_name}*的回复为：{response}\n\n"
            print(f"*{assistant_name}*的回复为：{response}")
            
        # 提取异常结论作为知识库查询条件
        abnormalities_for_query = []
        
        # 从AbnormalityAnalysisAssistant的响应中提取异常结论
        abnormality_response = previous_responses.get("AbnormalityAnalysisAssistant", "")
        try:
            # 尝试从响应中提取JSON部分
            import re
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, abnormality_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                abnormality_data = json.loads(json_str)
                
                # 提取体成分异常
                if "identified_abnormalities" in abnormality_data:
                    body_comp = abnormality_data["identified_abnormalities"].get("body_composition", [])
                    posture = abnormality_data["identified_abnormalities"].get("posture", [])
                    
                    for abnormality in body_comp:
                        abnormalities_for_query.append(abnormality.get("abnormality_name", ""))
                    
                    for abnormality in posture:
                        abnormalities_for_query.append(abnormality.get("abnormality_name", ""))
                
                print(f"提取到的异常结论用于知识库查询: {abnormalities_for_query}")
            else:
                print("未找到JSON格式的异常分析结果，将使用原始响应进行知识库查询")
                # 如果无法解析JSON，作为备选方案使用原始响应
                abnormalities_for_query = [abnormality_response[:200]]  # 使用前200字符作为查询
                
        except Exception as e:
            print(f"解析异常分析结果时出错: {e}")
            # 如果解析失败，作为备选方案使用关键词
            abnormalities_for_query = ["体成分异常", "体态异常", "身体健康问题"]
        
        # 构建知识库查询文本 - 只使用提取的异常名称
        if abnormalities_for_query:
            # 过滤掉空字符串
            valid_abnormalities = [ab for ab in abnormalities_for_query if ab and ab.strip()]
            if valid_abnormalities:
                query_text_for_kb = " ".join(valid_abnormalities)
                print(f"最终知识库查询文本: {query_text_for_kb}")
            else:
                query_text_for_kb = "身体异常 健康问题 解决方案"
        else:
            query_text_for_kb = "身体异常 健康问题 解决方案"
        
        # 直接调用知识库查询，而不是通过agent工具调用
        print("开始直接查询知识库...")
        knowledge_query_result = ""
        try:
            # 直接调用知识库查询函数
            knowledge_query_result = MedicalAnalysis.query_medical_knowledge(
                query_text=query_text_for_kb,
                knowledge_base_name=knowledge_base
            )
            print(f"知识库查询完成，结果长度: {len(knowledge_query_result)}")
            collected_knowledge_chunks = f"知识库查询结果：{knowledge_query_result}"
        except Exception as e:
            print(f"知识库查询失败: {e}")
            knowledge_query_result = "知识库查询失败，请检查相关配置"
            collected_knowledge_chunks = "知识库查询失败"
        
        # 所有Agent运行完毕后，调用SummaryAssistant进行最终总结
        # 为SummaryAssistant准备包含知识库查询结果的提示
        summary_prompt = f"""请基于以下异常分析结果和知识库查询结果，提供最终的身体异常完整分析报告。

原始用户问题：{query}

异常分析结果：
{Agent_Message}

知识库查询结果：
{knowledge_query_result}

请整合异常分析结果和知识库查询结果，生成包含异常结论、分析过程、解决方案、健康影响等完整信息的综合报告。所有解决方案、症状、影响等信息都应基于上述知识库查询结果。"""
        
        # 调用SummaryAssistant，不再需要工具调用
        multi_agent_response = get_agent_response(SummaryAssistant, summary_prompt, knowledge_base=knowledge_base)
        
        # 确保有召回文本段显示
        if not collected_knowledge_chunks:
            collected_knowledge_chunks = "多智能体模式：已完成异常解决方案查询，但未检索到足够相关的内容。建议提供更详细的身体数据或咨询专业健康顾问。"
        
        return multi_agent_response, collected_knowledge_chunks
    
    except Exception as e:
        print(f"Multi-agent processing failed: {e}")
        # 兜底策略，如果上述程序运行失败，则直接调用ChatAssistant
        fallback_response = get_agent_response(ChatAssistant, query, knowledge_base=knowledge_base)
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
        response, knowledge_chunks = get_multi_agent_response_internal(query, "异常2")
        print("=== 多智能体分析结果 ===")
        print(f"分析结果：{response}")
        #print("\n=== 知识库召回信息 ===")
        #print(knowledge_chunks)
        return response, knowledge_chunks
    except Exception as e:
        print(f"测试失败：{e}")
        return None, None

if __name__ == "__main__":
    # 运行测试
    test_body_analysis()