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

tichengfen_tiwei_decision_tree = """
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

# 体态异常决策树（有骨盆）
posture_decision_tree = """
{
    "体态": {
        "可能骨盆旋移": {
            "优先级": 3,
            "异常判断流程": [
                "骨盆前移角度≤179或骨盆前移距离≥2cm且双侧膝关节角度差异相差≥5°。同时较大腿型角度一侧的膝关节角度比较小膝关节一侧腿型角度相差>3"
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
        "头侧歪/高低肩/头前引 均由骨盆旋移诱发": {
            "优先级": 2,
            "异常判断流程": [
                "存在高低肩，且存在头歪、头前引或圆肩中至少一项，且（存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或存在头前引且存在高低肩，或存在头前引、存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或头歪向一侧、高低肩、圆肩及头前引这四项中至少两项同时存在且这些症状出现在身体同侧），且存在骨盆前移的可能性"
            ]
        },
        "头前引可能是由骨盆前倾或前移导致": {
            "优先级": 2,
            "异常判断流程": [
                "存在高低肩，且存在头歪、头前引或圆肩中至少一项，且（存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或存在头前引且存在高低肩，或存在头前引、存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或头歪向一侧、高低肩、圆肩及头前引这四项中至少两项同时存在且这些症状出现在身体同侧），且存在骨盆前移的可能性，并且头前引程度超标"
            ]
        },
        "上下交叉综合征": {
            "优先级": 1,
            "异常判断流程": [
                "存在双侧圆肩，且头前引超标，且可能骨盆前倾"
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
        "您的臀型可能是由X型腿诱发": {
            "优先级": 1,
            "异常判断流程": [
                "存在倒三角臀，且没有骨盆前后倾，且存在x型腿"
            ]
        }
    }
        
}
        

"""

# 体态异常决策树（无骨盆）
posture_decision_tree_no_pelvis = """
{
    "体态": {
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

        "单圆肩侧胸小肌、胸锁乳突肌、斜角肌、前锯肌紧张": {
            "优先级": 5,
            "异常判断流程": [
                "存在高低肩，且存在头歪、头前引或圆肩中至少一项，且（存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或存在头前引且存在高低肩，或存在头前引、存在头歪向一侧且存在高低肩并且高肩侧与头歪侧相反，或头歪向一侧、高低肩、圆肩及头前引这四项中至少两项同时存在且这些症状出现在身体同侧），且在圆肩侧肩部前屈上举角度 < 178° 和/或 外展上举角度 < 178°"
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

        "双侧胸小肌、胸大肌、三角肌前束与菱形肌、下斜方、三角肌后束、冈下肌和小圆肌力量不对称": {
            "优先级": 5,
            "异常判断流程": [
                "不存在圆肩，且头前引不超标"
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
        }
    }
}
"""

# ==================== AI助手配置定义 ====================
# 功能是回复日常问题
ChatAssistant = {
    "model": "qwen-turbo",
    "name": '回答日常问题的机器人',
    "description": '一个智能助手，解答用户的问题',
    "instructions": '请礼貌地回答用户的问题'
}

# 体成分+体围异常分析助手
BodyCompositionAnalysisAssistant = {
    "model": "qwen-max-latest",
    "name": '体成分体围异常分析机器人',
    "description": '专门负责体成分和体围异常分析的助手，严格基于决策树规则进行异常判断',
    "instructions": f"""你是一个严格按照决策树规则的体成分和体围异常分析专家，专门负责体成分和体围相关异常的判断。

【决策树规则】
{tichengfen_tiwei_decision_tree}

【核心分析流程 - 必须严格遵守】
**决策树严格分析**
1. **绝对禁止推测**：只能输出决策树中明确存在的异常名称
2. **判断条件**：有多条判断路径的情况，满足其中一条路径即可
3. **逻辑关系严格执行**：
   - "且"关系：所有条件必须同时满足，任何一个条件不满足则整个异常判断为假
   - "或"关系：至少一个条件满足即可
   - 异常判断流程中可能有多条路径，需要某一条路径完全满足，才能输出该异常
4. **严格验证原则**：如果任何一个必要条件不满足，绝对不能输出该异常，即使其他条件满足
5. **确保完整性**：确保对所有异常都进行了判断，不要遗漏

【输出格式要求 - 优先输出符合条件的异常】
必须严格按照以下JSON格式输出，不得添加任何额外文字说明：

```json
{{
  "analysis_type": "体成分体围决策树分析",
  "analysis_category": "body_composition_girth",
  "identified_abnormalities": {{
    "body_composition": [
      {{
        "abnormality_name": "异常名称",
        "priority": 优先级数字,
        "condition_verification": "用户数据带入决策树对比过程",
        "meets_decision_tree": true | false,
      }}
    ],
    "girth": [
      {{
        "abnormality_name": "异常名称", 
        "priority": 优先级数字,
        "condition_verification": "用户数据带入决策树对比过程"
        "meets_decision_tree": true | false,
      }}
    ]
  }},
}}
```

【重要输出原则】
1. **优先输出符合决策树的异常**：identified_abnormalities部分放在前面且详细描述
2. **按优先级排序**：所有符合的异常必须按优先级从高到低排序（数字越小优先级越高）
3. **确保完整性**：确保输出所有符合的异常，不要遗漏
4. **条件验证**：condition_verification必须详细描述用户数据带入决策树对比过程,尽量隐藏决策树的具体数值,只描述用户数据与决策树条件对比过程

【严格禁止 - 违反将导致分析失效】
- 在分析中输出决策树中不存在的异常名称
- **绝对禁止在任何一个必要条件不满足时输出该异常**（即使部分条件满足）
- 绕过条件验证过程直接给出结论
- 对"且"逻辑关系的误解（所有条件必须同时满足）
- 输出不符合条件的异常分析过程"""
}

# 体态异常分析助手（有骨盆）
PostureAnalysisAssistant = {
    "model": "qwen-max-latest", 
    "name": '体态异常分析机器人（有骨盆）',
    "description": '专门负责有骨盆体态异常分析的助手，严格基于决策树规则进行异常判断',
    "instructions": f"""你是一个严格按照决策树规则的体态异常分析专家，专门负责有骨盆相关的体态异常判断。

【决策树规则】
{posture_decision_tree}

【核心分析流程 - 必须严格遵守】
**决策树严格分析**
1. **绝对禁止推测**：只能输出决策树中明确存在的异常名称
2. **判断条件**：有多条判断路径的情况，满足其中一条路径即可
3. **逻辑关系严格执行**：
   - "且"关系：所有条件必须同时满足，任何一个条件不满足则整个异常判断为假
   - "或"关系：至少一个条件满足即可
   - 异常判断流程中可能有多条路径，需要某一条路径完全满足，才能输出该异常
4. **严格验证原则**：如果任何一个必要条件不满足，绝对不能输出该异常，即使其他条件满足
5. **确保完整性**：确保对所有异常都进行了判断，不要遗漏

【异常名称具体化要求】
1. **头侧歪侧方向具体化**：
   - 如果涉及"头侧歪侧"，根据数据判断是左侧还是右侧，替换为"左侧"或"右侧"
   - 例如："头侧歪侧斜角肌紧张" → "左侧斜角肌紧张"（如果头侧歪向左侧）

2. **多重条件具体化**：
   - 如果异常名称包含"头侧歪/头前引"等多重条件，只保留实际存在的症状
   - 例如："头侧歪/头前引是由xxx诱发" → "头侧歪是由xxx诱发"（如果只存在头侧歪）
   - 如果两者都存在："头侧歪和头前引是由xxx诱发"

3. **高肩侧/低肩侧具体化**：
   - 根据高低肩数据，将"高肩侧"替换为"左肩"或"右肩"
   - 将"低肩侧"替换为"左肩"或"右肩"

4. **其他侧别具体化**：
   - 圆肩侧、腿型异常侧等都要根据实际数据具体化

【输出格式要求 - 优先输出符合条件的异常】
必须严格按照以下JSON格式输出，不得添加任何额外文字说明：

```json
{{
  "analysis_type": "体态决策树分析（有骨盆）",
  "analysis_category": "posture_with_pelvis",
  "identified_abnormalities": {{
    "posture": [
      {{
        "original_name": "决策树中的原始异常名称", 
        "specific_name": "根据实际情况具体化的异常名称",
        "priority": 优先级数字,
        "condition_verification": "用户数据带入决策树对比过程",
        "meets_decision_tree": true | false,
      }}
    ]
  }},
}}
```

【重要输出原则】
1. **优先输出符合决策树的异常**：identified_abnormalities部分放在前面且详细描述
2. **按优先级排序**：所有符合的异常必须按优先级从高到低排序（数字越小优先级越高）
3. **确保完整性**：确保输出所有符合的异常，不要遗漏
4. **名称具体化**：original_name保持决策树原始名称，specific_name根据实际数据具体化
5. **条件验证**：condition_verification必须详细描述用户数据带入决策树对比过程,尽量隐藏决策树的具体数值,只描述用户数据与决策树条件对比过程

【严格禁止 - 违反将导致分析失效】
- 在分析中输出决策树中不存在的异常名称
- **绝对禁止在任何一个必要条件不满足时输出该异常**（即使部分条件满足）
- 绕过条件验证过程直接给出结论
- 对"且"逻辑关系的误解（所有条件必须同时满足）
- 输出不符合条件的异常分析过程"""
}

# 体态异常分析助手（无骨盆）
PostureAnalysisNoPelvisAssistant = {
    "model": "qwen-max-latest", 
    "name": '体态异常分析机器人（无骨盆）',
    "description": '专门负责无骨盆体态异常分析的助手，严格基于决策树规则进行异常判断',
    "instructions": f"""你是一个严格按照决策树规则的体态异常分析专家，专门负责无骨盆相关的体态异常判断。

【决策树规则】
{posture_decision_tree_no_pelvis}

【核心分析流程 - 必须严格遵守】
**决策树严格分析**
1. **绝对禁止推测**：只能输出决策树中明确存在的异常名称
2. **判断条件**：有多条判断路径的情况，满足其中一条路径即可
3. **逻辑关系严格执行**：
   - "且"关系：所有条件必须同时满足，任何一个条件不满足则整个异常判断为假
   - "或"关系：至少一个条件满足即可
   - 异常判断流程中可能有多条路径，需要某一条路径完全满足，才能输出该异常
4. **严格验证原则**：如果任何一个必要条件不满足，绝对不能输出该异常，即使其他条件满足
5. **确保完整性**：确保对所有异常都进行了判断，不要遗漏

【异常名称具体化要求】
1. **头侧歪侧方向具体化**：
   - 如果涉及"头侧歪侧"，根据数据判断是左侧还是右侧，替换为"左侧"或"右侧"
   - 例如："头侧歪侧斜角肌紧张" → "左侧斜角肌紧张"（如果头侧歪向左侧）

2. **多重条件具体化**：
   - 如果异常名称包含"头侧歪/头前引"等多重条件，只保留实际存在的症状
   - 例如："头侧歪/头前引是由xxx诱发" → "头侧歪是由xxx诱发"（如果只存在头侧歪）
   - 如果两者都存在："头侧歪和头前引是由xxx诱发"

3. **高肩侧/低肩侧具体化**：
   - 根据高低肩数据，将"高肩侧"替换为"左肩"或"右肩"
   - 将"低肩侧"替换为"左肩"或"右肩"

4. **对侧关系具体化**：
   - "头侧歪的对侧" → 如果头侧歪向左，则对侧为"右侧"
   - "高肩一侧/低肩一侧" → 根据实际高低肩情况具体化

5. **其他侧别具体化**：
   - 圆肩侧、腿型异常侧、K型腿/D型腿的具体侧别等

【输出格式要求 - 优先输出符合条件的异常】
必须严格按照以下JSON格式输出，不得添加任何额外文字说明：

```json
{{
  "analysis_type": "体态决策树分析（无骨盆）",
  "analysis_category": "posture_no_pelvis",
  "identified_abnormalities": {{
    "posture": [
      {{
        "original_name": "决策树中的原始异常名称", 
        "specific_name": "根据实际情况具体化的异常名称",
        "priority": 优先级数字,
        "condition_verification": "用户数据带入决策树对比过程",
        "meets_decision_tree": true | false,
      }}
    ]
  }},
}}
```

【重要输出原则】
1. **优先输出符合决策树的异常**：identified_abnormalities部分放在前面且详细描述
2. **按优先级排序**：所有符合的异常必须按优先级从高到低排序（数字越小优先级越高）
3. **确保完整性**：确保输出所有符合的异常，不要遗漏
4. **名称具体化**：original_name保持决策树原始名称，specific_name根据实际数据具体化
5. **条件验证**：condition_verification必须详细描述用户数据带入决策树对比过程,尽量隐藏决策树的具体数值,只描述用户数据与决策树条件对比过程

【严格禁止 - 违反将导致分析失效】
- 在分析中输出决策树中不存在的异常名称
- **绝对禁止在任何一个必要条件不满足时输出该异常**（即使部分条件满足）
- 绕过条件验证过程直接给出结论
- 对"且"逻辑关系的误解（所有条件必须同时满足）
- 输出不符合条件的异常分析过程"""
}

# 将工具函数的name映射到函数本体
function_mapper = {
    "初始化身体数据": MedicalAnalysis.initialize_structured_data,
    "异常解决方案查询": MedicalAnalysis.query_medical_knowledge,
}

# 将助手配置的name映射到配置本体
assistant_mapper = {
    "ChatAssistant": ChatAssistant,
    "BodyCompositionAnalysisAssistant": BodyCompositionAnalysisAssistant,
    "PostureAnalysisAssistant": PostureAnalysisAssistant,
    "PostureAnalysisNoPelvisAssistant": PostureAnalysisNoPelvisAssistant
}

# ==================== Agent处理函数 ====================

import concurrent.futures
import threading

def analyze_abnormalities_concurrently(user_body_data, knowledge_base=None):
    """并发执行体成分+体围、有骨盆体态和无骨盆体态异常分析"""
    
    import os
    from datetime import datetime
    
    # 创建输出目录
    output_dir = "analysis_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def analyze_body_composition_girth(user_data):
        """分析体成分和体围异常"""
        assistant_config = assistant_mapper["BodyCompositionAnalysisAssistant"]
        query = f"请基于以下用户身体数据，严格按照决策树规则分析体成分和体围异常。\n\n用户身体数据：{user_data}\n\n请详细显示每个异常的判断过程和数据匹配情况。"
        
        response = ""
        print(f"开始体成分+体围分析...")
        for chunk in get_agent_response_stream(assistant_config, query, knowledge_base=knowledge_base):
            response += chunk
        
        # 保存到txt文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/body_composition_girth_analysis_{timestamp}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"体成分+体围异常分析结果\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                f.write(f"查询内容:\n{query}\n\n")
                f.write("="*50 + "\n\n")
                f.write(f"分析结果:\n{response}\n")
            print(f"体成分+体围分析结果已保存到: {filename}")
        except Exception as e:
            print(f"保存体成分+体围分析结果失败: {e}")
        
        return "BodyComposition", response
    
    def analyze_posture_with_pelvis(user_data):
        """分析有骨盆体态异常"""
        assistant_config = assistant_mapper["PostureAnalysisAssistant"] 
        query = f"请基于以下用户身体数据，严格按照决策树规则分析有骨盆相关的体态异常。\n\n用户身体数据：{user_data}\n\n请详细显示每个异常的判断过程和数据匹配情况。"
        
        response = ""
        print(f"开始有骨盆体态分析...")
        for chunk in get_agent_response_stream(assistant_config, query, knowledge_base=knowledge_base):
            response += chunk
        
        # 保存到txt文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/posture_with_pelvis_analysis_{timestamp}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"有骨盆体态异常分析结果\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                f.write(f"查询内容:\n{query}\n\n")
                f.write("="*50 + "\n\n")
                f.write(f"分析结果:\n{response}\n")
            print(f"有骨盆体态分析结果已保存到: {filename}")
        except Exception as e:
            print(f"保存有骨盆体态分析结果失败: {e}")
        
        return "PostureWithPelvis", response
    
    def analyze_posture_no_pelvis(user_data):
        """分析无骨盆体态异常"""
        assistant_config = assistant_mapper["PostureAnalysisNoPelvisAssistant"] 
        query = f"请基于以下用户身体数据，严格按照决策树规则分析无骨盆相关的体态异常。\n\n用户身体数据：{user_data}\n\n请详细显示每个异常的判断过程和数据匹配情况。"
        
        response = ""
        print(f"开始无骨盆体态分析...")
        for chunk in get_agent_response_stream(assistant_config, query, knowledge_base=knowledge_base):
            response += chunk
        
        # 保存到txt文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{output_dir}/posture_no_pelvis_analysis_{timestamp}.txt"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"无骨盆体态异常分析结果\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                f.write(f"查询内容:\n{query}\n\n")
                f.write("="*50 + "\n\n")
                f.write(f"分析结果:\n{response}\n")
            print(f"无骨盆体态分析结果已保存到: {filename}")
        except Exception as e:
            print(f"保存无骨盆体态分析结果失败: {e}")
        
        return "PostureNoPelvis", response
    
    # 使用线程池并发执行三个分析任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # 提交三个任务
        #组合用户体成分体围数据
        user_body_data = json.loads(user_body_data)
        
        # 任务1：体成分+体围分析
        user_data_body_comp = {}
        user_data_body_comp["user_info"] = user_body_data["user_info"]
        user_data_body_comp["body_composition"] = user_body_data["body_composition"]
        user_data_body_comp["girth_info"] = user_body_data["girth_info"]
        future_body_comp = executor.submit(analyze_body_composition_girth, user_data_body_comp)
        
        # 任务2和3：体态分析（有骨盆和无骨盆都使用相同的体态数据）
        user_data_posture = {}
        user_data_posture["user_info"] = user_body_data["user_info"]
        user_data_posture["posture_metrics"] = user_body_data["posture_metrics"]
        user_data_posture["posture_conclusion"] = user_body_data["posture_conclusion"]
        user_data_posture["shoulder_info"] = user_body_data["shoulder_info"]
        
        future_posture_with_pelvis = executor.submit(analyze_posture_with_pelvis, user_data_posture)
        future_posture_no_pelvis = executor.submit(analyze_posture_no_pelvis, user_data_posture)
        
        # 等待所有任务完成并收集结果
        results = {}
        
        # 等待体成分+体围分析结果
        try:
            analysis_type, response = future_body_comp.result(timeout=180)
            results[analysis_type] = response
            print(f"\n{analysis_type} 分析完成，响应长度: {len(response)}")
        except Exception as e:
            print(f"体成分+体围分析失败: {e}")
            results["BodyComposition"] = f"分析失败: {str(e)}"
        
        # 等待有骨盆体态分析结果
        try:
            analysis_type, response = future_posture_with_pelvis.result(timeout=280)
            results[analysis_type] = response
            print(f"\n{analysis_type} 分析完成，响应长度: {len(response)}")
        except Exception as e:
            print(f"有骨盆体态分析失败: {e}")
            results["PostureWithPelvis"] = f"分析失败: {str(e)}"
        
        # 等待无骨盆体态分析结果
        try:
            analysis_type, response = future_posture_no_pelvis.result(timeout=180)
            results[analysis_type] = response
            print(f"\n{analysis_type} 分析完成，响应长度: {len(response)}")
        except Exception as e:
            print(f"无骨盆体态分析失败: {e}")
            results["PostureNoPelvis"] = f"分析失败: {str(e)}"
    
    # 保存并发分析汇总结果
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"{output_dir}/concurrent_analysis_summary_{timestamp}.txt"
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write(f"三项并发异常分析汇总\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for analysis_type, response in results.items():
                f.write(f"【{analysis_type} 分析结果】\n")
                f.write("-"*40 + "\n")
                f.write(f"{response}\n\n")
                f.write("="*60 + "\n\n")
        
        print(f"三项并发分析汇总已保存到: {summary_filename}")
    except Exception as e:
        print(f"保存三项并发分析汇总失败: {e}")
    
    return results

def merge_abnormality_results(analysis_results):
    """合并三项并发分析的结果为统一的JSON格式，只保留meets_decision_tree为true的异常"""
    merged_result = {
        "analysis_type": "三项并发决策树严格分析",
        "identified_abnormalities": {
            "body_composition": [],
            "girth": [], 
            "posture": []
        }
    }
    
    abnormalities_for_query = []  # 收集异常名称用于知识库查询
    
    for analysis_type, response in analysis_results.items():
        try:
            # 尝试从响应中提取JSON部分
            import re
            json_pattern = r'```json\s*(.*?)\s*```'
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                analysis_data = json.loads(json_str)
                
                # 合并identified_abnormalities，只保留meets_decision_tree为true的异常
                if "identified_abnormalities" in analysis_data:
                    abnormalities = analysis_data["identified_abnormalities"]
                    
                    # 合并体成分异常 - 只保留meets_decision_tree为true的
                    if "body_composition" in abnormalities:
                        for ab in abnormalities["body_composition"]:
                            if ab.get("meets_decision_tree", False) == True:
                                merged_result["identified_abnormalities"]["body_composition"].append(ab)
                                # 获取异常名称用于知识库查询
                                abnormality_name = ab.get("abnormality_name", "")
                                if abnormality_name:
                                    abnormalities_for_query.append(abnormality_name)
                    
                    # 合并体围异常 - 只保留meets_decision_tree为true的
                    if "girth" in abnormalities:
                        for ab in abnormalities["girth"]:
                            if ab.get("meets_decision_tree", False) == True:
                                merged_result["identified_abnormalities"]["girth"].append(ab)
                                # 获取异常名称用于知识库查询
                                abnormality_name = ab.get("abnormality_name", "")
                                if abnormality_name:
                                    abnormalities_for_query.append(abnormality_name)
                    
                    # 合并体态异常（包括有骨盆和无骨盆） - 只保留meets_decision_tree为true的
                    if "posture" in abnormalities:
                        for ab in abnormalities["posture"]:
                            if ab.get("meets_decision_tree", False) == True:
                                merged_result["identified_abnormalities"]["posture"].append(ab)
                                # 兼容新旧格式的异常名称，优先使用具体化名称
                                abnormality_name = ab.get("abnormality_name", "")
                                if abnormality_name:
                                    abnormalities_for_query.append(abnormality_name)
                
                
                print(f"成功解析 {analysis_type} 的分析结果，已过滤为只包含meets_decision_tree=true的异常")
            else:
                print(f"无法从 {analysis_type} 响应中找到JSON格式数据")
                # 作为备选，尝试从文本中提取关键信息
                if "体成分" in response or "体围" in response or "体态" in response:
                    abnormalities_for_query.append(response[:100])  # 使用前100字符
                    
        except Exception as e:
            print(f"解析 {analysis_type} 分析结果时出错: {e}")
            continue
    
    # 按优先级排序异常（数字越小优先级越高）
    for category in merged_result["identified_abnormalities"]:
        merged_result["identified_abnormalities"][category].sort(
            key=lambda x: x.get("priority", 999)
        )
    
    return merged_result, abnormalities_for_query

def get_agent_response_sync(assistant_config, message='', return_tool_output=False, knowledge_base=None):
    """输入message信息，输出为指定助手配置的回复（非流式）"""
    print(f"Assistant: {assistant_config['name']}")
    print(f"Query: {message}")
    
    all_tool_output = ""  # 存储所有工具输出
    
    try:
        # 创建OpenAI客户端
        client = OpenAI(
            api_key="sk-51d30a5436ca433b8ff81e624a23dcac",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 准备消息
        system_message = {'role': 'system', 'content': assistant_config['instructions']}
        user_message = {'role': 'user', 'content': message}
        messages = [system_message, user_message]
        
        # 非流式输出
        completion = client.chat.completions.create(
            model=assistant_config['model'],
            messages=messages,
            temperature=0.1,
            max_tokens=8000,
            stream=False
        )
        
        response = completion.choices[0].message.content
        print(f"AI回复: {response[:100]}...")
        
        if return_tool_output:
            return response, all_tool_output
        else:
            return response
                
    except Exception as e:
        print(f"AI调用失败: {e}")
        error_msg = "抱歉，处理过程中出现错误"
        if return_tool_output:
            return error_msg, all_tool_output
        else:
            return error_msg

def get_agent_response_stream(assistant_config, message='', knowledge_base=None):
    """输入message信息，输出为指定助手配置的回复（流式）"""
    
    try:
        # 创建OpenAI客户端
        client = OpenAI(
            api_key="sk-51d30a5436ca433b8ff81e624a23dcac",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        
        # 准备消息
        system_message = {'role': 'system', 'content': assistant_config['instructions']}
        user_message = {'role': 'user', 'content': message}
        messages = [system_message, user_message]
        
        # 流式输出
        completion = client.chat.completions.create(
            model=assistant_config['model'],
            messages=messages,
            temperature=0.1,
            max_tokens=8000,
            stream=True
        )
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content  # 流式输出每个chunk
                
    except Exception as e:
        print(f"AI调用失败: {e}")
        yield "抱歉，处理过程中出现错误"

def get_agent_response(assistant_config, message='', return_tool_output=False, knowledge_base=None, stream=False):
    """兼容性函数，根据stream参数选择同步或流式调用"""
    if stream:
        return get_agent_response_stream(assistant_config, message, knowledge_base)
    else:
        return get_agent_response_sync(assistant_config, message, return_tool_output, knowledge_base)

def get_multi_agent_response_internal(query, knowledge_base=None):
    """获得Multi Agent的回复的内部函数"""
    if len(query) == 0:
        return "请输入您的身体数据或问题", ""
    
    collected_knowledge_chunks = ""  # 收集知识库召回信息
    
    try:
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
        
        
        # 直接初始化结构化身体数据
        print("直接初始化结构化身体数据...")
        user_analysis = MedicalAnalysis.initialize_structured_data(user_body_data)

        # 并发执行异常分析
        
        # 执行并发分析
        analysis_results = analyze_abnormalities_concurrently(user_analysis, knowledge_base)
        
        # 合并分析结果
        print("合并并发分析结果...")
        merged_result, abnormalities_for_query = merge_abnormality_results(analysis_results)
        
        # 生成合并后的JSON响应
        merged_json_response = json.dumps(merged_result, ensure_ascii=False, indent=2)
        print(f"*并发异常分析*的合并结果为：\n```json\n{merged_json_response}\n```\n\n")
        
        
    
    except Exception as e:
        print(f"Multi-agent processing failed: {e}")


def test_body_analysis():
    """测试身体异常分析的三项多智能体流程 - 体成分+体围、有骨盆体态、无骨盆体态：优先输出符合决策树的异常"""
    # 示例用户数据
    user_body_data = {
    "scan_id": "25123456789125-812424e2-481b-11f0-a520-08bfb881e2bf",
    "product_model": "VD-PRO3",
    "mass_info": {
        "WT": {
            "l": 58.6,
            "m": 65.1,
            "h": 71.6,
            "v": 76.5,
            "status": 3
        },
        "FFM": {
            "l": 49.8,
            "m": 55.3,
            "h": 60.8,
            "v": 31.9,
            "status": 1
        },
        "TBW": {
            "l": 36.4,
            "m": 40.4,
            "h": 44.4,
            "v": 22.6,
            "status": 1
        },
        "BMI": {
            "l": 18.5,
            "m": 22,
            "h": 24.9,
            "v": 25.6,
            "status": 3
        },
        "PBF": {
            "l": 10,
            "m": 15,
            "h": 20,
            "v": 58.3,
            "status": 3
        },
        "BMR": {
            "l": 1445,
            "m": 1564.5,
            "h": 1683.9,
            "v": 1004.3,
            "status": 1
        },
        "WHR": {
            "l": 0.8,
            "m": 0.85,
            "h": 0.9,
            "v": 1,
            "status": 3
        },
        "SM": {
            "l": 27.5,
            "m": 30.6,
            "h": 33.7,
            "v": 18.5,
            "status": 1
        },
        "PROTEIN": {
            "l": 9.4,
            "m": 10.4,
            "h": 11.4,
            "v": 6.3,
            "status": 1
        },
        "ICW": {
            "l": 22.5,
            "m": 25,
            "h": 27.5,
            "v": 12.4,
            "status": 1
        },
        "ECW": {
            "l": 13.9,
            "m": 15.4,
            "h": 16.9,
            "v": 10.2,
            "status": 1
        },
        "VAG": {
            "l": 0.9,
            "h": 10,
            "m": 0,
            "v": 18,
            "status": 3
        }
    },
    "girth_info": {
        "neck": 0,
        "waist": 96.6,
        "hip": 101.9,
        "left_calf": 35.3,
        "right_calf": 35.1,
        "left_upper_arm": 32.1,
        "right_upper_arm": 31.4
    },
    "eval_info": {
        "id": 42984,
        "scan_id": "25123456789125-812424e2-481b-11f0-a520-08bfb881e2bf",
        "high_low_shoulder": -0.7,
        "head_slant": 1.4,
        "head_forward": -13.6,
        "left_leg_xo": 182.6,
        "right_leg_xo": 174.3,
        "pelvis_forward": 180.5,
        "left_knee_check": 186.4,
        "right_knee_check": 181.1,
        "round_shoulder_left": 18.9,
        "round_shoulder_right": 10,
        "body_slope": -0.7,
        "is_leg_diff_status": 1,
        "leg_length_diff": -6.4,
        "pelvic_forward_tilt_status": 1,
        "pelvic_forward_tilt": -5.3
    },
    "eval_conclusion": {
        "head_forward": {
            "val": -13.6,
            "conclusion": "正常",
            "conclusion_key": "body-product.bsEval.normal",
            "risk": "--",
            "state": 2
        },
        "head_slant": {
            "val": 1.4,
            "conclusion": "可能存在头侧歪(偏左)",
            "conclusion_key": "body-product.bsEval.headSlant.left",
            "risk": "头侧歪可能引发单侧颈部不适，单侧偏头痛，以及神经压迫的手臂发麻无力等症状",
            "state": 1
        },
        "round_shoulder_left": {
            "val": 18.9,
            "conclusion": "可能存在左圆肩",
            "conclusion_key": "body-product.bsEval.roundShoulder.left",
            "risk": "圆肩使胸廓容积变小、膈肌活动受限，影响呼吸、心血管及消化吸收功能，出现胸闷、头晕、气短等症状",
            "state": 1
        },
        "round_shoulder_right": {
            "val": 10,
            "conclusion": "正常",
            "conclusion_key": "body-product.bsEval.normal",
            "risk": "--",
            "state": 2
        },
        "high_low_shoudler": {
            "val": -0.7,
            "conclusion": "可能存在高低肩(右高)",
            "conclusion_key": "body-product.bsEval.highLowShoudler.right",
            "risk": "高低肩可引发颈肩部的慢性疼痛，常伴随脊柱侧弯、骨盆位移、长短腿等情况出现",
            "state": 3
        },
        "pelvis_forward": {
            "val": 180.5,
            "conclusion": "正常",
            "risk": "--",
            "state": 2
        },
        "pelvic_forward_tilt": {
            "val": -5.3,
            "conclusion": "可能存在骨盆后倾",
            "conclusion_key": "body-product.bsEval.pelvis.backward",
            "risk": "骨盆前/后倾可能会导致受力不均衡、身体比例失衡、便秘、脊柱侧弯、诱发腰椎疾病",
            "state": 3
        },
        "left_knee_check": {
            "val": 186.4,
            "conclusion": "正常",
            "conclusion_key": "body-product.bsEval.normal",
            "risk": "--",
            "state": 2
        },
        "right_knee_check": {
            "val": 181.1,
            "conclusion": "正常",
            "conclusion_key": "body-product.bsEval.normal",
            "risk": "--",
            "state": 2
        },
        "leg_xo": {
            "left_val": 182.6,
            "right_val": 174.3,
            "conclusion": "正常",
            "conclusion_key": "body-product.bsEval.normal",
            "risk": "--",
            "left_state": 2,
            "right_state": 2,
            "leg_type": 0
        },
        "leg_length_diff": {
            "val": -6.4,
            "conclusion": "存在长短腿风险(右长)",
            "conclusion_key": "body-product.bsEval.legLength.right",
            "risk": "长短腿会导致身体姿势不平衡，使得脊柱和骨盆处于不正常的位置。可能会导致脊柱侧弯、骨盆倾斜等问题，进而引发腰痛、颈椎病、坐骨神经痛等疼痛症状",
            "state": 3
        },
        "body_slope": {
            "val": -0.7,
            "conclusion": "可能存在身体倾斜(偏右)",
            "conclusion_key": "body-product.bsEval.bodySlope.right",
            "risk": "身体倾斜会导致踝关节、膝关节受力增大，引起腰背疼痛，严重的可能出现骨盆倾斜或长短腿",
            "state": 3
        },
        "weight_offset": {
            "val": {
                "id": 51332,
                "scan_id": "25123456789125-812424e2-481b-11f0-a520-08bfb881e2bf",
                "weighta": 26.4,
                "weightb": 24.8,
                "weightc": 13,
                "weightd": 12,
                "x": -0.1,
                "y": -6.9,
                "create_time": 1749794341
            }
        }
    },
    "shoulder_info": {
        "left_abduction":173.2, "right_abduction":181.1,
        "left_adduction":174.1, "right_adduction":171,
    },
    "spine_info": {
        "id": 28496,
        "scan_id": "25123456789125-812424e2-481b-11f0-a520-08bfb881e2bf",
        "back_s1c7": 0.2,
        "side_sisc7": 3,
        "create_time": 1749794556,
        "c7point": "{\"x\":-0.15433,\"y\":1428.86,\"z\":-75.4366}",
        "s1point": "{\"x\":-2.43302,\"y\":909.836,\"z\":-106.093}",
        "gravity_point": "{\"x\":-2.39786,\"y\":893.008,\"z\":-75.015}",
        "left_posterior_superior_iliac_spine": "{\"x\":54.4259,\"y\":936.251,\"z\":-147.556}",
        "right_posterior_superior_iliac_spine": "{\"x\":-52.7222,\"y\":934.353,\"z\":-149.542}",
        "back_view_conclusion": 1,
        "side_view_vonclusion": 0
    },
    "neck_info": {},
    "hip_info": {
        "id": 1703,
        "scan_id": "25123456789125-812424e2-481b-11f0-a520-08bfb881e2bf",
        "hip_type": 0,
        "hip_girth": 101.9,
        "create_time": 1749794523
    },
    "nutrition_info": {
        "weight": 76.5
    },
    "user_info": {
        "height": 173,
        "age": 28,
        "sex": 1
    },
    "region": "0",
    "language": "zh-CN",
    "unit": "metric"
}
    

    
    
    # 转换为字符串格式供分析使用
    query = f"请分析以下身体数据，进行三项并发分析（体成分+体围、有骨盆体态、无骨盆体态）：{json.dumps(user_body_data, ensure_ascii=False)}"
    
    # 调用多智能体分析
    try:
        import time
        time_start = time.time()
        get_multi_agent_response_internal(query, "异常2")
        time_end = time.time()
        #print("\n=== 知识库召回信息 ===")
        #print(knowledge_chunks)
        print(f"三项多智能体分析时间：{time_end - time_start}秒")
    except Exception as e:
        print(f"测试失败：{e}")

if __name__ == "__main__":
    # 运行测试
    test_body_analysis()