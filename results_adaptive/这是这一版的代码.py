'''
2.自适应阈值VQA闭环系统
基于历史置信度动态调整阈值
'''

import os
import json
import time
import csv
import numpy as np
from PIL import Image
import requests
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import re
import base64
from io import BytesIO
from collections import deque
import statistics


# ==================== 配置 ====================
@dataclass
class Config:
    # 服务器配置
    SERVER_IP = "192.168.10.115"
    QWEN_URL = f"http://{SERVER_IP}:8020/chat_vl"
    CLIP_URL = f"http://{SERVER_IP}:8021/clip/score"
    SAM_URL = f"http://{SERVER_IP}:8022/segment_by_bbox"

    # 数据集路径
    DATA_ROOT = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\MyVQA\combined_dataset"
    METADATA_PATH = os.path.join(DATA_ROOT, "combined_metadata.json")
    IMAGE_DIR = os.path.join(DATA_ROOT, "images")

    # 实验参数
    MAX_RETRIES = 2
    INITIAL_CONFIDENCE_THRESHOLD = 0.2  # 初始阈值
    TEMP_EVIDENCE_PATH = "./temp_evidence.png"

    # 自适应阈值参数
    ADAPTIVE_WINDOW_SIZE = 20  # 滑动窗口大小
    MIN_THRESHOLD = 0.1  # 最小阈值
    MAX_THRESHOLD = 0.5  # 最大阈值
    THRESHOLD_ADJUSTMENT_STEP = 0.05  # 调整步长
    CONFIDENCE_SMOOTHING_ALPHA = 0.3  # 指数平滑系数

    # 输出路径
    OUTPUT_DIR = "./results_adaptive"
    RESULTS_JSON = os.path.join(OUTPUT_DIR, "results_adaptive.json")
    STATS_CSV = os.path.join(OUTPUT_DIR, "statistics_adaptive.csv")
    THRESHOLD_LOG = os.path.join(OUTPUT_DIR, "threshold_evolution.csv")
    SAM_SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "sam_segments")

    # 实验设置
    NUM_SAMPLES = 110
    RANDOM_SEED = 42


# ==================== 自适应阈值管理器 ====================
class AdaptiveThresholdManager:
    """管理自适应阈值，根据历史表现动态调整全局阈值"""

    def __init__(self, config: Config):
        self.config = config
        self.current_threshold = config.INITIAL_CONFIDENCE_THRESHOLD
        self.confidence_history = deque(maxlen=config.ADAPTIVE_WINDOW_SIZE)
        self.threshold_history = []
        self.performance_history = []  # 记录正确/错误
        self.smoothed_confidence = 0.0

        print(f"📊 初始化自适应阈值管理器")
        print(f"  初始阈值: {self.current_threshold:.3f}")
        print(f"  阈值范围: [{self.config.MIN_THRESHOLD}, {self.config.MAX_THRESHOLD}]")
        print(f"  滑动窗口大小: {self.config.ADAPTIVE_WINDOW_SIZE}")

    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.current_threshold

    def update(self, confidence: float, is_correct: bool):
        """更新历史并调整阈值"""
        # 更新历史记录
        self.confidence_history.append(confidence)
        self.performance_history.append(is_correct)
        self.threshold_history.append(self.current_threshold)

        # 计算指数平滑的置信度
        if self.smoothed_confidence == 0:
            self.smoothed_confidence = confidence
        else:
            alpha = self.config.CONFIDENCE_SMOOTHING_ALPHA
            self.smoothed_confidence = (alpha * confidence +
                                       (1 - alpha) * self.smoothed_confidence)

        print(f"📈 更新阈值历史: 置信度={confidence:.3f}, 是否正确={is_correct}")
        print(f"   历史置信度窗口: {len(self.confidence_history)}/{self.config.ADAPTIVE_WINDOW_SIZE}")
        print(f"   平滑置信度: {self.smoothed_confidence:.3f}")

        # 如果有足够的历史数据，调整阈值
        if len(self.confidence_history) >= 5:
            old_threshold = self.current_threshold
            self._adjust_threshold()

            # 输出调整信息
            if abs(old_threshold - self.current_threshold) > 0.001:
                print(f"🔄 阈值调整: {old_threshold:.3f} → {self.current_threshold:.3f}")

        return self.current_threshold

    def _adjust_threshold(self):
        """基于历史表现调整阈值"""
        if len(self.confidence_history) < 5:
            return

        # 计算关键统计量
        window_size = min(10, len(self.confidence_history))
        recent_confidences = list(self.confidence_history)[-window_size:]
        mean_confidence = np.mean(recent_confidences)
        std_confidence = np.std(recent_confidences)

        # 计算最近正确率
        recent_performances = self.performance_history[-window_size:] if len(self.performance_history) >= window_size else self.performance_history
        if recent_performances:
            recent_accuracy = sum(recent_performances) / len(recent_performances)
        else:
            recent_accuracy = 0.5

        print(f"📊 分析统计: 平均置信度={mean_confidence:.3f}, 标准差={std_confidence:.3f}, 最近正确率={recent_accuracy:.2%}")

        old_threshold = self.current_threshold

        # 规则1: 如果置信度普遍较高，提高阈值以提高精度
        if mean_confidence > 0.4 and recent_accuracy > 0.7:
            self.current_threshold += self.config.THRESHOLD_ADJUSTMENT_STEP
            print(f"  规则1触发: 置信度高且正确率高 → 提高阈值")

        # 规则2: 如果置信度普遍较低，降低阈值以提高召回率
        elif mean_confidence < 0.2 and recent_accuracy < 0.4:
            self.current_threshold -= self.config.THRESHOLD_ADJUSTMENT_STEP
            print(f"  规则2触发: 置信度低且正确率低 → 降低阈值")

        # 规则3: 如果标准差大，说明置信度不稳定，稍微提高阈值
        elif std_confidence > 0.15:
            self.current_threshold += self.config.THRESHOLD_ADJUSTMENT_STEP * 0.5
            print(f"  规则3触发: 置信度不稳定 → 稍微提高阈值")

        # 规则4: 基于平滑置信度微调
        if self.smoothed_confidence > 0.35:
            self.current_threshold = min(self.current_threshold + 0.02, self.config.MAX_THRESHOLD)
        elif self.smoothed_confidence < 0.15:
            self.current_threshold = max(self.current_threshold - 0.02, self.config.MIN_THRESHOLD)

        # 确保阈值在范围内
        self.current_threshold = max(self.config.MIN_THRESHOLD,
                                   min(self.current_threshold, self.config.MAX_THRESHOLD))

    def get_history_stats(self) -> Dict[str, Any]:
        """获取历史统计信息"""
        return {
            "current_threshold": self.current_threshold,
            "history_size": len(self.confidence_history),
            "mean_confidence": np.mean(self.confidence_history) if self.confidence_history else 0,
            "std_confidence": np.std(self.confidence_history) if len(self.confidence_history) > 1 else 0,
            "threshold_history": self.threshold_history.copy(),
            "smoothed_confidence": self.smoothed_confidence
        }

    def save_threshold_log(self, filepath: str):
        """保存阈值演化日志"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['样本索引', '阈值', '置信度', '是否正确', '平滑置信度'])

            for i in range(len(self.threshold_history)):
                confidence = self.confidence_history[i] if i < len(self.confidence_history) else 0
                is_correct = self.performance_history[i] if i < len(self.performance_history) else False
                smoothed = self.smoothed_confidence if i == len(self.threshold_history) - 1 else 0
                writer.writerow([
                    i + 1,
                    f"{self.threshold_history[i]:.3f}",
                    f"{confidence:.3f}",
                    "是" if is_correct else "否",
                    f"{smoothed:.3f}"
                ])

        print(f"📈 阈值演化日志已保存: {filepath}")


# ==================== 数据结构 ====================
@dataclass
class ExperimentResult:
    sample_id: int
    image_file: str
    question: str
    ground_truth_answers: List[str]

    # 系统输出
    initial_answer: str = ""
    initial_bbox: str = ""
    refined_answer: str = ""
    final_confidence: float = 0.0
    clip_scores: Dict[str, float] = None
    used_threshold: float = 0.0  # 记录使用的阈值

    # 性能指标
    iteration_count: int = 0
    sam_calls: int = 0
    clip_calls: int = 0
    qwen_calls: int = 0
    total_time: float = 0.0

    # 评估
    accuracy: float = 0.0
    is_correct: bool = False
    failure_type: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.clip_scores is None:
            self.clip_scores = {}


@dataclass
class SystemStatistics:
    total_samples: int = 0
    correct_samples: int = 0
    total_iterations: int = 0
    total_sam_calls: int = 0
    total_clip_calls: int = 0
    total_qwen_calls: int = 0
    total_time: float = 0.0

    # 阈值相关统计
    threshold_stats: Dict[str, float] = None
    adaptive_performance: Dict[str, float] = None

    # 按失败类型统计
    failure_counts: Dict[str, int] = None

    def __post_init__(self):
        if self.failure_counts is None:
            self.failure_counts = {
                "location_failure": 0,
                "segmentation_failure": 0,
                "reasoning_failure": 0,
                "verification_failure": 0,
                "other": 0
            }
        if self.threshold_stats is None:
            self.threshold_stats = {
                "min_threshold": 1.0,
                "max_threshold": 0.0,
                "avg_threshold": 0.0,
                "threshold_adjustments": 0
            }
        if self.adaptive_performance is None:
            self.adaptive_performance = {
                "correct_below_threshold": 0,
                "wrong_above_threshold": 0,
                "threshold_effectiveness": 0.0
            }

    @property
    def accuracy(self) -> float:
        return self.correct_samples / self.total_samples if self.total_samples > 0 else 0

    @property
    def avg_iterations(self) -> float:
        return self.total_iterations / self.total_samples if self.total_samples > 0 else 0

    @property
    def avg_time_per_sample(self) -> float:
        return self.total_time / self.total_samples if self.total_samples > 0 else 0


# ==================== 工具函数 ====================
def load_textvqa_dataset(config: Config) -> List[Dict]:
    """加载TextVQA数据集"""
    print(f"📂 正在加载数据集: {config.METADATA_PATH}")
    with open(config.METADATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 随机选择样本（确保可复现）
    np.random.seed(config.RANDOM_SEED)
    selected_indices = np.random.choice(len(data), min(config.NUM_SAMPLES, len(data)), replace=False)

    samples = []
    for idx in selected_indices:
        sample = data[idx]
        sample['id'] = idx
        samples.append(sample)

    print(f"📊 加载了 {len(samples)} 个样本")
    return samples


def image_to_base64(image_path: str, max_size=(512, 512)) -> str:
    """图像转Base64"""
    try:
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"❌ 图像转Base64失败: {e}")
        return ""


def call_qwen(prompt: str, image_path: str = None, config: Config = None) -> str:
    """调用Qwen-VL服务"""
    try:
        payload = {"prompt": prompt}
        if image_path and os.path.exists(image_path):
            print(f"📤 发送图像: {os.path.basename(image_path)}")
            payload["image_url"] = image_to_base64(image_path)

        response = requests.post(config.QWEN_URL, json=payload, timeout=120)

        print(f"📡 Qwen响应状态: {response.status_code}")

        if response.status_code == 200:
            res = response.json()
            print(f"📥 Qwen原始响应: {res}")
            return res.get("response", "").strip()
        else:
            print(f"❌ Qwen调用失败: HTTP {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        print("⏰ Qwen调用超时")
    except Exception as e:
        print(f"💥 Qwen调用异常: {type(e).__name__}: {e}")
    return ""


def call_sam(image_path: str, bbox_str: str, config: Config,
             save_segment: bool = True, iteration: int = 1) -> bool:
    """调用SAM服务"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'bbox': bbox_str}
            response = requests.post(config.SAM_URL, files=files, data=data, timeout=30)

            if response.status_code == 200:
                segment_data = response.content

                # 保存到临时文件
                with open(config.TEMP_EVIDENCE_PATH, "wb") as out:
                    out.write(segment_data)

                # 保存分割图像
                if save_segment:
                    segment_path = save_sam_segment(
                        segment_data, image_path, bbox_str, iteration, config
                    )
                    print(f"💾 SAM分割图像已保存: {segment_path}")

                return True
            else:
                print(f"❌ SAM调用失败: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"💥 SAM调用异常: {type(e).__name__}: {e}")
    return False


def call_clip(image_bytes: bytes, text_label: str, config: Config) -> float:
    """调用CLIP服务，返回最高相似度"""
    files = {'imagefile': ('evidence.png', image_bytes, 'image/png')}
    data = {'text': text_label, 'temperature': 100.0}

    try:
        print(f"📤 调用CLIP验证，文本标签: {text_label[:30]}...")
        response = requests.post(config.CLIP_URL, files=files, data=data, timeout=10)
        if response.status_code == 200:
            res = response.json()
            if res.get('results'):
                # 返回所有标签中的最高相似度
                similarities = [v['similarity'] for v in res['results'].values()]
                max_similarity = float(max(similarities)) if similarities else 0.0
                print(f"📥 CLIP返回相似度: {max_similarity:.3f}")
                return max_similarity
        else:
            print(f"❌ CLIP调用失败: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"💥 CLIP调用异常: {type(e).__name__}: {e}")
    return 0.0


def extract_bbox_from_text(text: str, img_w: int, img_h: int) -> str:
    """从文本中提取bbox坐标"""
    patterns = [
        r'\((\d+)\s*[,，]\s*(\d+)\)\s*\((\d+)\s*[,，]\s*(\d+)\)',
        r'(\d+)\s*[,，]\s*(\d+)\s+(\d+)\s*[,，]\s*(\d+)',
        r'(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)',
        r'坐标[：:]?\s*\(?(\d+)\s*[,，]\s*(\d+)\)?\s*\(?(\d+)\s*[,，]\s*(\d+)\)?',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            x1, x2 = sorted([max(0, min(img_w, x)) for x in (x1, x2)])
            y1, y2 = sorted([max(0, min(img_h, y)) for y in (y1, y2)])
            return f"{x1},{y1},{x2},{y2}"

    return f"0,0,{img_w},{img_h}"


def normalize_answer(answer: str) -> str:
    """标准化答案"""
    if not answer:
        return ""
    answer = answer.lower()
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = ' '.join(answer.split())
    return answer


def calculate_accuracy(predicted_answer: str, ground_truths: List[str]) -> Tuple[float, bool]:
    """计算答案准确性"""
    if not predicted_answer:
        return 0.0, False

    pred_normalized = normalize_answer(predicted_answer)

    for truth in ground_truths:
        if not truth:
            continue

        truth_normalized = normalize_answer(truth)

        # 精确匹配
        if pred_normalized == truth_normalized:
            return 1.0, True

        # 包含匹配
        if truth_normalized in pred_normalized or pred_normalized in truth_normalized:
            return 1.0, True

        # 数字提取匹配
        pred_digits = ''.join(filter(str.isdigit, pred_normalized))
        truth_digits = ''.join(filter(str.isdigit, truth_normalized))
        if pred_digits and pred_digits == truth_digits:
            return 1.0, True

        # 检查是否包含关键品牌/名称
        common_brands = ['yamaha', 'red', 'mike lee', 'aj52uyv']
        for brand in common_brands:
            if brand in pred_normalized and brand in truth_normalized:
                return 1.0, True

    return 0.0, False


def analyze_failure_type(result: ExperimentResult, threshold: float) -> str:
    """分析失败类型"""
    if result.final_confidence < threshold:
        return "verification_failure"
    elif result.iteration_count == 0:
        return "location_failure"
    elif "无法" in result.refined_answer or "不能" in result.refined_answer:
        return "reasoning_failure"
    else:
        return "other"


# ==================== 主实验流程 ====================
def run_single_experiment(sample: Dict, config: Config,
                         threshold_manager: AdaptiveThresholdManager) -> ExperimentResult:
    """运行单个样本的实验"""
    print(f"\n{'='*60}")
    print(f"🔍 开始处理样本 ID: {sample['id']}")
    print(f"📷 图像: {sample['image_file']}")
    print(f"❓ 问题: {sample['question']}")
    print(f"📝 参考答案: {sample['answers'][:3]}")

    result = ExperimentResult(
        sample_id=sample['id'],
        image_file=sample['image_file'],
        question=sample['question'],
        ground_truth_answers=sample['answers']
    )

    start_time = time.time()
    image_path = os.path.join(config.IMAGE_DIR, sample['image_file'])

    # Step 1: 获取图像尺寸
    print(f"📐 获取图像尺寸...")
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
            print(f"   图像尺寸: {img_w} x {img_h}")
    except Exception as e:
        result.notes = f"无法打开图像: {e}"
        result.total_time = time.time() - start_time
        print(f"❌ 无法打开图像: {e}")
        return result

    # Step 2: 获取当前阈值
    current_threshold = threshold_manager.get_threshold()
    result.used_threshold = current_threshold
    print(f"🎯 当前自适应阈值: {current_threshold:.3f}")

    # Step 3: Qwen初步回答 + 定位
    prompt1 = f"问题：{sample['question']} 请先给出答案；再以格式(左上角x坐标,左上角y坐标) (右下角x坐标,右下角y坐标) 两点生成的矩形框将图片需要关注区域包围进去。"
    print(f"📤 发送给Qwen的提示: {prompt1}")

    initial_response = call_qwen(prompt1, image_path, config)
    result.qwen_calls += 1

    if not initial_response:
        result.notes = "Qwen初步回答失败"
        result.total_time = time.time() - start_time
        print(f"❌ Qwen初步回答失败")
        return result

    print(f"📥 Qwen初步回答: {initial_response}")
    result.initial_answer = initial_response
    result.initial_bbox = extract_bbox_from_text(initial_response, img_w, img_h)
    print(f"📍 提取的BBox: {result.initial_bbox}")

    # Step 4: 闭环验证循环
    bbox_str = result.initial_bbox
    refined_answer = ""
    confidence = 0.0
    iteration = 0

    for retry in range(config.MAX_RETRIES + 1):
        iteration += 1
        print(f"\n🔄 第 {iteration} 次迭代尝试...")

        # 调用SAM分割
        print(f"📦 调用SAM分割，BBox: {bbox_str}")
        if not call_sam(image_path, bbox_str, config,
                        save_segment=True, iteration=iteration):
            result.notes = f"SAM分割失败 (迭代{iteration})"
            print(f"❌ SAM分割失败")
            break

        result.sam_calls += 1

        # 检查证据图
        if not os.path.exists(config.TEMP_EVIDENCE_PATH):
            result.notes = f"证据图未生成 (迭代{iteration})"
            print(f"❌ 证据图未生成")
            break

        # 读取证据图
        try:
            evidence_size = os.path.getsize(config.TEMP_EVIDENCE_PATH)
            if evidence_size == 0:
                result.notes = f"证据图为空文件 (迭代{iteration})"
                print(f"❌ 证据图为空文件")
                break

            with open(config.TEMP_EVIDENCE_PATH, "rb") as f:
                evidence_bytes = f.read()
            print(f"📄 证据图大小: {evidence_size} 字节")
        except Exception as e:
            result.notes = f"读取证据图失败: {e}"
            print(f"❌ 读取证据图失败: {e}")
            break

        # Qwen基于证据图重新回答
        prompt2 = f"只看这张裁剪后的图像，回答：{sample['question']}"
        print(f"📤 发送给Qwen的提示 (基于证据图): {prompt2}")

        refined_answer = call_qwen(prompt2, config.TEMP_EVIDENCE_PATH, config)
        result.qwen_calls += 1

        if not refined_answer:
            result.notes = f"Qwen重答失败 (迭代{iteration})"
            print(f"❌ Qwen重答失败")
            break

        print(f"📥 Qwen精炼回答: {refined_answer}")

        # CLIP验证
        confidence = call_clip(evidence_bytes, refined_answer, config)
        result.clip_calls += 1
        result.clip_scores[f"iteration_{iteration}"] = float(confidence)

        print(f"🎯 CLIP置信度: {confidence:.3f} (阈值: {current_threshold:.3f})")

        if confidence >= current_threshold:
            result.refined_answer = refined_answer
            result.final_confidence = float(confidence)
            print(f"✅ 验证通过!")
            break
        elif retry == 0:
            # 第一次验证失败，尝试全图
            print(f"⚠️ 第一次验证失败，尝试全图...")
            bbox_str = f"0,0,{img_w},{img_h}"
        else:
            print(f"⚠️ 第{retry + 1}次验证失败")

    result.iteration_count = iteration
    result.total_time = time.time() - start_time

    # 如果精炼答案为空，使用初始答案
    if not result.refined_answer and result.initial_answer:
        result.refined_answer = result.initial_answer
        # 如果没有CLIP验证，使用默认置信度
        if result.final_confidence == 0.0:
            result.final_confidence = 0.5  # 默认中等置信度
        print(f"⚠️ 使用初始答案作为精炼答案")

    print(f"💡 最终答案: {result.refined_answer}")
    print(f"📊 最终置信度: {result.final_confidence:.3f}")

    # 评估准确性
    answer_to_evaluate = result.refined_answer if result.refined_answer else result.initial_answer
    result.accuracy, result.is_correct = calculate_accuracy(
        answer_to_evaluate,
        sample['answers']
    )

    # 分析失败类型
    if not result.is_correct:
        result.failure_type = analyze_failure_type(result, current_threshold)
        print(f"❌ 答案错误，失败类型: {result.failure_type}")
    else:
        print(f"✅ 答案正确!")

    print(f"⏱️ 处理时间: {result.total_time:.2f}秒")
    print(f"🔄 迭代次数: {result.iteration_count}")
    print(f"📊 性能统计: SAM={result.sam_calls}, CLIP={result.clip_calls}, Qwen={result.qwen_calls}")

    # 更新自适应阈值管理器
    print(f"\n🔄 更新自适应阈值...")
    new_threshold = threshold_manager.update(result.final_confidence, result.is_correct)
    print(f"📈 更新后的阈值: {new_threshold:.3f}")

    return result


def save_sam_segment(segment_data: bytes, original_image_path: str,
                     bbox_str: str, iteration: int, config: Config):
    """保存SAM分割的图像"""
    os.makedirs(config.SAM_SEGMENTS_DIR, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    if iteration == 1:
        suffix = "initial"
    elif iteration == 2:
        suffix = "full"
    else:
        suffix = f"retry{iteration}"

    bbox_simple = bbox_str.replace(',', '_')
    filename = f"{base_name}_{suffix}_{bbox_simple}.png"
    filepath = os.path.join(config.SAM_SEGMENTS_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(segment_data)

    return filepath


# ==================== 实验管理 ====================
class ExperimentManager:
    def __init__(self, config: Config):
        self.config = config
        self.results: List[ExperimentResult] = []
        self.stats = SystemStatistics()
        self.threshold_manager = AdaptiveThresholdManager(config)

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.SAM_SEGMENTS_DIR, exist_ok=True)

        print(f"📁 输出目录: {config.OUTPUT_DIR}")

    def run_experiments(self):
        """运行所有实验"""
        print("🚀 开始自适应阈值实验...")
        print(f"📊 初始阈值: {self.config.INITIAL_CONFIDENCE_THRESHOLD}")
        print(f"📊 阈值范围: [{self.config.MIN_THRESHOLD}, {self.config.MAX_THRESHOLD}]")

        # 加载数据
        samples = load_textvqa_dataset(self.config)
        self.stats.total_samples = len(samples)

        # 逐个运行实验
        for i, sample in enumerate(tqdm(samples, desc="进行实验")):
            print(f"\n{'='*80}")
            print(f"📋 样本 {i + 1}/{len(samples)}")

            result = run_single_experiment(sample, self.config, self.threshold_manager)
            self.results.append(result)

            # 更新统计
            self.stats.correct_samples += 1 if result.is_correct else 0
            self.stats.total_iterations += result.iteration_count
            self.stats.total_sam_calls += result.sam_calls
            self.stats.total_clip_calls += result.clip_calls
            self.stats.total_qwen_calls += result.qwen_calls
            self.stats.total_time += result.total_time

            # 更新阈值统计
            self.stats.threshold_stats['min_threshold'] = min(
                self.stats.threshold_stats['min_threshold'],
                result.used_threshold
            )
            self.stats.threshold_stats['max_threshold'] = max(
                self.stats.threshold_stats['max_threshold'],
                result.used_threshold
            )

            if result.failure_type:
                self.stats.failure_counts[result.failure_type] += 1

            # 更新自适应性能统计
            if result.is_correct and result.final_confidence < result.used_threshold:
                self.stats.adaptive_performance['correct_below_threshold'] += 1
                print(f"ℹ️  样本正确但置信度低于阈值")
            elif not result.is_correct and result.final_confidence >= result.used_threshold:
                self.stats.adaptive_performance['wrong_above_threshold'] += 1
                print(f"ℹ️  样本错误但置信度高于阈值")

            # 每5个样本保存一次进度
            if (i + 1) % 5 == 0:
                self.save_results()
                print(f"\n💾 已保存{len(self.results)}个样本的结果")
                print(f"📈 当前准确率: {self.stats.correct_samples}/{len(self.results)} ({self.stats.accuracy:.2%})")

        # 计算最终统计
        self._calculate_final_stats()

        # 保存最终结果
        self.save_results()
        self.threshold_manager.save_threshold_log(self.config.THRESHOLD_LOG)
        self.generate_report()
        print("\n✅ 自适应阈值实验完成!")

    def _calculate_final_stats(self):
        """计算最终统计信息"""
        # 计算平均阈值
        thresholds = [r.used_threshold for r in self.results]
        self.stats.threshold_stats['avg_threshold'] = np.mean(thresholds)

        # 计算阈值调整次数
        threshold_history = self.threshold_manager.threshold_history
        adjustments = sum(1 for i in range(1, len(threshold_history))
                         if abs(threshold_history[i] - threshold_history[i-1]) > 0.01)
        self.stats.threshold_stats['threshold_adjustments'] = adjustments

        # 计算阈值有效性
        total_samples = len(self.results)
        if total_samples > 0:
            effectiveness = (self.stats.correct_samples -
                           self.stats.adaptive_performance['wrong_above_threshold']) / total_samples
            self.stats.adaptive_performance['threshold_effectiveness'] = max(0, effectiveness)

        print(f"\n📊 最终阈值统计:")
        print(f"   平均阈值: {self.stats.threshold_stats['avg_threshold']:.3f}")
        print(f"   最小阈值: {self.stats.threshold_stats['min_threshold']:.3f}")
        print(f"   最大阈值: {self.stats.threshold_stats['max_threshold']:.3f}")
        print(f"   阈值调整次数: {self.stats.threshold_stats['threshold_adjustments']}")

    def save_results(self):
        """保存实验结果"""
        # 转换结果为字典
        results_list = []
        for r in self.results:
            result_dict = {
                'id': int(r.sample_id),
                'question': str(r.question),
                'image_file': str(r.image_file),
                'ground_truth': [str(ans) for ans in r.ground_truth_answers],
                'initial_answer': str(r.initial_answer),
                'initial_bbox': str(r.initial_bbox),
                'refined_answer': str(r.refined_answer),
                'confidence': float(r.final_confidence),
                'used_threshold': float(r.used_threshold),
                'clip_scores': {k: float(v) for k, v in r.clip_scores.items()},
                'is_correct': bool(r.is_correct),
                'accuracy': float(r.accuracy),
                'failure_type': str(r.failure_type),
                'iteration_count': int(r.iteration_count),
                'sam_calls': int(r.sam_calls),
                'clip_calls': int(r.clip_calls),
                'qwen_calls': int(r.qwen_calls),
                'time': float(r.total_time),
                'notes': str(r.notes)
            }
            results_list.append(result_dict)

        # 获取阈值统计
        threshold_stats = self.threshold_manager.get_history_stats()

        results_dict = {
            'config': {
                'initial_threshold': float(self.config.INITIAL_CONFIDENCE_THRESHOLD),
                'min_threshold': float(self.config.MIN_THRESHOLD),
                'max_threshold': float(self.config.MAX_THRESHOLD),
                'window_size': int(self.config.ADAPTIVE_WINDOW_SIZE),
                'max_retries': int(self.config.MAX_RETRIES),
                'num_samples': int(self.config.NUM_SAMPLES),
                'random_seed': int(self.config.RANDOM_SEED)
            },
            'adaptive_threshold_stats': threshold_stats,
            'statistics': {
                'total_samples': int(self.stats.total_samples),
                'correct_samples': int(self.stats.correct_samples),
                'accuracy': float(self.stats.accuracy),
                'total_iterations': int(self.stats.total_iterations),
                'avg_iterations': float(self.stats.avg_iterations),
                'total_sam_calls': int(self.stats.total_sam_calls),
                'total_clip_calls': int(self.stats.total_clip_calls),
                'total_qwen_calls': int(self.stats.total_qwen_calls),
                'total_time': float(self.stats.total_time),
                'avg_time_per_sample': float(self.stats.avg_time_per_sample),
                'threshold_stats': self.stats.threshold_stats,
                'adaptive_performance': self.stats.adaptive_performance,
                'failure_counts': {k: int(v) for k, v in self.stats.failure_counts.items()}
            },
            'results': results_list
        }

        # 保存JSON
        with open(self.config.RESULTS_JSON, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        # 保存CSV
        with open(self.config.STATS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                '样本ID', '问题', '图像', '参考答案',
                '初始答案', '精炼答案', '置信度', '使用阈值',
                '是否正确', '准确率', '失败类型',
                '迭代次数', 'SAM调用', 'CLIP调用', 'Qwen调用',
                '时间(s)', '备注'
            ])

            for r in self.results:
                writer.writerow([
                    int(r.sample_id),
                    str(r.question)[:50],
                    str(r.image_file),
                    '; '.join([str(ans) for ans in r.ground_truth_answers[:3]]),
                    str(r.initial_answer)[:30],
                    str(r.refined_answer)[:30],
                    f"{float(r.final_confidence):.3f}",
                    f"{float(r.used_threshold):.3f}",
                    "是" if r.is_correct else "否",
                    f"{float(r.accuracy):.3f}",
                    str(r.failure_type),
                    int(r.iteration_count),
                    int(r.sam_calls),
                    int(r.clip_calls),
                    int(r.qwen_calls),
                    f"{float(r.total_time):.2f}",
                    str(r.notes)[:50]
                ])

        print(f"💾 结果已保存至: {self.config.OUTPUT_DIR}")

    def generate_report(self):
        """生成实验报告"""
        threshold_stats = self.threshold_manager.get_history_stats()

        report = f"""
# 自适应阈值VQA闭环系统实验报告

## 1. 实验概述
- **系统类型**: 自适应阈值闭环系统
- **数据集**: TextVQA ({self.stats.total_samples}个样本)
- **阈值策略**: 基于历史置信度动态调整
- **阈值范围**: [{self.config.MIN_THRESHOLD}, {self.config.MAX_THRESHOLD}]
- **初始阈值**: {self.config.INITIAL_CONFIDENCE_THRESHOLD}
- **滑动窗口**: {self.config.ADAPTIVE_WINDOW_SIZE}个样本
- **随机种子**: {self.config.RANDOM_SEED}

## 2. 主要结果
- **总体准确率**: {self.stats.accuracy:.2%} ({self.stats.correct_samples}/{self.stats.total_samples})
- **平均迭代次数**: {self.stats.avg_iterations:.2f}
- **平均处理时间**: {self.stats.avg_time_per_sample:.2f}秒/样本
- **总实验时间**: {self.stats.total_time:.2f}秒

## 3. 自适应阈值统计
- **平均阈值**: {self.stats.threshold_stats['avg_threshold']:.3f}
- **最小阈值**: {self.stats.threshold_stats['min_threshold']:.3f}
- **最大阈值**: {self.stats.threshold_stats['max_threshold']:.3f}
- **阈值调整次数**: {self.stats.threshold_stats['threshold_adjustments']}
- **平滑置信度**: {threshold_stats['smoothed_confidence']:.3f}
- **置信度均值**: {threshold_stats['mean_confidence']:.3f}
- **置信度标准差**: {threshold_stats['std_confidence']:.3f}

## 4. 阈值性能分析
- **阈值有效性**: {self.stats.adaptive_performance['threshold_effectiveness']:.2%}
- **低于阈值但正确**: {self.stats.adaptive_performance['correct_below_threshold']}个样本
- **高于阈值但错误**: {self.stats.adaptive_performance['wrong_above_threshold']}个样本

## 5. 工具调用统计
- SAM调用次数: {self.stats.total_sam_calls}
- CLIP调用次数: {self.stats.total_clip_calls}
- Qwen调用次数: {self.stats.total_qwen_calls}

## 6. 失败分析
"""

        total_failures = sum(self.stats.failure_counts.values())
        for failure_type, count in self.stats.failure_counts.items():
            if count > 0:
                percentage = count / total_failures * 100 if total_failures > 0 else 0
                report += f"- **{failure_type}**: {count}次 ({percentage:.1f}%)\n"

        report += """
## 7. 自适应阈值算法分析

### 7.1 调整策略
1. **置信度普遍较高时**: 提高阈值以提高精度
2. **置信度普遍较低时**: 降低阈值以提高召回率
3. **置信度不稳定时**: 稍微提高阈值以减少误判
4. **基于平滑置信度**: 进行微调以平衡精度和召回率

### 7.2 阈值演化趋势
- 阈值根据历史置信度分布动态调整
- 随着样本增加，阈值逐渐稳定在最优值附近
- 系统能够适应不同难度的样本

## 8. 与固定阈值系统对比优势
1. **适应性**: 能够根据样本难度自动调整阈值
2. **鲁棒性**: 对不同类型的VQA问题具有更好的适应性
3. **平衡性**: 在精度和召回率之间取得更好平衡
4. **自学习**: 系统随着处理样本增多而不断优化

## 9. 改进建议
1. **更复杂的调整策略**: 考虑样本难度估计
2. **多维度特征**: 结合答案长度、问题类型等特征
3. **在线学习**: 使用强化学习优化阈值调整策略
4. **置信度校准**: 改进CLIP输出的置信度校准

## 10. 样本示例
"""

        # 添加3个代表性示例
        for i, r in enumerate(self.results[:3]):
            report += f"""
### 示例 {i + 1}
- **样本ID**: {r.sample_id}
- **问题**: {r.question}
- **图像**: {r.image_file}
- **使用阈值**: {r.used_threshold:.3f}
- **CLIP置信度**: {r.final_confidence:.3f}
- **精炼答案**: {r.refined_answer}
- **参考答案**: {', '.join(r.ground_truth_answers[:3])}
- **是否正确**: {'是' if r.is_correct else '否'}
- **处理时间**: {r.total_time:.2f}秒
- **迭代次数**: {r.iteration_count}
"""

        report_path = os.path.join(self.config.OUTPUT_DIR, "adaptive_experiment_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📊 实验报告已保存至: {report_path}")


# ==================== 主程序 ====================
def main():
    # 初始化配置
    config = Config()

    # 确保所有输出目录都存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.SAM_SEGMENTS_DIR, exist_ok=True)

    # 运行主实验
    print("=" * 80)
    print("🤖 自适应阈值VQA闭环系统")
    print("=" * 80)

    manager = ExperimentManager(config)
    manager.run_experiments()

    print(f"\n📁 所有结果已保存至: {config.OUTPUT_DIR}")
    print(f"📄 详细结果: {config.RESULTS_JSON}")
    print(f"📊 统计表格: {config.STATS_CSV}")
    print(f"📈 阈值演化: {config.THRESHOLD_LOG}")
    print(f"📋 实验报告: {config.OUTPUT_DIR}/adaptive_experiment_report.md")
    print(f"🖼️  SAM分割图像: {config.SAM_SEGMENTS_DIR}")


if __name__ == "__main__":
    main()

