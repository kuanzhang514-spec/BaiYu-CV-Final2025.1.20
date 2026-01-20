'''
DocVQA/
├── images/
│   ├── 1.png
│   ├── 2.png
│   └── ...
└── questions/
    ├── 1.txt
    ├── 2.txt
    └── ...
'''

'''
DocVQA实验文档

'''

# main_experiment_docvqa.py - 针对DocVQA数据集的版本

import os
import json
import time
import csv
import numpy as np
from PIL import Image
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
import re
import base64
import ast
from io import BytesIO
import glob


# ==================== 配置 ====================
@dataclass
class Config:
    # 服务器配置
    SERVER_IP = "这是我的服务器IP地址，我隐藏了"
    QWEN_URL = f"http://{SERVER_IP}:8020/chat_vl"
    CLIP_URL = f"http://{SERVER_IP}:8021/clip/score"
    SAM_URL = f"http://{SERVER_IP}:8022/segment_by_bbox"

    # DocVQA数据集路径
    DATA_ROOT = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\DocVQA"  # 修改这里
    IMAGE_DIR = os.path.join(DATA_ROOT, "images")
    QUESTION_DIR = os.path.join(DATA_ROOT, "questions")
    METADATA_PATH = os.path.join(DATA_ROOT, "metadata.json")  # 会生成这个文件

    # 实验参数
    MAX_RETRIES = 2
    CONFIDENCE_THRESHOLD = 0.2
    TEMP_EVIDENCE_PATH = "./temp_evidence.png"

    # 输出路径
    OUTPUT_DIR = "./docvqa_experiment_results"
    RESULTS_JSON = os.path.join(OUTPUT_DIR, "results.json")
    STATS_CSV = os.path.join(OUTPUT_DIR, "statistics.csv")
    SAM_SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "sam_segments")

    # 实验设置,样本数，随机种子
    NUM_SAMPLES = 200
    RANDOM_SEED = 42


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
def parse_question_file(file_path: str) -> Tuple[str, List[str]]:
    """解析问题文件，提取问题和答案"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        # 提取问题
        question_match = re.search(r'QUESTION:\s*(.*?)(?:\n|$)', content)
        question = question_match.group(1).strip() if question_match else ""

        # 提取答案
        answers_match = re.search(r'ANSWERS:\s*(.*?)(?:\n|$)', content)
        answers_str = answers_match.group(1).strip() if answers_match else "[]"

        # 将字符串形式的列表转换为实际的列表
        try:
            # 使用ast.literal_eval安全地评估字符串
            answers = ast.literal_eval(answers_str)
            # 确保所有答案是字符串
            answers = [str(ans) for ans in answers]
        except:
            # 如果解析失败，尝试手动提取
            answers = []
            # 匹配单引号或双引号内的内容
            answer_matches = re.findall(r"['\"](.*?)['\"]", answers_str)
            answers = answer_matches if answer_matches else []

        return question, answers
    except Exception as e:
        print(f"解析问题文件失败 {file_path}: {e}")
        return "", []


def load_docvqa_dataset(config: Config) -> List[Dict]:
    """加载DocVQA数据集"""
    print("📊 开始加载DocVQA数据集...")

    # 检查是否已有缓存的元数据
    if os.path.exists(config.METADATA_PATH):
        print(f"📁 从缓存加载元数据: {config.METADATA_PATH}")
        with open(config.METADATA_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 从缓存加载了 {len(data)} 个样本")
        return data

    # 扫描图像文件
    image_files = []
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(config.IMAGE_DIR, f"*{ext}")))

    print(f"📷 找到 {len(image_files)} 个图像文件")

    samples = []

    # 处理每个图像
    for image_path in tqdm(image_files, desc="处理图像"):
        image_file = os.path.basename(image_path)
        image_id = os.path.splitext(image_file)[0]  # 去掉扩展名

        # 构建对应的问题文件路径
        question_file = os.path.join(config.QUESTION_DIR, f"{image_id}.txt")

        # 检查问题文件是否存在
        if not os.path.exists(question_file):
            # 尝试其他可能的扩展名
            found = False
            for q_ext in ['.txt', '.TXT', '.text']:
                alt_question_file = os.path.join(config.QUESTION_DIR, f"{image_id}{q_ext}")
                if os.path.exists(alt_question_file):
                    question_file = alt_question_file
                    found = True
                    break

            if not found:
                print(f"⚠️ 未找到问题文件: {image_id}")
                continue

        # 解析问题文件
        question, answers = parse_question_file(question_file)

        if not question:
            print(f"⚠️ 问题为空: {image_id}")
            continue

        # 创建样本
        sample = {
            'id': len(samples) + 1,
            'image_file': image_file,
            'question': question,
            'answers': answers,
            'image_id': image_id
        }

        samples.append(sample)

    print(f"📊 成功加载 {len(samples)} 个样本")

    # 保存元数据以便下次快速加载
    os.makedirs(os.path.dirname(config.METADATA_PATH), exist_ok=True)
    with open(config.METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    print(f"💾 元数据已保存到: {config.METADATA_PATH}")
    return samples


def load_docvqa_dataset_for_experiment(config: Config) -> List[Dict]:
    """加载DocVQA数据集并随机选择样本"""
    # 加载完整数据集
    all_samples = load_docvqa_dataset(config)

    if not all_samples:
        print("❌ 未找到任何样本")
        return []

    # 随机选择样本（确保可复现）
    np.random.seed(config.RANDOM_SEED)
    num_samples = min(config.NUM_SAMPLES, len(all_samples))
    selected_indices = np.random.choice(len(all_samples), num_samples, replace=False)

    selected_samples = []
    for idx in selected_indices:
        sample = all_samples[idx].copy()
        selected_samples.append(sample)

    print(f"🎯 随机选择了 {len(selected_samples)} 个样本进行实验")

    # 显示一些统计信息
    if selected_samples:
        print("\n📈 样本统计:")
        print(f"  平均答案数量: {np.mean([len(s['answers']) for s in selected_samples]):.2f}")
        print(f"  问题平均长度: {np.mean([len(s['question']) for s in selected_samples]):.2f} 字符")

        # 显示前3个样本
        print("\n📋 前3个样本预览:")
        for i in range(min(3, len(selected_samples))):
            print(f"  样本 {i + 1}: {selected_samples[i]['question'][:50]}...")
            print(f"       答案: {selected_samples[i]['answers'][:2] if selected_samples[i]['answers'] else '无答案'}")

    return selected_samples


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
    """调用SAM服务，添加图像验证"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'bbox': bbox_str}
            response = requests.post(config.SAM_URL, files=files, data=data, timeout=30)

            if response.status_code == 200:
                # 检查响应内容类型
                content_type = response.headers.get('content-type', '')

                if 'image/png' in content_type:
                    segment_data = response.content

                    # === 关键修复：验证图像数据是否有效 ===
                    if len(segment_data) == 0:
                        print(f"❌ SAM返回空图像数据")
                        return False

                    # 尝试解析图像验证其有效性
                    try:
                        from PIL import Image
                        import io
                        img = Image.open(io.BytesIO(segment_data))
                        img.verify()  # 验证图像完整性
                        width, height = img.size

                        if width == 0 or height == 0:
                            print(f"❌ SAM返回无效图像尺寸: {width}x{height}")
                            return False

                    except Exception as e:
                        print(f"❌ SAM返回无效图像数据: {e}")
                        return False
                    # === 结束验证 ===

                    # 保存到临时文件用于后续处理
                    with open(config.TEMP_EVIDENCE_PATH, "wb") as out:
                        out.write(segment_data)

                    # 如果需要保存分割图像
                    if save_segment:
                        segment_path = save_sam_segment(
                            segment_data, image_path, bbox_str, iteration, config
                        )
                        print(f"💾 SAM分割图像已保存: {segment_path}")

                    return True
                else:
                    # 可能是JSON错误响应
                    try:
                        error_info = response.json()
                        print(f"❌ SAM服务返回错误: {error_info.get('error', '未知错误')}")
                    except:
                        print(f"❌ SAM返回非图像响应: {content_type}")
                    return False
            else:
                print(f"❌ SAM调用失败: HTTP {response.status_code} - {response.text}")
                return False
    except Exception as e:
        print(f"💥 SAM调用异常: {type(e).__name__}: {e}")
        return False


def call_clip(image_bytes: bytes, text_label: str, config: Config) -> float:
    """调用CLIP服务，返回最高相似度"""
    files = {'imagefile': ('evidence.png', image_bytes, 'image/png')}
    data = {'text': text_label, 'temperature': 100.0}

    try:
        response = requests.post(config.CLIP_URL, files=files, data=data, timeout=10)
        if response.status_code == 200:
            res = response.json()
            if res.get('results'):
                # 返回所有标签中的最高相似度
                similarities = [v['similarity'] for v in res['results'].values()]
                return float(max(similarities)) if similarities else 0.0
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
            # 确保坐标在图像范围内
            x1, x2 = sorted([max(0, min(img_w, x)) for x in (x1, x2)])
            y1, y2 = sorted([max(0, min(img_h, y)) for y in (y1, y2)])
            return f"{x1},{y1},{x2},{y2}"

    # 未找到坐标，返回全图
    return f"0,0,{img_w},{img_h}"


def normalize_answer(answer: str) -> str:
    """标准化答案：小写、移除标点、空格"""
    if not answer:
        return ""
    # 转换为小写
    answer = answer.lower()
    # 移除标点符号（保留数字和字母）
    answer = re.sub(r'[^\w\s\d]', '', answer)
    # 移除多余空格
    answer = ' '.join(answer.split())
    return answer


def calculate_accuracy(predicted_answer: str, ground_truths: List[str]) -> Tuple[float, bool]:
    """计算答案准确性（针对DocVQA优化）"""
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

        # 包含匹配（针对较长答案）
        if truth_normalized in pred_normalized or pred_normalized in truth_normalized:
            return 1.0, True

        # 数字提取匹配（针对DocVQA中的数值问题）
        pred_digits = re.findall(r'\d+\.?\d*', pred_normalized)
        truth_digits = re.findall(r'\d+\.?\d*', truth_normalized)

        if pred_digits and truth_digits:
            # 检查是否有相同的数字
            for p_digit in pred_digits:
                for t_digit in truth_digits:
                    # 移除前导零和小数点后的零
                    p_clean = p_digit.lstrip('0').rstrip('.') if '.' in p_digit else p_digit.lstrip('0')
                    t_clean = t_digit.lstrip('0').rstrip('.') if '.' in t_digit else t_digit.lstrip('0')

                    if p_clean and t_clean and p_clean == t_clean:
                        return 1.0, True

        # 检查是否包含关键信息（针对DocVQA文档中的特定信息）
        # 可以在这里添加针对文档理解的特定匹配规则

        # 检查是否为"yes/no"类型问题
        if pred_normalized in ['yes', 'no', 'true', 'false'] and truth_normalized in ['yes', 'no', 'true', 'false']:
            if pred_normalized == truth_normalized:
                return 1.0, True

    return 0.0, False


def analyze_failure_type(result: ExperimentResult, config: Config) -> str:
    """分析失败类型"""
    if result.final_confidence < config.CONFIDENCE_THRESHOLD:
        return "verification_failure"
    elif result.iteration_count == 0:
        return "location_failure"
    elif "无法" in result.refined_answer or "不能" in result.refined_answer or "no" in result.refined_answer.lower():
        return "reasoning_failure"
    else:
        return "other"


def get_fallback_bbox(current_bbox: str, img_w: int, img_h: int, iteration: int) -> str:
    """获取智能回退的bbox"""
    try:
        # 解析当前bbox
        x1, y1, x2, y2 = map(int, current_bbox.split(','))

        if iteration == 1:
            # 第一次回退：扩大区域（1.5倍）
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            # 确保有最小尺寸
            min_size = 50
            width = max(width, min_size)
            height = max(height, min_size)

            # 扩大1.5倍
            new_width = int(width * 1.5)
            new_height = int(height * 1.5)

            x1 = max(0, center_x - new_width // 2)
            y1 = max(0, center_y - new_height // 2)
            x2 = min(img_w, center_x + new_width // 2)
            y2 = min(img_h, center_y + new_height // 2)

            return f"{x1},{y1},{x2},{y2}"

        else:
            # 后续回退：使用全图
            return f"0,0,{img_w},{img_h}"

    except:
        # 如果解析失败，返回全图
        return f"0,0,{img_w},{img_h}"


# ==================== 主实验流程 ====================
def run_single_experiment(sample: Dict, config: Config) -> ExperimentResult:
    """运行单个样本的实验"""
    result = ExperimentResult(
        sample_id=sample['id'],
        image_file=sample['image_file'],
        question=sample['question'],
        ground_truth_answers=sample['answers']
    )

    start_time = time.time()
    image_path = os.path.join(config.IMAGE_DIR, sample['image_file'])

    # Step 1: 获取图像尺寸
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
            print(f"📐 图像尺寸: {img_w}x{img_h}")

            # 检查图像是否有效
            if img_w == 0 or img_h == 0:
                result.notes = f"无效图像尺寸: {img_w}x{img_h}"
                result.total_time = time.time() - start_time
                return result
    except Exception as e:
        result.notes = f"无法打开图像: {e}"
        result.total_time = time.time() - start_time
        return result

    # Step 2: Qwen初步回答 + 定位
    prompt1 = f"问题：{sample['question']} 请先给出答案；再以格式(左上角x坐标,左上角y坐标) (右下角x坐标,右下角y坐标) 两点生成的矩形框将图片需要关注区域包围进去。"
    print(f"📤 发送给Qwen的提示: {prompt1}")

    initial_response = call_qwen(prompt1, image_path, config)
    result.qwen_calls += 1

    if not initial_response:
        result.notes = "Qwen初步回答失败"
        result.total_time = time.time() - start_time
        return result

    print(f"📥 Qwen初步回答: {initial_response}")
    result.initial_answer = initial_response
    result.initial_bbox = extract_bbox_from_text(initial_response, img_w, img_h)
    print(f"📍 提取的BBox: {result.initial_bbox}")

    # Step 3: 闭环验证循环
    bbox_str = result.initial_bbox
    refined_answer = ""
    confidence = 0.0
    iteration = 0
    sam_failures = 0  # SAM失败计数器

    for retry in range(config.MAX_RETRIES + 1):
        iteration += 1
        print(f"🔄 第 {iteration} 次迭代尝试...")
        print(f"📍 使用BBox: {bbox_str}")

        # 调用SAM分割，并保存图像
        sam_success = call_sam(image_path, bbox_str, config,
                               save_segment=True, iteration=iteration)

        # === 关键修复：验证SAM调用是否成功 ===
        if not sam_success:
            sam_failures += 1
            result.notes = f"SAM分割失败 (第{sam_failures}次)"

            # 如果SAM连续失败2次，直接退出循环
            if sam_failures >= 2:
                print(f"⚠️ SAM连续失败{sam_failures}次，跳过此样本")
                break

            # 尝试使用回退策略：先扩大区域，再全图
            bbox_str = get_fallback_bbox(bbox_str, img_w, img_h, iteration)
            print(f"🔄 尝试回退BBox: {bbox_str}")
            continue  # 跳过后续步骤，继续下一次迭代

        result.sam_calls += 1
        sam_failures = 0  # 重置失败计数器

        # 检查证据图是否存在且有效
        if not os.path.exists(config.TEMP_EVIDENCE_PATH):
            result.notes = f"证据图未生成 (迭代{iteration})"
            # 尝试回退
            bbox_str = get_fallback_bbox(bbox_str, img_w, img_h, iteration)
            continue

        # 检查证据图是否为空
        try:
            evidence_size = os.path.getsize(config.TEMP_EVIDENCE_PATH)
            if evidence_size == 0:
                result.notes = f"证据图为空文件 (迭代{iteration})"
                bbox_str = get_fallback_bbox(bbox_str, img_w, img_h, iteration)
                continue
        except:
            result.notes = f"检查证据图失败 (迭代{iteration})"
            bbox_str = get_fallback_bbox(bbox_str, img_w, img_h, iteration)
            continue

        # === 额外验证：检查证据图是否能被正确打开 ===
        try:
            evidence_img = Image.open(config.TEMP_EVIDENCE_PATH)
            evidence_img.verify()  # 验证图像完整性
            evidence_img.close()
        except Exception as e:
            result.notes = f"证据图损坏或格式错误: {e}"
            # 删除损坏的文件
            try:
                os.remove(config.TEMP_EVIDENCE_PATH)
            except:
                pass

            # 尝试回退
            bbox_str = get_fallback_bbox(bbox_str, img_w, img_h, iteration)
            continue

        # 读取证据图用于后续处理
        try:
            with open(config.TEMP_EVIDENCE_PATH, "rb") as f:
                evidence_bytes = f.read()
        except Exception as e:
            result.notes = f"读取证据图失败: {e}"
            bbox_str = get_fallback_bbox(bbox_str, img_w, img_h, iteration)
            continue

        # Qwen基于证据图重新回答
        prompt2 = f"只看这张裁剪后的图像，回答：{sample['question']}"
        refined_answer = call_qwen(prompt2, config.TEMP_EVIDENCE_PATH, config)
        result.qwen_calls += 1

        if not refined_answer:
            result.notes = f"Qwen重答失败 (迭代{iteration})"
            # Qwen重答失败时，如果还有重试次数，尝试全图
            if retry < config.MAX_RETRIES:
                bbox_str = f"0,0,{img_w},{img_h}"
                continue
            else:
                break

        print(f"📥 Qwen精炼回答: {refined_answer}")

        # CLIP验证
        confidence = call_clip(evidence_bytes, refined_answer, config)
        result.clip_calls += 1
        result.clip_scores[f"iteration_{iteration}"] = float(confidence)

        print(f"🎯 CLIP置信度: {confidence:.3f} (阈值: {config.CONFIDENCE_THRESHOLD})")

        if confidence >= config.CONFIDENCE_THRESHOLD:
            result.refined_answer = refined_answer
            result.final_confidence = float(confidence)
            print(f"✅ 验证通过!")
            break
        elif retry < config.MAX_RETRIES:
            # 验证失败，如果还有重试次数，尝试全图
            print(f"⚠️ 第{retry + 1}次验证失败，尝试全图...")
            bbox_str = f"0,0,{img_w},{img_h}"
        else:
            print(f"⚠️ 第{retry + 1}次验证失败，达到最大重试次数")

    result.iteration_count = iteration
    result.total_time = time.time() - start_time

    # 如果精炼答案为空，使用初始答案
    if not result.refined_answer and result.initial_answer:
        result.refined_answer = result.initial_answer
        # 如果没有CLIP验证，使用默认置信度
        if result.final_confidence == 0.0:
            result.final_confidence = 0.5  # 默认中等置信度

    # 评估准确性
    answer_to_evaluate = result.refined_answer if result.refined_answer else result.initial_answer
    result.accuracy, result.is_correct = calculate_accuracy(
        answer_to_evaluate,
        sample['answers']
    )

    # 分析失败类型
    if not result.is_correct:
        result.failure_type = analyze_failure_type(result, config)
        print(f"❌ 答案错误，失败类型: {result.failure_type}")
    else:
        print(f"✅ 答案正确!")

    print(f"⏱️ 处理时间: {result.total_time:.2f}秒")
    print(f"🔄 迭代次数: {result.iteration_count}")

    return result

def save_sam_segment(segment_data: bytes, original_image_path: str,
                     bbox_str: str, iteration: int, config: Config):
    """保存SAM分割的图像"""
    # 创建目录
    os.makedirs(config.SAM_SEGMENTS_DIR, exist_ok=True)

    # 生成文件名
    base_name = os.path.splitext(os.path.basename(original_image_path))[0]
    if iteration == 1:
        suffix = "initial"
    elif iteration == 2:
        suffix = "full"
    else:
        suffix = f"retry{iteration}"

    # 简化bbox字符串用于文件名（移除逗号）
    bbox_simple = bbox_str.replace(',', '_')

    # 完整的文件名
    filename = f"{base_name}_{suffix}_{bbox_simple}.png"
    filepath = os.path.join(config.SAM_SEGMENTS_DIR, filename)

    # 保存文件
    with open(filepath, "wb") as f:
        f.write(segment_data)

    return filepath


# ==================== 实验管理 ====================
class ExperimentManager:
    def __init__(self, config: Config):
        self.config = config
        self.results: List[ExperimentResult] = []
        self.stats = SystemStatistics()

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.SAM_SEGMENTS_DIR, exist_ok=True)

    def run_experiments(self):
        """运行所有实验"""
        print("🚀 开始DocVQA实验...")

        # 加载数据
        samples = load_docvqa_dataset_for_experiment(self.config)
        self.stats.total_samples = len(samples)

        if self.stats.total_samples == 0:
            print("❌ 没有可用的样本，实验终止")
            return

        # 逐个运行实验
        for i, sample in enumerate(tqdm(samples, desc="进行实验")):
            print(f"\n{'=' * 60}")
            print(f"样本 {i + 1}/{len(samples)}: {sample['question']}")
            print(f"图像: {sample['image_file']}")
            print(f"参考答案: {sample['answers'][:3]}")  # 显示前3个参考答案

            result = run_single_experiment(sample, self.config)
            self.results.append(result)

            # 更新统计
            self.stats.correct_samples += 1 if result.is_correct else 0
            self.stats.total_iterations += result.iteration_count
            self.stats.total_sam_calls += result.sam_calls
            self.stats.total_clip_calls += result.clip_calls
            self.stats.total_qwen_calls += result.qwen_calls
            self.stats.total_time += result.total_time

            if result.failure_type:
                self.stats.failure_counts[result.failure_type] += 1

            # 每5个样本保存一次进度
            if (i + 1) % 5 == 0:
                self.save_results()
                print(f"\n💾 已保存{len(self.results)}个样本的结果")

        # 保存最终结果
        self.save_results()
        self.generate_report()
        print("\n✅ 实验完成!")

    def save_results(self):
        """保存实验结果"""
        # 转换结果为可序列化的字典
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

        results_dict = {
            'config': {
                'max_retries': int(self.config.MAX_RETRIES),
                'confidence_threshold': float(self.config.CONFIDENCE_THRESHOLD),
                'num_samples': int(self.config.NUM_SAMPLES),
                'random_seed': int(self.config.RANDOM_SEED)
            },
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
                'failure_counts': {k: int(v) for k, v in self.stats.failure_counts.items()}
            },
            'results': results_list
        }

        # 保存JSON格式的详细结果
        with open(self.config.RESULTS_JSON, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2, default=str)

        # 保存CSV格式的统计信息
        with open(self.config.STATS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                '样本ID', '问题', '图像', '参考答案',
                '初始答案', '精炼答案', '置信度',
                '是否正确', '准确率', '失败类型',
                '迭代次数', 'SAM调用', 'CLIP调用', 'Qwen调用',
                '时间(s)', '备注'
            ])

            for r in self.results:
                writer.writerow([
                    int(r.sample_id),
                    str(r.question)[:80],  # 针对DocVQA问题可能较长，增加截断长度
                    str(r.image_file),
                    '; '.join([str(ans) for ans in r.ground_truth_answers[:3]]),
                    str(r.initial_answer)[:50],
                    str(r.refined_answer)[:50],
                    f"{float(r.final_confidence):.3f}",
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
        report = f"""
# 可验证视觉问答闭环系统实验报告 - DocVQA数据集

## 1. 实验概述
- 数据集：DocVQA（{self.stats.total_samples}个样本）
- 闭环配置：最大迭代{self.config.MAX_RETRIES}次，置信度阈值{self.config.CONFIDENCE_THRESHOLD}
- 随机种子：{self.config.RANDOM_SEED}

## 2. 主要结果
- **总体准确率**：{self.stats.accuracy:.2%} ({self.stats.correct_samples}/{self.stats.total_samples})
- **平均迭代次数**：{self.stats.avg_iterations:.2f}
- **平均处理时间**：{self.stats.avg_time_per_sample:.2f}秒/样本
- **总实验时间**：{self.stats.total_time:.2f}秒

## 3. 工具调用统计
- SAM调用次数：{self.stats.total_sam_calls}
- CLIP调用次数：{self.stats.total_clip_calls}
- Qwen调用次数：{self.stats.total_qwen_calls}

## 4. 失败分析
"""

        total_failures = sum(self.stats.failure_counts.values())
        for failure_type, count in self.stats.failure_counts.items():
            if count > 0:
                percentage = count / total_failures * 100 if total_failures > 0 else 0
                report += f"- **{failure_type}**: {count}次 ({percentage:.1f}%)\n"

        report += """
## 5. DocVQA数据集特点分析
1. **文档类型多样**：包含表格、图表、票据、文档等
2. **文本密集**：需要精确的OCR能力
3. **数值问题多**：很多问题涉及数字和计算
4. **结构理解重要**：需要理解表格结构和文档布局

## 6. 关键发现
1. **定位挑战**：文档图像中的文本区域定位比自然图像更具挑战性
2. **OCR准确性**：Qwen的OCR能力对文档图像准确率影响大
3. **数值验证**：CLIP对数值类答案的验证效果需要进一步评估
4. **证据质量**：文档分割需要更精确的边界框

## 7. 改进建议
1. **预处理优化**：对文档图像进行增强预处理（去噪、二值化等）
2. **坐标提取改进**：针对文档坐标格式优化正则表达式
3. **多尺度验证**：尝试不同尺度的证据图进行验证
4. **后处理规则**：针对数值答案添加后处理规则

## 8. 样本示例
"""

        # 添加3个示例结果
        for i, r in enumerate(self.results[:3]):
            report += f"""
### 示例 {i + 1}
- **问题**: {r.question}
- **初始答案**: {r.initial_answer}
- **精炼答案**: {r.refined_answer}
- **置信度**: {r.final_confidence:.3f}
- **是否正确**: {'是' if r.is_correct else '否'}
- **处理时间**: {r.total_time:.2f}秒
- **失败类型**: {r.failure_type if r.failure_type else 'N/A'}
"""

        report_path = os.path.join(self.config.OUTPUT_DIR, "experiment_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📊 报告已保存至: {report_path}")


def validate_bbox(x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, min_size=20) -> Tuple[int, int, int, int]:
    """验证并修正bbox坐标，确保其有效"""
    # 确保坐标顺序正确
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    # 确保坐标在图像范围内
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    # 确保bbox有最小尺寸
    if (x2 - x1) < min_size:
        # 扩大宽度，保持中心不变
        center_x = (x1 + x2) // 2
        x1 = max(0, center_x - min_size // 2)
        x2 = min(img_w, center_x + min_size // 2)
        if (x2 - x1) < min_size:  # 如果还在边界处不够
            x2 = min(img_w, x1 + min_size)

    if (y2 - y1) < min_size:
        # 扩大高度，保持中心不变
        center_y = (y1 + y2) // 2
        y1 = max(0, center_y - min_size // 2)
        y2 = min(img_h, center_y + min_size // 2)
        if (y2 - y1) < min_size:  # 如果还在边界处不够
            y2 = min(img_h, y1 + min_size)

    return x1, y1, x2, y2


def extract_bbox_from_text(text: str, img_w: int, img_h: int) -> str:
    """从文本中提取bbox坐标，并验证修正"""
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
            # 验证并修正bbox
            x1, y1, x2, y2 = validate_bbox(x1, y1, x2, y2, img_w, img_h)
            return f"{x1},{y1},{x2},{y2}"

    # 未找到坐标，返回全图
    return f"0,0,{img_w},{img_h}"



# ==================== 主程序 ====================
def main():
    # 初始化配置
    config = Config()

    # 确保所有输出目录都存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.SAM_SEGMENTS_DIR, exist_ok=True)

    # 运行主实验
    print("=" * 60)
    print("🎓 计算机视觉结课论文实验系统 - DocVQA数据集")
    print("=" * 60)

    manager = ExperimentManager(config)
    manager.run_experiments()

    print(f"\n📁 所有结果已保存至: {config.OUTPUT_DIR}")
    print(f"📄 详细结果: {config.RESULTS_JSON}")
    print(f"📊 统计表格: {config.STATS_CSV}")
    print(f"📋 实验报告: {config.OUTPUT_DIR}/experiment_report.md")


if __name__ == "__main__":
    main()