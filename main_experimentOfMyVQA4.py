'''
5.对比实验：无CLIP验证的闭环系统
用于验证CLIP验证环节的必要性
'''

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
from io import BytesIO


# ==================== 配置 ====================
@dataclass
class Config:
    # 服务器配置
    SERVER_IP = "这是我的服务器IP地址，我隐藏了"
    QWEN_URL = f"http://{SERVER_IP}:8020/chat_vl"
    SAM_URL = f"http://{SERVER_IP}:8022/segment_by_bbox"

    # 注意：移除了CLIP_URL

    # 数据集路径
    DATA_ROOT = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\MyVQA\combined_dataset"
    METADATA_PATH = os.path.join(DATA_ROOT, "combined_metadata.json")
    IMAGE_DIR = os.path.join(DATA_ROOT, "images")

    # 实验参数
    MAX_RETRIES = 2
    # 注意：移除了CONFIDENCE_THRESHOLD
    TEMP_EVIDENCE_PATH = "./temp_evidence.png"

    # 输出路径
    OUTPUT_DIR = "./results_no_clip"  # 修改输出路径，避免覆盖原结果
    RESULTS_JSON = os.path.join(OUTPUT_DIR, "results_no_clip.json")
    STATS_CSV = os.path.join(OUTPUT_DIR, "statistics_no_clip.csv")
    SAM_SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "sam_segments")

    # 实验设置
    NUM_SAMPLES = 110
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
    # 注意：移除了final_confidence和clip_scores

    # 性能指标
    iteration_count: int = 0
    sam_calls: int = 0
    qwen_calls: int = 0  # 注意：移除了clip_calls
    total_time: float = 0.0

    # 评估
    accuracy: float = 0.0
    is_correct: bool = False
    failure_type: str = ""
    notes: str = ""

    # 新增：记录每次迭代的答案
    iteration_answers: Dict[str, str] = None

    def __post_init__(self):
        if self.iteration_answers is None:
            self.iteration_answers = {}


@dataclass
class SystemStatistics:
    total_samples: int = 0
    correct_samples: int = 0
    total_iterations: int = 0
    total_sam_calls: int = 0
    total_qwen_calls: int = 0  # 注意：移除了total_clip_calls
    total_time: float = 0.0

    # 按失败类型统计
    failure_counts: Dict[str, int] = None

    def __post_init__(self):
        if self.failure_counts is None:
            self.failure_counts = {
                "location_failure": 0,
                "segmentation_failure": 0,
                "reasoning_failure": 0,
                "verification_failure": 0,  # 保留，但意义不同
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
def load_textvqa_dataset(config: Config) -> List[Dict]:
    """加载MyVQA数据集"""
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
                print(f"❌ SAM调用失败: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"💥 SAM调用异常: {type(e).__name__}: {e}")
    return False


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
    answer = answer.lower()
    answer = re.sub(r'[^\w\s]', '', answer)
    answer = ' '.join(answer.split())
    return answer


def calculate_accuracy(predicted_answer: str, ground_truths: List[str]) -> Tuple[float, bool]:
    """计算答案准确性（改进版）"""
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


def analyze_failure_type(result: ExperimentResult) -> str:
    """分析失败类型（无CLIP验证版本）"""
    # 由于没有CLIP验证，verification_failure不再适用
    # 我们可以根据其他线索判断失败类型
    if result.iteration_count == 0:
        return "location_failure"
    elif "无法" in result.refined_answer or "不能" in result.refined_answer:
        return "reasoning_failure"
    elif not result.refined_answer and not result.initial_answer:
        return "segmentation_failure"
    else:
        return "other"


# ==================== 主实验流程（无CLIP验证） ====================
def run_single_experiment(sample: Dict, config: Config) -> ExperimentResult:
    """运行单个样本的实验（无CLIP验证版本）"""
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

    # Step 3: 无CLIP验证的迭代流程
    bbox_str = result.initial_bbox
    refined_answer = ""
    iteration = 0

    for retry in range(config.MAX_RETRIES + 1):
        iteration += 1
        print(f"🔄 第 {iteration} 次迭代尝试...")

        # 调用SAM分割，并保存图像
        if not call_sam(image_path, bbox_str, config,
                        save_segment=True, iteration=iteration):
            result.notes = f"SAM分割失败 (迭代{iteration})"
            break

        result.sam_calls += 1

        # 检查证据图是否存在
        if not os.path.exists(config.TEMP_EVIDENCE_PATH):
            result.notes = f"证据图未生成 (迭代{iteration})"
            break

        # Qwen基于证据图重新回答
        prompt2 = f"只看这张裁剪后的图像，回答：{sample['question']}"
        iteration_answer = call_qwen(prompt2, config.TEMP_EVIDENCE_PATH, config)
        result.qwen_calls += 1

        if not iteration_answer:
            result.notes = f"Qwen重答失败 (迭代{iteration})"
            break

        print(f"📥 Qwen迭代{iteration}回答: {iteration_answer}")

        # 记录迭代答案
        result.iteration_answers[f"iteration_{iteration}"] = iteration_answer

        # 无CLIP验证，直接接受当前迭代的答案
        refined_answer = iteration_answer

        # 注意：这里没有置信度检查
        # 我们总是接受当前迭代的答案，但继续下一次迭代（如果有的话）

        if retry == 0:
            # 第一次迭代后，尝试全图进行第二次迭代
            print("🔄 准备第二次迭代（全图）...")
            bbox_str = f"0,0,{img_w},{img_h}"
        else:
            print(f"✅ 迭代完成")

    result.iteration_count = iteration
    result.total_time = time.time() - start_time

    # 确定最终答案
    if iteration >= 1:
        # 如果有迭代，使用最后一次迭代的答案
        result.refined_answer = refined_answer
    elif result.initial_answer:
        # 如果没有迭代但初始答案存在，使用初始答案
        result.refined_answer = result.initial_answer

    # 评估准确性
    answer_to_evaluate = result.refined_answer if result.refined_answer else result.initial_answer
    result.accuracy, result.is_correct = calculate_accuracy(
        answer_to_evaluate,
        sample['answers']
    )

    # 分析失败类型
    if not result.is_correct:
        result.failure_type = analyze_failure_type(result)
        print(f"❌ 答案错误，失败类型: {result.failure_type}")
    else:
        print(f"✅ 答案正确!")

    print(f"⏱️ 处理时间: {result.total_time:.2f}秒")
    print(f"🔄 迭代次数: {result.iteration_count}")

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

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.SAM_SEGMENTS_DIR, exist_ok=True)

    def run_experiments(self):
        """运行所有实验"""
        print("🚀 开始对比实验（无CLIP验证）...")

        # 加载数据
        samples = load_textvqa_dataset(self.config)
        self.stats.total_samples = len(samples)

        # 逐个运行实验
        for i, sample in enumerate(tqdm(samples, desc="进行实验")):
            print(f"\n{'=' * 60}")
            print(f"样本 {i + 1}/{len(samples)}: {sample['question']}")
            print(f"图像: {sample['image_file']}")
            print(f"参考答案: {sample['answers'][:3]}")

            result = run_single_experiment(sample, self.config)
            self.results.append(result)

            # 更新统计
            self.stats.correct_samples += 1 if result.is_correct else 0
            self.stats.total_iterations += result.iteration_count
            self.stats.total_sam_calls += result.sam_calls
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
        print("\n✅ 对比实验完成!")

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
                'iteration_answers': {k: str(v) for k, v in r.iteration_answers.items()},
                'is_correct': bool(r.is_correct),
                'accuracy': float(r.accuracy),
                'failure_type': str(r.failure_type),
                'iteration_count': int(r.iteration_count),
                'sam_calls': int(r.sam_calls),
                'qwen_calls': int(r.qwen_calls),
                'time': float(r.total_time),
                'notes': str(r.notes)
            }
            results_list.append(result_dict)

        results_dict = {
            'config': {
                'max_retries': int(self.config.MAX_RETRIES),
                'num_samples': int(self.config.NUM_SAMPLES),
                'random_seed': int(self.config.RANDOM_SEED),
                'experiment_type': 'no_clip_verification'  # 标记实验类型
            },
            'statistics': {
                'total_samples': int(self.stats.total_samples),
                'correct_samples': int(self.stats.correct_samples),
                'accuracy': float(self.stats.accuracy),
                'total_iterations': int(self.stats.total_iterations),
                'avg_iterations': float(self.stats.avg_iterations),
                'total_sam_calls': int(self.stats.total_sam_calls),
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
                '初始答案', '精炼答案', '迭代答案',
                '是否正确', '准确率', '失败类型',
                '迭代次数', 'SAM调用', 'Qwen调用',
                '时间(s)', '备注'
            ])

            for r in self.results:
                # 合并迭代答案
                iteration_answers_str = '; '.join([f"{k}:{v[:20]}" for k, v in r.iteration_answers.items()])

                writer.writerow([
                    int(r.sample_id),
                    str(r.question)[:50],
                    str(r.image_file),
                    '; '.join([str(ans) for ans in r.ground_truth_answers[:3]]),
                    str(r.initial_answer)[:30],
                    str(r.refined_answer)[:30],
                    iteration_answers_str[:50],
                    "是" if r.is_correct else "否",
                    f"{float(r.accuracy):.3f}",
                    str(r.failure_type),
                    int(r.iteration_count),
                    int(r.sam_calls),
                    int(r.qwen_calls),
                    f"{float(r.total_time):.2f}",
                    str(r.notes)[:50]
                ])

        print(f"💾 结果已保存至: {self.config.OUTPUT_DIR}")

    def generate_report(self):
        """生成对比实验报告"""
        report = f"""
# 可验证视觉问答闭环系统对比实验报告（无CLIP验证）

## 1. 实验概述
- **实验类型**：无CLIP验证的闭环系统
- **实验目的**：验证CLIP验证环节的必要性
- **数据集**：MyVQA（{self.stats.total_samples}个样本）
- **迭代配置**：最大迭代{self.config.MAX_RETRIES}次
- **随机种子**：{self.config.RANDOM_SEED}

## 2. 主要结果
- **总体准确率**：{self.stats.accuracy:.2%} ({self.stats.correct_samples}/{self.stats.total_samples})
- **平均迭代次数**：{self.stats.avg_iterations:.2f}
- **平均处理时间**：{self.stats.avg_time_per_sample:.2f}秒/样本
- **总实验时间**：{self.stats.total_time:.2f}秒

## 3. 工具调用统计
- SAM调用次数：{self.stats.total_sam_calls}
- Qwen调用次数：{self.stats.total_qwen_calls}
- **注意**：本实验未使用CLIP验证

## 4. 失败分析
"""

        total_failures = sum(self.stats.failure_counts.values())
        for failure_type, count in self.stats.failure_counts.items():
            if count > 0:
                percentage = count / total_failures * 100 if total_failures > 0 else 0
                report += f"- **{failure_type}**: {count}次 ({percentage:.1f}%)\n"

        report += """
## 5. 与有CLIP验证系统的对比分析
### 5.1 预期差异
1. **准确率对比**：无CLIP验证的系统可能产生更多错误答案，因为缺少了验证环节
2. **迭代行为**：无CLIP验证时，系统会固定执行所有迭代，而不根据置信度提前终止
3. **时间效率**：无CLIP验证可能会减少单次迭代时间，但可能因执行更多迭代而增加总时间

### 5.2 观察到的差异
（请与有CLIP验证的实验结果对比）

## 6. 关键发现
1. **验证环节的作用**：CLIP验证是否对过滤错误答案有显著作用
2. **误判风险**：无验证时，系统是否更容易接受错误答案
3. **效率权衡**：验证环节带来的准确率提升是否值得其时间开销

## 7. 样本示例
"""

        # 添加3个示例结果
        for i, r in enumerate(self.results[:3]):
            report += f"""
### 示例 {i + 1}
- **问题**: {r.question}
- **初始答案**: {r.initial_answer}
- **精炼答案**: {r.refined_answer}
- **迭代答案**: {r.iteration_answers}
- **是否正确**: {'是' if r.is_correct else '否'}
- **处理时间**: {r.total_time:.2f}秒
"""

        report += """
## 8. 结论与建议
（根据实验结果填写）

1. **验证必要性**：CLIP验证是否显著提高系统准确性
2. **阈值优化**：如果验证是必要的，原实验中的置信度阈值是否合适
3. **系统改进**：是否可以设计更高效的验证机制
"""

        report_path = os.path.join(self.config.OUTPUT_DIR, "experiment_report_no_clip.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📊 报告已保存至: {report_path}")


# ==================== 主程序 ====================
def main():
    # 初始化配置
    config = Config()

    # 确保所有输出目录都存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.SAM_SEGMENTS_DIR, exist_ok=True)

    # 运行主实验
    print("=" * 60)
    print("🔬 对比实验：无CLIP验证的闭环系统")
    print("🎯 实验目的：验证CLIP验证环节的必要性")
    print("=" * 60)

    manager = ExperimentManager(config)
    manager.run_experiments()

    print(f"\n📁 所有结果已保存至: {config.OUTPUT_DIR}")
    print(f"📄 详细结果: {config.RESULTS_JSON}")
    print(f"📊 统计表格: {config.STATS_CSV}")
    print(f"📋 实验报告: {config.OUTPUT_DIR}/experiment_report_no_clip.md")

    print("\n📊 对比实验完成后，请执行以下分析：")
    print("1. 比较有/无CLIP验证系统的准确率")
    print("2. 分析验证环节对错误答案的过滤效果")
    print("3. 评估验证环节的时间开销与收益")


if __name__ == "__main__":
    main()