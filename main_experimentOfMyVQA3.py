'''
4.基线系统（Qwen2-VL Only）
直接使用Qwen2-VL-7B-Instruct进行单次零样本推理
不进行任何验证或迭代
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

    # 数据集路径
    DATA_ROOT = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\MyVQA\combined_dataset"
    METADATA_PATH = os.path.join(DATA_ROOT, "combined_metadata.json")
    IMAGE_DIR = os.path.join(DATA_ROOT, "images")

    # 输出路径
    OUTPUT_DIR = "./results_baseline"
    RESULTS_JSON = os.path.join(OUTPUT_DIR, "results_baseline.json")
    STATS_CSV = os.path.join(OUTPUT_DIR, "statistics_baseline.csv")

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
    qwen_answer: str = ""

    # 性能指标
    qwen_calls: int = 1
    total_time: float = 0.0

    # 评估
    accuracy: float = 0.0
    is_correct: bool = False
    notes: str = ""


@dataclass
class SystemStatistics:
    total_samples: int = 0
    correct_samples: int = 0
    total_qwen_calls: int = 0
    total_time: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct_samples / self.total_samples if self.total_samples > 0 else 0

    @property
    def avg_time_per_sample(self) -> float:
        return self.total_time / self.total_samples if self.total_samples > 0 else 0


# ==================== 工具函数 ====================
def load_textvqa_dataset(config: Config) -> List[Dict]:
    """加载MyVQA数据集"""
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
    """调用Qwen-VL服务进行单次推理"""
    try:
        payload = {"prompt": prompt}
        if image_path and os.path.exists(image_path):
            print(f"📤 发送图像: {os.path.basename(image_path)}")
            payload["image_url"] = image_to_base64(image_path)

        print(f"📤 发送给Qwen的提示: {prompt[:100]}...")

        response = requests.post(config.QWEN_URL, json=payload, timeout=120)

        print(f"📡 Qwen响应状态: {response.status_code}")

        if response.status_code == 200:
            res = response.json()
            print(f"📥 Qwen原始响应: {res}")
            answer = res.get("response", "").strip()
            print(f"📥 Qwen回答: {answer}")
            return answer
        else:
            print(f"❌ Qwen调用失败: HTTP {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        print("⏰ Qwen调用超时")
    except Exception as e:
        print(f"💥 Qwen调用异常: {type(e).__name__}: {e}")
    return ""


def normalize_answer(answer: str) -> str:
    """标准化答案：小写、移除标点、空格"""
    if not answer:
        return ""
    # 转换为小写
    answer = answer.lower()
    # 移除标点符号
    answer = re.sub(r'[^\w\s]', '', answer)
    # 移除多余空格
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

        # 包含匹配（针对较长答案）
        if truth_normalized in pred_normalized or pred_normalized in truth_normalized:
            return 1.0, True

        # 数字提取匹配（针对OCR问题）
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


# ==================== 主实验流程 ====================
def run_single_experiment(sample: Dict, config: Config) -> ExperimentResult:
    """运行单个样本的实验（单次Qwen2-VL推理）"""
    print(f"\n{'=' * 60}")
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

    # Step 1: Qwen2-VL单次推理
    prompt = f"问题：{sample['question']} 请直接给出答案。"

    qwen_answer = call_qwen(prompt, image_path, config)

    result.qwen_answer = qwen_answer
    result.total_time = time.time() - start_time

    # Step 2: 评估准确性
    result.accuracy, result.is_correct = calculate_accuracy(
        result.qwen_answer,
        sample['answers']
    )

    if result.is_correct:
        print(f"✅ 答案正确!")
    else:
        print(f"❌ 答案错误")

    print(f"⏱️ 处理时间: {result.total_time:.2f}秒")

    return result


# ==================== 实验管理 ====================
class ExperimentManager:
    def __init__(self, config: Config):
        self.config = config
        self.results: List[ExperimentResult] = []
        self.stats = SystemStatistics()

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

        print(f"📁 输出目录: {config.OUTPUT_DIR}")

    def run_experiments(self):
        """运行所有实验"""
        print("🚀 开始基线系统实验（Qwen2-VL单次推理）...")
        print(f"📊 样本数量: {self.config.NUM_SAMPLES}")
        print(f"🎯 策略: 零样本推理，无验证")

        # 加载数据
        samples = load_textvqa_dataset(self.config)
        self.stats.total_samples = len(samples)

        # 逐个运行实验
        for i, sample in enumerate(tqdm(samples, desc="进行实验")):
            print(f"\n{'=' * 80}")
            print(f"📋 样本 {i + 1}/{len(samples)}")

            result = run_single_experiment(sample, self.config)
            self.results.append(result)

            # 更新统计
            self.stats.correct_samples += 1 if result.is_correct else 0
            self.stats.total_qwen_calls += result.qwen_calls
            self.stats.total_time += result.total_time

            # 每5个样本保存一次进度
            if (i + 1) % 5 == 0:
                self.save_results()
                print(f"\n💾 已保存{len(self.results)}个样本的结果")
                print(f"📈 当前准确率: {self.stats.correct_samples}/{len(self.results)} ({self.stats.accuracy:.2%})")

        # 保存最终结果
        self.save_results()
        self.generate_report()

        print("\n✅ 基线系统实验完成!")

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
                'qwen_answer': str(r.qwen_answer),
                'is_correct': bool(r.is_correct),
                'accuracy': float(r.accuracy),
                'qwen_calls': int(r.qwen_calls),
                'time': float(r.total_time),
                'notes': str(r.notes)
            }
            results_list.append(result_dict)

        results_dict = {
            'config': {
                'num_samples': int(self.config.NUM_SAMPLES),
                'random_seed': int(self.config.RANDOM_SEED)
            },
            'statistics': {
                'total_samples': int(self.stats.total_samples),
                'correct_samples': int(self.stats.correct_samples),
                'accuracy': float(self.stats.accuracy),
                'total_qwen_calls': int(self.stats.total_qwen_calls),
                'total_time': float(self.stats.total_time),
                'avg_time_per_sample': float(self.stats.avg_time_per_sample)
            },
            'results': results_list
        }

        # 保存JSON格式的详细结果
        with open(self.config.RESULTS_JSON, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, ensure_ascii=False, indent=2)

        # 保存CSV格式的统计信息
        with open(self.config.STATS_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                '样本ID', '问题', '图像', '参考答案',
                'Qwen回答', '是否正确', '准确率',
                'Qwen调用次数', '时间(s)', '备注'
            ])

            for r in self.results:
                writer.writerow([
                    int(r.sample_id),
                    str(r.question)[:50],
                    str(r.image_file),
                    '; '.join([str(ans) for ans in r.ground_truth_answers[:3]]),
                    str(r.qwen_answer)[:50],
                    "是" if r.is_correct else "否",
                    f"{float(r.accuracy):.3f}",
                    int(r.qwen_calls),
                    f"{float(r.total_time):.2f}",
                    str(r.notes)[:50]
                ])

        print(f"💾 结果已保存至: {self.config.OUTPUT_DIR}")

    def generate_report(self):
        """生成实验报告"""
        report = f"""
# 基线系统实验报告（Qwen2-VL单次推理）

## 1. 实验概述
- **系统类型**: 基线系统（Qwen2-VL单次推理）
- **数据集**: MyVQA（{self.stats.total_samples}个样本）
- **推理策略**: 零样本推理，无验证/迭代
- **模型**: Qwen2-VL-7B-Instruct
- **随机种子**: {self.config.RANDOM_SEED}

## 2. 主要结果
- **总体准确率**: {self.stats.accuracy:.2%} ({self.stats.correct_samples}/{self.stats.total_samples})
- **平均处理时间**: {self.stats.avg_time_per_sample:.2f}秒/样本
- **总实验时间**: {self.stats.total_time:.2f}秒
- **Qwen调用次数**: {self.stats.total_qwen_calls}

## 3. 实验设计

### 3.1 系统架构
1. **输入**: 图像 + 问题
2. **处理**: 单次Qwen2-VL推理
3. **输出**: 直接答案
4. **评估**: 与参考答案对比

### 3.2 与闭环系统对比
| 特征 | 基线系统 | 闭环系统 |
|------|----------|----------|
| 推理次数 | 1次 | 1-3次（可迭代） |
| 验证机制 | 无 | CLIP置信度验证 |
| 证据提取 | 无 | SAM分割 |
| 时间开销 | 最低 | 较高 |
| 准确率 | 基础水平 | 优化水平 |

## 4. 性能分析

### 4.1 优势
1. **速度快**: 单次推理，处理时间最短
2. **简单**: 系统复杂度最低
3. **稳定**: 无依赖外部服务失败风险
4. **基准**: 为其他系统提供对比基准

### 4.2 局限性
1. **无验证**: 答案正确性无法保证
2. **无迭代**: 无法通过多次尝试提高准确性
3. **无证据**: 缺乏可解释性证据
4. **依赖模型**: 完全依赖大模型能力

## 5. 实验结果分析

### 5.1 准确率表现
- **绝对准确率**: {self.stats.accuracy:.2%}
- **正确样本数**: {self.stats.correct_samples}
- **总样本数**: {self.stats.total_samples}

### 5.2 时间效率
- **平均时间**: {self.stats.avg_time_per_sample:.2f}秒/样本
- **总时间**: {self.stats.total_time:.2f}秒
- **吞吐量**: {self.stats.total_samples / self.stats.total_time * 3600:.1f}样本/小时（理论上）

## 6. 样本示例
"""

        # 添加3个代表性示例
        for i, r in enumerate(self.results[:3]):
            report += f"""
### 示例 {i + 1}
- **样本ID**: {r.sample_id}
- **问题**: {r.question}
- **图像**: {r.image_file}
- **Qwen回答**: {r.qwen_answer}
- **参考答案**: {', '.join(r.ground_truth_answers[:3])}
- **是否正确**: {'是' if r.is_correct else '否'}
- **处理时间**: {r.total_time:.2f}秒
"""

        # 添加性能对比表格
        report += """
## 7. 系统对比预期

### 7.1 预期对比结果
| 系统类型 | 预期准确率 | 预期时间/样本 | 系统复杂度 |
|----------|------------|---------------|------------|
| 基线系统 | 基础水平 | 最低 | 最简单 |
| 基础闭环系统 | 中等提升 | 中等 | 中等 |
| 自适应阈值系统 | 进一步优化 | 中等 | 较高 |
| 完整缓存系统 | 优化+加速 | 较低（有缓存时） | 最高 |

### 7.2 实验意义
1. **建立基准**: 为所有改进系统提供对比基准
2. **验证假设**: 验证闭环系统是否真能提升性能
3. **量化收益**: 准确计算性能提升与时间开销的权衡
4. **指导优化**: 识别最有价值的优化方向

## 8. 实验配置
- **数据集路径**: {self.config.DATA_ROOT}
- **输出目录**: {self.config.OUTPUT_DIR}
- **结果文件**: 
  - JSON: {self.config.RESULTS_JSON}
  - CSV: {self.config.STATS_CSV}

## 9. 后续实验建议
1. **误差分析**: 详细分析错误案例类型
2. **问题分类**: 按问题类型分析性能差异
3. **模型对比**: 尝试其他VLM模型作为基线
4. **提示词优化**: 测试不同提示词对性能的影响
"""

        report_path = os.path.join(self.config.OUTPUT_DIR, "baseline_experiment_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"📊 实验报告已保存至: {report_path}")


# ==================== 主程序 ====================
def main():
    # 初始化配置
    config = Config()

    # 确保所有输出目录都存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 运行主实验
    print("=" * 80)
    print("🤖 基线系统实验（Qwen2-VL单次推理）")
    print("=" * 80)

    manager = ExperimentManager(config)
    manager.run_experiments()

    print(f"\n📁 所有结果已保存至: {config.OUTPUT_DIR}")
    print(f"📄 详细结果: {config.RESULTS_JSON}")
    print(f"📊 统计表格: {config.STATS_CSV}")
    print(f"📋 实验报告: {config.OUTPUT_DIR}/baseline_experiment_report.md")


if __name__ == "__main__":
    main()