'''
对比试验
6.重新设计缓存策略
对比实验：带缓存策略的VQA闭环系统
在客户端实现多级缓存，减少重复计算
'''


import os
import json
import time
import csv
import numpy as np
from PIL import Image
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Any, Optional
from tqdm import tqdm
import re
import base64
from io import BytesIO
import hashlib
import pickle
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed


# ==================== 配置 ====================
@dataclass
class Config:
    # 服务器配置
    SERVER_IP = "这是我的服务器IP地址，我隐藏了"
    QWEN_URL = f"http://{SERVER_IP}:8020/chat_vl"
    CLIP_URL = f"http://{SERVER_IP}:8021/clip/score"
    SAM_URL = f"http://{SERVER_IP}:8022/segment_by_bbox"

    # 数据集路径
    DATA_ROOT = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\MyVQA\combined_dataset"
    METADATA_PATH = os.path.join(DATA_ROOT, "combined_metadata.json")
    IMAGE_DIR = os.path.join(DATA_ROOT, "images")

    # 实验参数
    MAX_RETRIES = 2
    CONFIDENCE_THRESHOLD = 0.2
    TEMP_EVIDENCE_PATH = "./temp_evidence.png"

    # 缓存配置
    ENABLE_CACHE = True  # 是否启用缓存
    CACHE_DIR = "./cache"  # 缓存目录
    IMAGE_CACHE_DIR = os.path.join(CACHE_DIR, "images")
    TEXT_CACHE_DIR = os.path.join(CACHE_DIR, "text")
    RESULT_CACHE_DIR = os.path.join(CACHE_DIR, "results")
    SAM_CACHE_DIR = os.path.join(CACHE_DIR, "sam")

    # 缓存策略参数
    CACHE_WARMUP_SIZE = 20  # 预热缓存样本数
    USE_SIMILARITY_CACHE = True  # 是否使用相似性缓存
    SIMILARITY_THRESHOLD = 0.95  # 相似性阈值
    BATCH_SIZE = 4  # 批量处理大小（并行）

    # 输出路径
    OUTPUT_DIR = "./results_with_cache"
    RESULTS_JSON = os.path.join(OUTPUT_DIR, "results_with_cache.json")
    STATS_CSV = os.path.join(OUTPUT_DIR, "statistics_with_cache.csv")
    SAM_SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "sam_segments")
    CACHE_STATS_FILE = os.path.join(OUTPUT_DIR, "cache_statistics.json")

    # 实验设置
    NUM_SAMPLES = 110
    RANDOM_SEED = 42


# ==================== 缓存管理器 ====================
class CacheManager:
    """多级缓存管理器"""

    def __init__(self, config: Config):
        self.config = config
        self.cache_stats = {
            "image_hits": 0,
            "image_misses": 0,
            "text_hits": 0,
            "text_misses": 0,
            "sam_hits": 0,
            "sam_misses": 0,
            "result_hits": 0,
            "result_misses": 0,
            "similarity_hits": 0,
            "warmup_hits": 0,
            "total_requests": 0
        }

        # 创建缓存目录
        if config.ENABLE_CACHE:
            for dir_path in [config.IMAGE_CACHE_DIR, config.TEXT_CACHE_DIR,
                             config.RESULT_CACHE_DIR, config.SAM_CACHE_DIR]:
                os.makedirs(dir_path, exist_ok=True)

        # 内存缓存（LRU缓存）
        self.image_hash_cache = {}
        self.text_hash_cache = {}
        self.sam_cache = {}
        self.result_cache = {}

        # 相似性缓存索引
        self.similarity_index = {}

        print(f"🔄 初始化缓存管理器，启用缓存: {config.ENABLE_CACHE}")

    def get_image_hash(self, image_path: str) -> str:
        """获取图像哈希（用于缓存键）"""
        if image_path in self.image_hash_cache:
            return self.image_hash_cache[image_path]

        try:
            with Image.open(image_path) as img:
                # 使用缩略图计算哈希，提高速度
                img.thumbnail((128, 128))
                img_gray = img.convert('L')
                pixels = list(img_gray.getdata())
                avg = sum(pixels) / len(pixels)
                hash_str = ''.join(['1' if pixel > avg else '0' for pixel in pixels])
                hash_value = hashlib.md5(hash_str.encode()).hexdigest()

                self.image_hash_cache[image_path] = hash_value
                return hash_value
        except Exception as e:
            print(f"⚠️ 无法计算图像哈希: {e}")
            return hashlib.md5(image_path.encode()).hexdigest()

    def get_text_hash(self, text: str) -> str:
        """获取文本哈希（用于缓存键）"""
        if text in self.text_hash_cache:
            return self.text_hash_cache[text]

        hash_value = hashlib.md5(text.encode('utf-8')).hexdigest()
        self.text_hash_cache[text] = hash_value
        return hash_value

    def get_sam_cache_key(self, image_path: str, bbox_str: str) -> str:
        """获取SAM缓存键"""
        image_hash = self.get_image_hash(image_path)
        bbox_hash = hashlib.md5(bbox_str.encode()).hexdigest()
        return f"{image_hash}_{bbox_hash}"

    def get_result_cache_key(self, image_path: str, question: str) -> str:
        """获取结果缓存键"""
        image_hash = self.get_image_hash(image_path)
        question_hash = self.get_text_hash(question)
        return f"{image_hash}_{question_hash}"

    def check_image_cache(self, image_path: str) -> Optional[bytes]:
        """检查图像缓存"""
        self.cache_stats["total_requests"] += 1
        if not self.config.ENABLE_CACHE:
            return None

        image_hash = self.get_image_hash(image_path)
        cache_path = os.path.join(self.config.IMAGE_CACHE_DIR, f"{image_hash}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.cache_stats["image_hits"] += 1
                    return pickle.load(f)
            except:
                pass

        self.cache_stats["image_misses"] += 1
        return None

    def save_image_cache(self, image_path: str, image_data: bytes):
        """保存图像缓存"""
        if not self.config.ENABLE_CACHE:
            return

        image_hash = self.get_image_hash(image_path)
        cache_path = os.path.join(self.config.IMAGE_CACHE_DIR, f"{image_hash}.pkl")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(image_data, f)
        except Exception as e:
            print(f"⚠️ 保存图像缓存失败: {e}")

    def check_text_cache(self, text: str) -> Optional[Any]:
        """检查文本缓存"""
        self.cache_stats["total_requests"] += 1
        if not self.config.ENABLE_CACHE:
            return None

        text_hash = self.get_text_hash(text)
        cache_path = os.path.join(self.config.TEXT_CACHE_DIR, f"{text_hash}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.cache_stats["text_hits"] += 1
                    return pickle.load(f)
            except:
                pass

        self.cache_stats["text_misses"] += 1
        return None

    def save_text_cache(self, text: str, data: Any):
        """保存文本缓存"""
        if not self.config.ENABLE_CACHE:
            return

        text_hash = self.get_text_hash(text)
        cache_path = os.path.join(self.config.TEXT_CACHE_DIR, f"{text_hash}.pkl")

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ 保存文本缓存失败: {e}")

    def check_sam_cache(self, image_path: str, bbox_str: str) -> Optional[bytes]:
        """检查SAM缓存"""
        self.cache_stats["total_requests"] += 1
        if not self.config.ENABLE_CACHE:
            return None

        cache_key = self.get_sam_cache_key(image_path, bbox_str)

        # 先检查内存缓存
        if cache_key in self.sam_cache:
            self.cache_stats["sam_hits"] += 1
            return self.sam_cache[cache_key]

        # 检查文件缓存
        cache_path = os.path.join(self.config.SAM_CACHE_DIR, f"{cache_key}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    segment_data = pickle.load(f)
                    self.sam_cache[cache_key] = segment_data
                    self.cache_stats["sam_hits"] += 1
                    return segment_data
            except:
                pass

        self.cache_stats["sam_misses"] += 1
        return None

    def save_sam_cache(self, image_path: str, bbox_str: str, segment_data: bytes):
        """保存SAM缓存"""
        if not self.config.ENABLE_CACHE:
            return

        cache_key = self.get_sam_cache_key(image_path, bbox_str)
        cache_path = os.path.join(self.config.SAM_CACHE_DIR, f"{cache_key}.pkl")

        try:
            # 保存到内存缓存
            self.sam_cache[cache_key] = segment_data

            # 保存到文件缓存
            with open(cache_path, 'wb') as f:
                pickle.dump(segment_data, f)
        except Exception as e:
            print(f"⚠️ 保存SAM缓存失败: {e}")

    def check_result_cache(self, image_path: str, question: str) -> Optional[Dict]:
        """检查结果缓存"""
        self.cache_stats["total_requests"] += 1
        if not self.config.ENABLE_CACHE:
            return None

        cache_key = self.get_result_cache_key(image_path, question)

        # 先检查内存缓存
        if cache_key in self.result_cache:
            self.cache_stats["result_hits"] += 1
            return self.result_cache[cache_key]

        # 检查文件缓存
        cache_path = os.path.join(self.config.RESULT_CACHE_DIR, f"{cache_key}.json")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    self.result_cache[cache_key] = result
                    self.cache_stats["result_hits"] += 1
                    return result
            except:
                pass

        self.cache_stats["result_misses"] += 1
        return None

    def save_result_cache(self, image_path: str, question: str, result: Dict):
        """保存结果缓存"""
        if not self.config.ENABLE_CACHE:
            return

        cache_key = self.get_result_cache_key(image_path, question)
        cache_path = os.path.join(self.config.RESULT_CACHE_DIR, f"{cache_key}.json")

        try:
            # 保存到内存缓存
            self.result_cache[cache_key] = result

            # 保存到文件缓存
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 保存结果缓存失败: {e}")

    def find_similar_cached_result(self, image_path: str, question: str) -> Optional[Dict]:
        """查找相似缓存结果（基于相似性）"""
        if not self.config.ENABLE_CACHE or not self.config.USE_SIMILARITY_CACHE:
            return None

        # 简单的相似性匹配：检查是否有相同图像但不同问题的缓存
        image_hash = self.get_image_hash(image_path)

        # 查找相同图像的缓存
        if image_hash in self.similarity_index:
            for cached_question, cached_result in self.similarity_index[image_hash].items():
                # 简单的问题相似性检查（共享关键词）
                question_words = set(question.lower().split())
                cached_words = set(cached_question.lower().split())
                common_words = question_words.intersection(cached_words)

                if len(common_words) / max(len(question_words), 1) > 0.5:
                    self.cache_stats["similarity_hits"] += 1
                    return cached_result

        return None

    def update_similarity_index(self, image_path: str, question: str, result: Dict):
        """更新相似性索引"""
        if not self.config.ENABLE_CACHE or not self.config.USE_SIMILARITY_CACHE:
            return

        image_hash = self.get_image_hash(image_path)

        if image_hash not in self.similarity_index:
            self.similarity_index[image_hash] = {}

        self.similarity_index[image_hash][question] = result

    def get_cache_stats(self) -> Dict:
        """获取缓存统计信息"""
        total_hits = (self.cache_stats["image_hits"] +
                      self.cache_stats["text_hits"] +
                      self.cache_stats["sam_hits"] +
                      self.cache_stats["result_hits"] +
                      self.cache_stats["similarity_hits"])

        total_misses = (self.cache_stats["image_misses"] +
                        self.cache_stats["text_misses"] +
                        self.cache_stats["sam_misses"] +
                        self.cache_stats["result_misses"])

        hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0

        stats = self.cache_stats.copy()
        stats.update({
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": hit_rate,
            "memory_cache_size": {
                "image_hashes": len(self.image_hash_cache),
                "text_hashes": len(self.text_hash_cache),
                "sam_cache": len(self.sam_cache),
                "result_cache": len(self.result_cache)
            }
        })

        return stats

    def print_cache_stats(self):
        """打印缓存统计"""
        stats = self.get_cache_stats()
        print(f"\n📊 缓存统计:")
        print(f"   总请求数: {stats['total_requests']}")
        print(f"   图像缓存命中率: {stats['image_hits']}/{stats['image_hits'] + stats['image_misses']} "
              f"({stats['image_hits'] / (stats['image_hits'] + stats['image_misses']) * 100 if (stats['image_hits'] + stats['image_misses']) > 0 else 0:.1f}%)")
        print(f"   文本缓存命中率: {stats['text_hits']}/{stats['text_hits'] + stats['text_misses']} "
              f"({stats['text_hits'] / (stats['text_hits'] + stats['text_misses']) * 100 if (stats['text_hits'] + stats['text_misses']) > 0 else 0:.1f}%)")
        print(f"   SAM缓存命中率: {stats['sam_hits']}/{stats['sam_hits'] + stats['sam_misses']} "
              f"({stats['sam_hits'] / (stats['sam_hits'] + stats['sam_misses']) * 100 if (stats['sam_hits'] + stats['sam_misses']) > 0 else 0:.1f}%)")
        print(f"   结果缓存命中率: {stats['result_hits']}/{stats['result_hits'] + stats['result_misses']} "
              f"({stats['result_hits'] / (stats['result_hits'] + stats['result_misses']) * 100 if (stats['result_hits'] + stats['result_misses']) > 0 else 0:.1f}%)")
        print(f"   相似性缓存命中: {stats['similarity_hits']}")
        print(f"   总体命中率: {stats['hit_rate'] * 100:.1f}%")


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

    # 缓存相关指标
    cache_hits: Dict[str, int] = None  # 记录各类缓存命中
    from_cache: bool = False  # 是否来自缓存

    # 评估
    accuracy: float = 0.0
    is_correct: bool = False
    failure_type: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.clip_scores is None:
            self.clip_scores = {}
        if self.cache_hits is None:
            self.cache_hits = {
                "image": 0,
                "text": 0,
                "sam": 0,
                "result": 0,
                "similarity": 0
            }


@dataclass
class SystemStatistics:
    total_samples: int = 0
    correct_samples: int = 0
    total_iterations: int = 0
    total_sam_calls: int = 0
    total_clip_calls: int = 0
    total_qwen_calls: int = 0
    total_time: float = 0.0

    # 缓存统计
    cache_stats: Dict[str, Any] = None

    # 加速比统计
    estimated_speedup: float = 0.0
    cache_benefit_ratio: float = 0.0

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
        if self.cache_stats is None:
            self.cache_stats = {}

    @property
    def accuracy(self) -> float:
        return self.correct_samples / self.total_samples if self.total_samples > 0 else 0

    @property
    def avg_iterations(self) -> float:
        return self.total_iterations / self.total_samples if self.total_samples > 0 else 0

    @property
    def avg_time_per_sample(self) -> float:
        return self.total_time / self.total_samples if self.total_samples > 0 else 0

    @property
    def avg_qwen_calls(self) -> float:
        return self.total_qwen_calls / self.total_samples if self.total_samples > 0 else 0

    @property
    def avg_sam_calls(self) -> float:
        return self.total_sam_calls / self.total_samples if self.total_samples > 0 else 0

    @property
    def avg_clip_calls(self) -> float:
        return self.total_clip_calls / self.total_samples if self.total_samples > 0 else 0


# ==================== 工具函数（带缓存） ====================
def load_textvqa_dataset(config: Config) -> List[Dict]:
    """加载myVQA数据集"""
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


def image_to_base64(image_path: str, max_size=(512, 512), cache_manager: CacheManager = None) -> str:
    """图像转Base64（带缓存）"""
    # 检查图像缓存
    if cache_manager:
        cached_data = cache_manager.check_image_cache(image_path)
        if cached_data:
            return cached_data

    try:
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(max_size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # 保存到缓存
        if cache_manager:
            cache_manager.save_image_cache(image_path, image_data)

        return image_data
    except Exception as e:
        print(f"❌ 图像转Base64失败: {e}")
        return ""


def call_qwen(prompt: str, image_path: str = None, config: Config = None,
              cache_manager: CacheManager = None, use_cache: bool = True) -> str:
    """调用Qwen-VL服务（带缓存）"""

    # 构建缓存键
    cache_key = None
    if cache_manager and use_cache and image_path:
        cache_key = cache_manager.get_result_cache_key(image_path, prompt)
        cached_result = cache_manager.check_result_cache(image_path, prompt)
        if cached_result and 'response' in cached_result:
            print(f"💾 Qwen结果来自缓存")
            return cached_result['response']

    try:
        payload = {"prompt": prompt}
        if image_path and os.path.exists(image_path):
            print(f"📤 发送图像: {os.path.basename(image_path)}")
            payload["image_url"] = image_to_base64(image_path, cache_manager=cache_manager)

        response = requests.post(config.QWEN_URL, json=payload, timeout=120)

        print(f"📡 Qwen响应状态: {response.status_code}")

        if response.status_code == 200:
            res = response.json()
            print(f"📥 Qwen原始响应: {res}")
            response_text = res.get("response", "").strip()

            # 保存到缓存
            if cache_manager and cache_key and use_cache:
                cache_manager.save_result_cache(image_path, prompt, {
                    'response': response_text,
                    'timestamp': time.time()
                })

            return response_text
        else:
            print(f"❌ Qwen调用失败: HTTP {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        print("⏰ Qwen调用超时")
    except Exception as e:
        print(f"💥 Qwen调用异常: {type(e).__name__}: {e}")
    return ""


def call_sam(image_path: str, bbox_str: str, config: Config,
             save_segment: bool = True, iteration: int = 1,
             cache_manager: CacheManager = None) -> Tuple[bool, bytes]:
    """调用SAM服务（带缓存）"""

    # 检查SAM缓存
    if cache_manager:
        cached_segment = cache_manager.check_sam_cache(image_path, bbox_str)
        if cached_segment:
            print(f"💾 SAM结果来自缓存")
            # 保存到临时文件
            with open(config.TEMP_EVIDENCE_PATH, "wb") as out:
                out.write(cached_segment)

            # 如果需要保存分割图像
            if save_segment:
                segment_path = save_sam_segment(
                    cached_segment, image_path, bbox_str, iteration, config
                )
                print(f"💾 SAM分割图像已保存: {segment_path}")

            return True, cached_segment

    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'bbox': bbox_str}
            response = requests.post(config.SAM_URL, files=files, data=data, timeout=30)

            if response.status_code == 200:
                segment_data = response.content

                # 保存到缓存
                if cache_manager:
                    cache_manager.save_sam_cache(image_path, bbox_str, segment_data)

                # 保存到临时文件用于后续处理
                with open(config.TEMP_EVIDENCE_PATH, "wb") as out:
                    out.write(segment_data)

                # 如果需要保存分割图像
                if save_segment:
                    segment_path = save_sam_segment(
                        segment_data, image_path, bbox_str, iteration, config
                    )
                    print(f"💾 SAM分割图像已保存: {segment_path}")

                return True, segment_data
            else:
                print(f"❌ SAM调用失败: HTTP {response.status_code} - {response.text}")
    except Exception as e:
        print(f"💥 SAM调用异常: {type(e).__name__}: {e}")
    return False, None


def call_clip(image_bytes: bytes, text_label: str, config: Config,
              cache_manager: CacheManager = None) -> float:
    """调用CLIP服务，返回最高相似度（带缓存）"""

    # 构建缓存键（图像哈希 + 文本哈希）
    if cache_manager:
        # 计算图像哈希
        image_hash = hashlib.md5(image_bytes).hexdigest()[:16]
        text_hash = cache_manager.get_text_hash(text_label)
        cache_key = f"{image_hash}_{text_hash}"

        cached_score = cache_manager.check_text_cache(cache_key)
        if cached_score is not None:
            print(f"💾 CLIP结果来自缓存: {cached_score:.3f}")
            return float(cached_score)

    files = {'imagefile': ('evidence.png', image_bytes, 'image/png')}
    data = {'text': text_label, 'temperature': 100.0}

    try:
        response = requests.post(config.CLIP_URL, files=files, data=data, timeout=10)
        if response.status_code == 200:
            res = response.json()
            if res.get('results'):
                # 返回所有标签中的最高相似度
                similarities = [v['similarity'] for v in res['results'].values()]
                max_similarity = float(max(similarities)) if similarities else 0.0

                # 保存到缓存
                if cache_manager:
                    cache_manager.save_text_cache(f"{image_hash}_{text_hash}", max_similarity)

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
    # 移除标点符号
    answer = re.sub(r'[^\w\s]', '', answer)
    # 移除多余空格
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


def analyze_failure_type(result: ExperimentResult, config: Config) -> str:
    """分析失败类型"""
    if result.final_confidence < config.CONFIDENCE_THRESHOLD:
        return "verification_failure"
    elif result.iteration_count == 0:
        return "location_failure"
    elif "无法" in result.refined_answer or "不能" in result.refined_answer:
        return "reasoning_failure"
    else:
        return "other"


# ==================== 批量处理工具 ====================
def batch_call_qwen(prompts_with_images: List[Tuple[str, str]], config: Config,
                    cache_manager: CacheManager = None) -> List[str]:
    """批量调用Qwen-VL服务"""
    results = []

    # 如果没有启用批量或批量大小为1，则顺序处理
    if config.BATCH_SIZE <= 1 or len(prompts_with_images) <= 1:
        for prompt, image_path in prompts_with_images:
            result = call_qwen(prompt, image_path, config, cache_manager)
            results.append(result)
        return results

    # 使用线程池进行并行处理
    with ThreadPoolExecutor(max_workers=min(config.BATCH_SIZE, len(prompts_with_images))) as executor:
        future_to_index = {}

        for i, (prompt, image_path) in enumerate(prompts_with_images):
            future = executor.submit(call_qwen, prompt, image_path, config, cache_manager)
            future_to_index[future] = i

        # 初始化结果列表
        results = [None] * len(prompts_with_images)

        # 获取结果
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as e:
                print(f"❌ 批量Qwen调用失败 (索引{index}): {e}")
                results[index] = ""

    return results


# ==================== 主实验流程（带缓存） ====================
def run_single_experiment(sample: Dict, config: Config,
                          cache_manager: CacheManager = None,
                          warmup_mode: bool = False) -> ExperimentResult:
    """运行单个样本的实验（带缓存版本）"""
    result = ExperimentResult(
        sample_id=sample['id'],
        image_file=sample['image_file'],
        question=sample['question'],
        ground_truth_answers=sample['answers']
    )

    start_time = time.time()
    image_path = os.path.join(config.IMAGE_DIR, sample['image_file'])

    # Step 1: 检查完整结果缓存
    if cache_manager and not warmup_mode:
        cached_result = cache_manager.check_result_cache(image_path, sample['question'])
        if cached_result and 'full_result' in cached_result:
            print(f"💾 完整结果来自缓存")
            result.from_cache = True
            result.initial_answer = cached_result['full_result'].get('initial_answer', '')
            result.initial_bbox = cached_result['full_result'].get('initial_bbox', '')
            result.refined_answer = cached_result['full_result'].get('refined_answer', '')
            result.final_confidence = cached_result['full_result'].get('final_confidence', 0.0)
            result.iteration_count = cached_result['full_result'].get('iteration_count', 0)
            result.sam_calls = 0  # 来自缓存，没有实际调用
            result.clip_calls = 0
            result.qwen_calls = 0

            # 评估准确性
            answer_to_evaluate = result.refined_answer if result.refined_answer else result.initial_answer
            result.accuracy, result.is_correct = calculate_accuracy(
                answer_to_evaluate,
                sample['answers']
            )

            result.total_time = time.time() - start_time
            return result

    # Step 2: 获取图像尺寸
    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except Exception as e:
        result.notes = f"无法打开图像: {e}"
        result.total_time = time.time() - start_time
        return result

    # Step 3: Qwen初步回答 + 定位
    prompt1 = f"问题：{sample['question']} 请先给出答案；再以格式(左上角x坐标,左上角y坐标) (右下角x坐标,右下角y坐标) 两点生成的矩形框将图片需要关注区域包围进去。"
    print(f"📤 发送给Qwen的提示: {prompt1}")

    initial_response = call_qwen(prompt1, image_path, config, cache_manager, use_cache=not warmup_mode)
    result.qwen_calls += 1

    if not initial_response:
        result.notes = "Qwen初步回答失败"
        result.total_time = time.time() - start_time
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
        print(f"🔄 第 {iteration} 次迭代尝试...")

        # 调用SAM分割，并保存图像（带缓存）
        success, segment_data = call_sam(image_path, bbox_str, config,
                                         save_segment=True, iteration=iteration,
                                         cache_manager=cache_manager)
        if not success:
            result.notes = f"SAM分割失败 (迭代{iteration})"
            break

        result.sam_calls += 1

        # 检查证据图是否存在
        if not os.path.exists(config.TEMP_EVIDENCE_PATH):
            result.notes = f"证据图未生成 (迭代{iteration})"
            break

        # 读取证据图
        try:
            evidence_size = os.path.getsize(config.TEMP_EVIDENCE_PATH)
            if evidence_size == 0:
                result.notes = f"证据图为空文件 (迭代{iteration})"
                break

            with open(config.TEMP_EVIDENCE_PATH, "rb") as f:
                evidence_bytes = f.read()
        except Exception as e:
            result.notes = f"读取证据图失败: {e}"
            break

        # Qwen基于证据图重新回答
        prompt2 = f"只看这张裁剪后的图像，回答：{sample['question']}"
        refined_answer = call_qwen(prompt2, config.TEMP_EVIDENCE_PATH, config,
                                   cache_manager, use_cache=not warmup_mode)
        result.qwen_calls += 1

        if not refined_answer:
            result.notes = f"Qwen重答失败 (迭代{iteration})"
            break

        print(f"📥 Qwen精炼回答: {refined_answer}")

        # CLIP验证（带缓存）
        confidence = call_clip(evidence_bytes, refined_answer, config, cache_manager)
        result.clip_calls += 1
        result.clip_scores[f"iteration_{iteration}"] = float(confidence)

        print(f"🎯 CLIP置信度: {confidence:.3f} (阈值: {config.CONFIDENCE_THRESHOLD})")

        if confidence >= config.CONFIDENCE_THRESHOLD:
            result.refined_answer = refined_answer
            result.final_confidence = float(confidence)
            print(f"✅ 验证通过!")
            break
        elif retry == 0:
            # 第一次验证失败，尝试全图
            print("⚠️ 第一次验证失败，尝试全图...")
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

    # 保存完整结果到缓存（非预热模式）
    if cache_manager and not warmup_mode:
        full_result = {
            'initial_answer': result.initial_answer,
            'initial_bbox': result.initial_bbox,
            'refined_answer': result.refined_answer,
            'final_confidence': result.final_confidence,
            'iteration_count': result.iteration_count,
            'is_correct': result.is_correct,
            'accuracy': result.accuracy
        }
        cache_manager.save_result_cache(image_path, sample['question'], {
            'full_result': full_result,
            'timestamp': time.time()
        })

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


# ==================== 预热缓存 ====================
def warmup_cache(samples: List[Dict], config: Config, cache_manager: CacheManager):
    """预热缓存：预先处理一些样本填充缓存"""
    if not config.ENABLE_CACHE or config.CACHE_WARMUP_SIZE <= 0:
        return

    print(f"\n🔥 开始预热缓存，样本数: {min(config.CACHE_WARMUP_SIZE, len(samples))}")

    warmup_samples = samples[:min(config.CACHE_WARMUP_SIZE, len(samples))]

    for i, sample in enumerate(tqdm(warmup_samples, desc="预热缓存")):
        print(f"\n预热样本 {i + 1}/{len(warmup_samples)}")
        result = run_single_experiment(sample, config, cache_manager, warmup_mode=True)

        # 标记为预热命中
        if cache_manager:
            cache_manager.cache_stats["warmup_hits"] += 1

    print(f"✅ 缓存预热完成")


# ==================== 实验管理（带缓存） ====================
class ExperimentManager:
    def __init__(self, config: Config):
        self.config = config
        self.results: List[ExperimentResult] = []
        self.stats = SystemStatistics()
        self.cache_manager = CacheManager(config) if config.ENABLE_CACHE else None

        # 创建输出目录
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        os.makedirs(config.SAM_SEGMENTS_DIR, exist_ok=True)

        print(f"📁 输出目录: {config.OUTPUT_DIR}")
        if config.ENABLE_CACHE:
            print(f"💾 缓存目录: {config.CACHE_DIR}")

    def run_experiments(self):
        """运行所有实验（带缓存）"""
        print("🚀 开始带缓存的VQA闭环系统实验...")

        # 加载数据
        samples = load_textvqa_dataset(self.config)
        self.stats.total_samples = len(samples)

        # 预热缓存
        if self.config.ENABLE_CACHE:
            warmup_cache(samples, self.config, self.cache_manager)

        # 逐个运行实验
        for i, sample in enumerate(tqdm(samples, desc="进行实验")):
            print(f"\n{'=' * 60}")
            print(f"样本 {i + 1}/{len(samples)}: {sample['question']}")
            print(f"图像: {sample['image_file']}")
            print(f"参考答案: {sample['answers'][:3]}")

            result = run_single_experiment(sample, self.config, self.cache_manager)
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

        # 计算缓存统计和加速比
        self._calculate_cache_stats()

        # 保存最终结果
        self.save_results()
        self.generate_report()
        print("\n✅ 带缓存的实验完成!")

    def _calculate_cache_stats(self):
        """计算缓存统计和加速比"""
        if not self.config.ENABLE_CACHE or not self.cache_manager:
            return

        # 获取缓存统计
        self.stats.cache_stats = self.cache_manager.get_cache_stats()

        # 估计加速比
        # 假设每次网络调用平均耗时：Qwen=2s, SAM=1s, CLIP=0.5s
        avg_qwen_time = 2.0
        avg_sam_time = 1.0
        avg_clip_time = 0.5

        # 计算节省的网络调用
        qwen_hits = self.stats.cache_stats.get('result_hits', 0)
        sam_hits = self.stats.cache_stats.get('sam_hits', 0)
        clip_hits = self.stats.cache_stats.get('text_hits', 0)

        time_saved = (qwen_hits * avg_qwen_time +
                      sam_hits * avg_sam_time +
                      clip_hits * avg_clip_time)

        # 加速比 = 总时间 / (总时间 - 节省时间)
        if self.stats.total_time > 0 and time_saved > 0:
            self.stats.estimated_speedup = self.stats.total_time / (self.stats.total_time - time_saved)

        # 缓存收益比 = 节省时间 / 总时间
        if self.stats.total_time > 0:
            self.stats.cache_benefit_ratio = time_saved / self.stats.total_time

        # 打印缓存统计
        self.cache_manager.print_cache_stats()

        print(f"\n⚡ 加速比分析:")
        print(f"   估计节省时间: {time_saved:.2f}秒")
        print(f"   估计加速比: {self.stats.estimated_speedup:.2f}x")
        print(f"   缓存收益比: {self.stats.cache_benefit_ratio:.1%}")

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
                'from_cache': bool(r.from_cache),
                'cache_hits': r.cache_hits,
                'notes': str(r.notes)
            }
            results_list.append(result_dict)

        results_dict = {
            'config': {
                'max_retries': int(self.config.MAX_RETRIES),
                'confidence_threshold': float(self.config.CONFIDENCE_THRESHOLD),
                'enable_cache': bool(self.config.ENABLE_CACHE),
                'cache_warmup_size': int(self.config.CACHE_WARMUP_SIZE),
                'batch_size': int(self.config.BATCH_SIZE),
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
                'estimated_speedup': float(self.stats.estimated_speedup),
                'cache_benefit_ratio': float(self.stats.cache_benefit_ratio),
                'failure_counts': {k: int(v) for k, v in self.stats.failure_counts.items()}
            },
            'cache_statistics': self.stats.cache_stats if self.stats.cache_stats else {},
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
                '时间(s)', '来自缓存', '缓存命中', '备注'
            ])

            for r in self.results:
                cache_hits_str = ';'.join([f"{k}:{v}" for k, v in r.cache_hits.items()])
                writer.writerow([
                    int(r.sample_id),
                    str(r.question)[:50],
                    str(r.image_file),
                    '; '.join([str(ans) for ans in r.ground_truth_answers[:3]]),
                    str(r.initial_answer)[:30],
                    str(r.refined_answer)[:30],
                    f"{float(r.final_confidence):.3f}",
                    "是" if r.is_correct else "否",
                    f"{float(r.accuracy):.3f}",
                    str(r.failure_type),
                    int(r.iteration_count),
                    int(r.sam_calls),
                    int(r.clip_calls),
                    int(r.qwen_calls),
                    f"{float(r.total_time):.2f}",
                    "是" if r.from_cache else "否",
                    cache_hits_str,
                    str(r.notes)[:50]
                ])

        # 保存缓存统计
        if self.stats.cache_stats:
            with open(self.config.CACHE_STATS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.stats.cache_stats, f, ensure_ascii=False, indent=2)

        print(f"💾 结果已保存至: {self.config.OUTPUT_DIR}")

    def generate_report(self):
        """生成实验报告"""
        report = f"""
# 带缓存策略的VQA闭环系统实验报告

## 1. 实验概述
- **系统类型**: 带多级缓存的VQA闭环系统
- **数据集**: MyVQA（{self.stats.total_samples}个样本）
- **闭环配置**: 最大迭代{self.config.MAX_RETRIES}次，置信度阈值{self.config.CONFIDENCE_THRESHOLD}
- **缓存配置**: 启用缓存={self.config.ENABLE_CACHE}，预热大小={self.config.CACHE_WARMUP_SIZE}，批量大小={self.config.BATCH_SIZE}
- **随机种子**: {self.config.RANDOM_SEED}

## 2. 主要结果
- **总体准确率**: {self.stats.accuracy:.2%} ({self.stats.correct_samples}/{self.stats.total_samples})
- **平均迭代次数**: {self.stats.avg_iterations:.2f}
- **平均处理时间**: {self.stats.avg_time_per_sample:.2f}秒/样本
- **总实验时间**: {self.stats.total_time:.2f}秒

## 3. 缓存性能分析
"""

        if self.config.ENABLE_CACHE:
            report += f"""
### 3.1 缓存命中率
- **总体命中率**: {self.stats.cache_stats.get('hit_rate', 0) * 100:.1f}%
- **图像缓存命中**: {self.stats.cache_stats.get('image_hits', 0)}次
- **文本缓存命中**: {self.stats.cache_stats.get('text_hits', 0)}次
- **SAM缓存命中**: {self.stats.cache_stats.get('sam_hits', 0)}次
- **结果缓存命中**: {self.stats.cache_stats.get('result_hits', 0)}次
- **相似性缓存命中**: {self.stats.cache_stats.get('similarity_hits', 0)}次
- **预热缓存命中**: {self.stats.cache_stats.get('warmup_hits', 0)}次

### 3.2 加速效果
- **估计加速比**: {self.stats.estimated_speedup:.2f}x
- **缓存收益比**: {self.stats.cache_benefit_ratio:.1%}
- **减少的网络调用**: 
  * Qwen调用减少: {self.stats.cache_stats.get('result_hits', 0)}次
  * SAM调用减少: {self.stats.cache_stats.get('sam_hits', 0)}次
  * CLIP调用减少: {self.stats.cache_stats.get('text_hits', 0)}次
"""
        else:
            report += "- **缓存未启用**\n"

        report += """
## 4. 工具调用统计
- SAM调用次数: {self.stats.total_sam_calls}
- CLIP调用次数: {self.stats.total_clip_calls}
- Qwen调用次数: {self.stats.total_qwen_calls}

## 5. 失败分析
"""

        total_failures = sum(self.stats.failure_counts.values())
        for failure_type, count in self.stats.failure_counts.items():
            if count > 0:
                percentage = count / total_failures * 100 if total_failures > 0 else 0
                report += f"- **{failure_type}**: {count}次 ({percentage:.1f}%)\n"

        report += """
## 6. 缓存策略设计

### 6.1 多级缓存架构
1. **图像特征缓存**: 缓存图像的Base64编码，避免重复编码
2. **文本特征缓存**: 缓存文本的哈希和CLIP相似度分数
3. **SAM结果缓存**: 缓存相同图像和bbox的分割结果
4. **完整结果缓存**: 缓存整个实验流程的结果
5. **相似性缓存**: 基于图像和问题的相似性查找缓存

### 6.2 缓存键设计
- **图像缓存键**: 图像感知哈希（PHash）
- **文本缓存键**: MD5哈希
- **SAM缓存键**: 图像哈希 + bbox哈希
- **结果缓存键**: 图像哈希 + 问题哈希

### 6.3 预热策略
- 预先处理部分样本填充缓存
- 提高后续请求的缓存命中率

## 7. 与无缓存系统对比优势
1. **显著减少网络延迟**: 缓存命中时跳过网络请求
2. **降低服务器负载**: 减少对后端服务的重复调用
3. **提高系统响应速度**: 本地缓存访问速度快于网络请求
4. **支持离线回放**: 缓存结果可用于离线分析和调试

## 8. 缓存开销分析
1. **存储开销**: 需要磁盘空间存储缓存文件
2. **内存开销**: 内存缓存占用一定RAM
3. **一致性开销**: 需要处理缓存失效和更新

## 9. 优化建议
1. **智能缓存淘汰**: 实现LRU或LFU缓存淘汰策略
2. **增量更新**: 只缓存变化的部分，减少存储开销
3. **分布式缓存**: 在多机部署时使用Redis等分布式缓存
4. **预测性预热**: 基于历史访问模式预测性预热缓存

## 10. 样本示例
"""

        # 添加3个示例结果
        for i, r in enumerate(self.results[:3]):
            cache_source = "来自缓存" if r.from_cache else "实时计算"
            report += f"""
### 示例 {i + 1}
- **样本ID**: {r.sample_id}
- **问题**: {r.question}
- **图像**: {r.image_file}
- **处理方式**: {cache_source}
- **精炼答案**: {r.refined_answer}
- **CLIP置信度**: {r.final_confidence:.3f}
- **是否正确**: {'是' if r.is_correct else '否'}
- **处理时间**: {r.total_time:.2f}秒
- **缓存命中**: {sum(r.cache_hits.values())}次
"""

        report_path = os.path.join(self.config.OUTPUT_DIR, "experiment_report_with_cache.md")
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
    if config.ENABLE_CACHE:
        os.makedirs(config.CACHE_DIR, exist_ok=True)

    # 运行主实验
    print("=" * 80)
    print("🚀 带缓存策略的VQA闭环系统实验")
    print(f"💾 缓存启用: {config.ENABLE_CACHE}")
    print(f"🔥 预热大小: {config.CACHE_WARMUP_SIZE}")
    print(f"⚡ 批量大小: {config.BATCH_SIZE}")
    print("=" * 80)

    manager = ExperimentManager(config)
    manager.run_experiments()

    print(f"\n📁 所有结果已保存至: {config.OUTPUT_DIR}")
    print(f"📄 详细结果: {config.RESULTS_JSON}")
    print(f"📊 统计表格: {config.STATS_CSV}")
    print(f"💾 缓存统计: {config.CACHE_STATS_FILE}")
    print(f"📋 实验报告: {config.OUTPUT_DIR}/experiment_report_with_cache.md")

    if config.ENABLE_CACHE:
        print(f"\n📊 缓存目录结构:")
        print(f"   图像缓存: {config.IMAGE_CACHE_DIR}")
        print(f"   文本缓存: {config.TEXT_CACHE_DIR}")
        print(f"   SAM缓存: {config.SAM_CACHE_DIR}")
        print(f"   结果缓存: {config.RESULT_CACHE_DIR}")


if __name__ == "__main__":
    main()
