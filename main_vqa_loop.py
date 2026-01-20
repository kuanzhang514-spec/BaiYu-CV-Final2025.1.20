'''
测试一下方案行不行

'''

import requests
import base64
import os
from PIL import Image
from io import BytesIO
import json
import time

# ===== 配置 =====
SERVER_IP = "这是我的服务器IP地址，我隐藏了"
QWEN_URL = f"http://{SERVER_IP}:8020/chat_vl"
CLIP_URL = f"http://{SERVER_IP}:8021/clip/score"
SAM_URL = f"http://{SERVER_IP}:8022/sam/segment"

# 本地测试图
IMAGE_PATH = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\pingguo.jpg"
QUESTION = "What color is the apple?"

# 工具调用计数器
stats = {
    "qwen_calls": 0,
    "sam_calls": 0,
    "clip_calls": 0,
    "iterations": 0
}

def image_to_base64(image_path, max_size=(512, 512)):
    """将图像转为 base64（用于 Qwen3-VL）"""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail(max_size)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_qwen(prompt, image_path=None):
    """调用 Qwen3-VL"""
    stats["qwen_calls"] += 1
    payload = {"prompt": prompt}
    if image_path:
        payload["image_url"] = image_to_base64(image_path)
    try:
        response = requests.post(QWEN_URL, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        else:
            print(f"❌ Qwen error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"💥 Qwen exception: {e}")
        return ""

def call_sam(original_img_path, bbox_str):
    """调用 SAM，返回裁剪图的 bytes"""
    stats["sam_calls"] += 1
    with open(original_img_path, 'rb') as f:
        files = {'imagefile': ('image.jpg', f, 'image/jpeg')}
        data = {'bbox': bbox_str}
        response = requests.post(SAM_URL, files=files, data=data, timeout=30)
        if response.status_code == 200:
            return response.content  # PNG bytes
        else:
            print(f"❌ SAM error: {response.text}")
            return None

def call_clip(image_bytes, text):
    """调用 CLIP，返回相似度分数（假设 text 是唯一标签）"""
    stats["clip_calls"] += 1
    files = {'imagefile': ('evidence.png', image_bytes, 'image/png')}
    data = {'text': text}
    try:
        response = requests.post(CLIP_URL, files=files, data=data, timeout=10)
        if response.status_code == 200:
            res = response.json()
            # 你的 CLIP 返回 results: {"a photo of cat": {"similarity": 0.8}, ...}
            # 我们取第一个（或匹配 text 的）
            for label, val in res['results'].items():
                if text.lower() in label.lower() or label.lower() in text.lower():
                    return float(val['similarity'])
            # 如果没找到，返回第一个
            first_key = list(res['results'].keys())[0]
            return float(res['results'][first_key]['similarity'])
        else:
            print(f"❌ CLIP error: {response.text}")
            return 0.0
    except Exception as e:
        print(f"💥 CLIP exception: {e}")
        return 0.0

def extract_bbox_from_description(desc, img_w=1000, img_h=1000):
    """
    简化版：从描述中提取 bbox（实际项目可用 CLIP 网格定位）
    这里我们用启发式规则（仅作演示）：
    - 如果提到 "left" → x1=0, x2=img_w//2
    - 如果提到 "right" → x1=img_w//2, x2=img_w
    - 默认整个图
    """
    desc = desc.lower()
    if "left" in desc:
        return f"0,0,{img_w//2},{img_h}"
    elif "right" in desc:
        return f"{img_w//2},0,{img_w},{img_h}"
    elif "top" in desc:
        return f"0,0,{img_w},{img_h//2}"
    elif "bottom" in desc:
        return f"0,{img_h//2},{img_w},{img_h}"
    else:
        # 默认中心区域（可根据需求调整）
        cx, cy = img_w // 2, img_h // 2
        size = min(img_w, img_h) // 2
        x1, y1 = cx - size//2, cy - size//2
        x2, y2 = cx + size//2, cy + size//2
        return f"{x1},{y1},{x2},{y2}"

def main():
    print(f"🚀 开始 VQA 闭环任务")
    print(f"🖼️ 图像: {os.path.basename(IMAGE_PATH)}")
    print(f"❓ 问题: {QUESTION}\n")

    # Step 1: 初始提问，获取答案 + 证据描述
    prompt1 = f"Question: {QUESTION}. First, describe what visual region you need to see to answer this question. Then give your answer."
    response1 = call_qwen(prompt1, IMAGE_PATH)
    print(f"🧠 Qwen 初步回答:\n{response1}\n")

    # 简单解析：假设最后一句是答案，前面是描述
    lines = response1.split(". ")
    evidence_desc = ". ".join(lines[:-1]) + "."
    initial_answer = lines[-1].strip()

    # Step 2: 获取原始图像尺寸（用于 bbox）
    with Image.open(IMAGE_PATH) as img:
        img_w, img_h = img.size

    # Step 3: 提取 bbox（实际可用 CLIP 定位，此处简化）
    bbox = extract_bbox_from_description(evidence_desc, img_w, img_h)
    print(f"📍 提取 BBox: {bbox}")

    # Step 4: 调用 SAM 获取证据图
    evidence_img_bytes = call_sam(IMAGE_PATH, bbox)
    if not evidence_img_bytes:
        print("🛑 SAM 失败，终止流程")
        return

    # 保存证据图（可选）
    with open("evidence_step1.png", "wb") as f:
        f.write(evidence_img_bytes)

    # Step 5: 基于证据图再次提问
    prompt2 = f"Based ONLY on this image, answer: {QUESTION}"
    # 将 evidence_img_bytes 转为临时路径供 Qwen 使用
    evidence_temp_path = "temp_evidence.jpg"
    with Image.open(BytesIO(evidence_img_bytes)) as evidence_img:
        evidence_img.convert("RGB").save(evidence_temp_path, "JPEG")

    refined_answer = call_qwen(prompt2, evidence_temp_path)
    print(f"🎯 基于证据的回答: {refined_answer}")

    # Step 6: 用 CLIP 验证
    verification_text = f"The answer is {refined_answer}."
    clip_score = call_clip(evidence_img_bytes, verification_text)
    print(f"🔍 CLIP 验证分数: {clip_score:.3f}")

    # Step 7: 决策
    CONFIDENCE_THRESHOLD = 0.7
    final_answer = refined_answer
    stats["iterations"] = 1

    if clip_score < CONFIDENCE_THRESHOLD:
        print("⚠️ 置信度低，尝试扩大区域重试...")
        # 扩大 bbox（例如全图）
        full_bbox = f"0,0,{img_w},{img_h}"
        evidence_img_bytes2 = call_sam(IMAGE_PATH, full_bbox)
        if evidence_img_bytes2:
            with open("evidence_step2.png", "wb") as f:
                f.write(evidence_img_bytes2)
                print(f"📁 temp_evidence2.jpg 大小: {len(f.read())} bytes")
            with Image.open(BytesIO(evidence_img_bytes2)) as img2:
                img2.convert("RGB").save("temp_evidence2.jpg", "JPEG")
            refined_answer2 = call_qwen(prompt2, "temp_evidence2.jpg")
            print(f"🔁 第二次回答: {refined_answer2}")
            verification_text2 = f"The answer is {refined_answer2}."
            clip_score2 = call_clip(evidence_img_bytes2, verification_text2)
            print(f"🔄 第二次 CLIP 分数: {clip_score2:.3f}")
            stats["iterations"] = 2
            if clip_score2 > clip_score:
                final_answer = refined_answer2
                clip_score = clip_score2

    # 最终输出
    print("\n" + "="*50)
    print(f"✅ 最终答案: {final_answer}")
    print(f"📊 置信度: {clip_score:.3f}")
    print(f"📈 统计: {stats}")
    print("="*50)

    # 清理临时文件
    for f in ["temp_evidence.jpg", "temp_evidence2.jpg"]:
        if os.path.exists(f):
            os.remove(f)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\n⏱️ 总耗时: {time.time() - start_time:.2f} 秒")