'''
这个文档试验一下Qwen + CLIP + SAM 行不行

'''
import os
import time
import json
import base64  # ← 已补全
import requests
from PIL import Image
from io import BytesIO

# ==================== 全局配置 ====================
SERVER_IP = "这是我的服务器IP地址，我隐藏了"
QWEN_URL = f"http://{SERVER_IP}:8020/chat_vl"
CLIP_URL = f"http://{SERVER_IP}:8021/clip/score"
SAM_URL = f"http://{SERVER_IP}:8022/segment_by_bbox"

# 输入图像路径（Windows 格式）
INPUT_IMAGE_PATH = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\1234.jpg"
QUESTION = "图中是什么物体？"

# 临时证据图文件
TEMP_EVIDENCE_PATH = "./evidence_crop.png"

# 阈值与重试
CONFIDENCE_THRESHOLD = 0.2
MAX_RETRIES = 2


# ==============================================

def image_to_base64(image_path, max_size=(512, 512)):
    """将图像转为 Base64（用于 Qwen-VL）"""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail(max_size)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def call_qwen(prompt, image_path=None):
    """调用 Qwen-VL 服务（支持多模态）"""
    payload = {"prompt": prompt}
    if image_path and os.path.exists(image_path):
        payload["image_url"] = image_to_base64(image_path)
    try:
        response = requests.post(QWEN_URL, json=payload, timeout=120)
        print(f"\n[发起Qwen请求 Prompt {prompt}...")
        if response.status_code == 200:
            res = response.json()
            answer = res.get("response", "").strip()
            print(f"🤖 Qwen 完整响应: {json.dumps(res, ensure_ascii=False, indent=2)}")
            return answer
        else:
            print(f"❌ Qwen 请求失败: {response.status_code} - {response.text}")
            return ""
    except Exception as e:
        print(f"💥 Qwen 异常: {e}")
        return ""


def call_sam(original_img_path, bbox_str):
    """调用 SAM 服务，返回裁剪后的 PNG bytes"""
    try:
        with open(original_img_path, 'rb') as f:
            files = {'image': f}
            data = {'bbox': bbox_str}
            response = requests.post(SAM_URL, files=files, data=data, timeout=30)
            print(f"\n[🔍 SAM 请求] BBox坐标: {bbox_str}")
            if response.status_code == 200:
                with open(TEMP_EVIDENCE_PATH, "wb") as out:
                    out.write(response.content)
                print(f"✂️ 证据图已保存: {TEMP_EVIDENCE_PATH} (大小: {len(response.content)} bytes)")
                return True
            else:
                print(f"❌ SAM 失败: HTTP {response.status_code} - {response.text}")
                return False
    except Exception as e:
        print(f"💥 SAM 异常: {e}")
        return False


def call_clip(image_bytes, text_label):
    """调用 CLIP，返回 similarity 分数，并打印完整结果"""
    files = {'imagefile': ('evidence.png', image_bytes, 'image/png')}
    data = {'text': text_label}
    try:
        print(f"\n[🔍 CLIP 请求] Text label(s): '{text_label}'")
        response = requests.post(CLIP_URL, files=files, data=data, timeout=10)
        if response.status_code == 200:
            res = response.json()
            print(f"📊 CLIP 完整响应:\n{json.dumps(res, ensure_ascii=False, indent=2)}")

            # 尝试匹配目标标签
            for label, val in res.get('results', {}).items():
                if text_label.strip().lower() in label.lower():
                    score = float(val['similarity'])
                    return score

            # 未匹配则取第一个
            if res.get('results'):
                first_key = list(res['results'].keys())[0]
                score = float(res['results'][first_key]['similarity'])
                print(f"⚠️ 但是未找到精确匹配，使用首个标签 '{first_key}': similarity={score:.4f}")
                return score
            else:
                print("❌ CLIP 返回结果为空")
                return 0.0
        else:
            print(f"❌ CLIP 失败: {response.status_code} - {response.text}")
            return 0.0
    except Exception as e:
        print(f"💥 CLIP 异常: {e}")
        return 0.0


# 匹配提取坐标，作为变量返回
import re


def extract_bbox_from_text(text, img_w=1000, img_h=1000):
    print(f"🔍 坐标提取，输入 text = {repr(text)}")

    # 更健壮的正则表达式，匹配各种格式的坐标
    patterns = [
        r'\((\d+)\s*[,，]\s*(\d+)\)\s*\((\d+)\s*[,，]\s*(\d+)\)',  # (100,100) (600,600)
        r'(\d+)\s*[,，]\s*(\d+)\s+(\d+)\s*[,，]\s*(\d+)',  # 100,100 600,600
        r'(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)',  # 100,100,600,600
        r'坐标[：:]?\s*\(?(\d+)\s*[,，]\s*(\d+)\)?\s*\(?(\d+)\s*[,，]\s*(\d+)\)?',  # 坐标: (100,100)(600,600)
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            x1, y1, x2, y2 = map(int, match.groups())
            # 坐标范围限制
            x1 = max(0, min(img_w, x1))
            y1 = max(0, min(img_h, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))
            if x1 > x2: x1, x2 = x2, x1
            if y1 > y2: y1, y2 = y2, y1
            print(f"✅ 匹配成功: ({x1},{y1}) ({x2},{y2})")
            return f"{x1},{y1},{x2},{y2}"

    print("❌ 未能匹配坐标格式，使用全图")
    return f"0,0,{img_w},{img_h}"


def main():
    print("=" * 60)
    print("🚀 启动「可验证视觉问答」闭环系统")
    print(f"📸 图像: {os.path.basename(INPUT_IMAGE_PATH)}")
    print(f"❓ 问题: {QUESTION}")
    print("=" * 60)

    # === Step 1: Qwen 初步回答 + 证据位置描述 ===
    prompt1 = f"问题：{QUESTION} 请先给出答案；再以格式(左上角x坐标,左上角y坐标) (右下角x坐标,右下角y坐标) 两点生成的矩形框将图片需要关注区域包围进去(输出不换行)。"
    initial_response = call_qwen(prompt1, INPUT_IMAGE_PATH)
    if not initial_response:
        return

    # 分割答案和坐标字符串
    parts = [p for p in initial_response.split("。") if p.strip()]
    if len(parts) >= 3 and parts[-2].startswith('(') and parts[-1].startswith('('):
        initial_answer = " ".join(parts[:-2])
    else:
        initial_answer = initial_response

    # 获取图像真实尺寸（用于坐标校验）
    with Image.open(INPUT_IMAGE_PATH) as img:
        img_w, img_h = img.size

    bbox_str = extract_bbox_from_text(initial_response, img_w, img_h)  # 提取坐标

    # === Step 2 & 3 & 4: SAM 抠图 → Qwen 重答 → CLIP 验证（带重试）===
    final_answer = initial_answer
    confidence = 0.0
    retry = 0

    while retry <= MAX_RETRIES:
        print(f"\n🔄 第 {retry + 1} 次验证循环...")

        # SAM
        if not call_sam(INPUT_IMAGE_PATH, bbox_str):
            break

        # 在这里读取的是经过SAM分割得到的证据图
        with open(TEMP_EVIDENCE_PATH, "rb") as f:
            evidence_bytes = f.read()

        prompt2 = f"只看这张图，回答：{QUESTION}"  # 输入给Qwen模型的问题
        refined_answer = call_qwen(prompt2, TEMP_EVIDENCE_PATH)
        if not refined_answer:
            break

        # CLIP
        verification_text = f"{refined_answer}"

        confidence = call_clip(evidence_bytes, verification_text)

        if confidence >= CONFIDENCE_THRESHOLD:  #和阈值判断一下
            final_answer = refined_answer
            print(f"\n✅ 验证通过！最终答案: {final_answer} (相似度: {confidence:.3f})")
            break
        else:
            print(f"⚠️ 相似度不足 ({confidence:.3f} < {CONFIDENCE_THRESHOLD})")
            if retry == 0:
                print(" → 尝试扩大到全图...")
                bbox_str = f"0,0,{img_w},{img_h}"
            retry += 1
        time.sleep(1)

    # 清理临时文件
    # if os.path.exists(TEMP_EVIDENCE_PATH):
    #     os.remove(TEMP_EVIDENCE_PATH)

    # === 最终输出 ===
    print("\n" + "=" * 60)
    print("🎯 最终结果:")
    print(f"问题: {QUESTION}")
    print(f"答案: {final_answer}")
    print(f"CLIP 置信度 (similarity): {confidence:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
