'''
测试一下Qwen能不能用

'''
import os
import json
import requests
from PIL import Image
import base64
from io import BytesIO


def test_basic_functionality():
    """测试基础功能"""

    # 配置
    SERVER_IP = "这是我的服务器IP地址，我隐藏了"
    QWEN_URL = f"http://{SERVER_IP}:8020/chat_vl"
    DATA_ROOT = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\TextVQA"
    IMAGE_DIR = os.path.join(DATA_ROOT, "images")
    METADATA_PATH = os.path.join(DATA_ROOT, "metadata.json")

    # 加载一个样本
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    sample = data[0]  # 第一个样本
    image_path = os.path.join(IMAGE_DIR, sample['image_file'])

    print(f"📸 测试图像: {image_path}")
    print(f"❓ 问题: {sample['question']}")

    # 检查图像是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图像不存在: {image_path}")
        return

    # 图像转Base64
    try:
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail((512, 512))
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"❌ 图像处理失败: {e}")
        return

    # 调用Qwen
    prompt = f"问题：{sample['question']} 请直接回答。"
    payload = {
        "prompt": prompt,
        "image_url": img_base64
    }

    print(f"📤 发送提示: {prompt}")

    try:
        response = requests.post(QWEN_URL, json=payload, timeout=30)
        print(f"📡 响应状态: {response.status_code}")

        if response.status_code == 200:
            res = response.json()
            print(f"✅ Qwen回答: {res.get('response', '')}")
        else:
            print(f"❌ 请求失败: {response.text}")
    except Exception as e:
        print(f"💥 请求异常: {e}")


if __name__ == "__main__":
    test_basic_functionality()