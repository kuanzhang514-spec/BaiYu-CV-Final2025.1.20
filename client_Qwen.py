import requests
import base64
import os
from PIL import Image
from io import BytesIO

def get_processed_image_base64(image_path, max_size=(512, 512)):
    """处理图片并转为 Base64"""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.thumbnail(max_size)
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def call_qwen(prompt, image_path=None, server_ip="这是我的服务器IP地址，我隐藏了"):
    """
    通用请求函数：
    - 如果 image_path 为 None，自动切换为单模态（文字）
    - 如果 image_path 有值，自动切换为多模态（文字+图片）
    """
    url = f"http://{server_ip}:8020/chat_vl"
    payload = {"prompt": prompt, "image_url": ""}
    
    # 逻辑判断：是否开启多模态
    if image_path and os.path.exists(image_path):
        print(f"📸 [多模态模式] 正在处理图片: {os.path.basename(image_path)}")
        payload["image_url"] = get_processed_image_base64(image_path)
    else:
        print(f"📝 [单模态模式] 纯文字发送")

    try:
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            res = response.json()
            print("\n🤖 Qwen2-VL 回答：")
            print("-" * 40)
            print(res.get("response"))
            print("-" * 40)
        else:
            print(f"❌ 请求失败: {response.status_code}")
    except Exception as e:
        print(f"💥 错误: {e}")

if __name__ == "__main__":
    SERVER_IP = "192.168.10.115"
    IMG_PATH = r'C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\catsleep.jpg'  # 猫
    # IMG_PATH = r'C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\hongyu.jpg'  # 红鱼
    # IMG_PATH = r'C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\dihuangkaijia.jpg'  # 帝皇铠甲

    # 使用示例：

    # 1. 单模态：直接不传 image_path 参数
    # call_qwen("背诵李白古诗", server_ip=SERVER_IP)

    # 2. 多模态：传入图片路径
    call_qwen("请给我图片中动物的坐标位置，左上角右下角就好", image_path=IMG_PATH, server_ip=SERVER_IP)
    
# 交互式对话框一样输入
# if __name__ == "__main__":
#     print("🌟 Qwen2-VL 终端交互已启动 (输入 'quit' 退出)")
#     while True:
#         user_input = input("\n请输入问题: ")
#         if user_input.lower() == 'quit': break
        
#         has_img = input("是否附加图片？(y/n): ")
#         if has_img.lower() == 'y':
#             path = r'C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\catsleep.jpg' # 或者让用户输入路径
#             call_qwen(user_input, image_path=path)
#         else:
#             call_qwen(user_input)