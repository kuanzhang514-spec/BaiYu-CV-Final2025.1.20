# 尧的SAM
import requests
import os

# 服务器 IP 地址
SERVER_IP = "这是我的服务器IP地址，我隐藏了" 
URL = f"http://{SERVER_IP}:8022/sam/segment"

# 本地图片路径
LOCAL_IMG = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\pingguo.jpg"
SAVE_PATH = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\result.png"

# 定义你想抠图的框 [左上x, 左上y, 右下x, 右下y]
# 注意：这必须是像素坐标
payload = {
    "bbox": "0,0,1000,1100" 
}

files = [
    ('imagefile', ('image.jpg', open(LOCAL_IMG, 'rb'), 'image/jpeg'))
]

print("📡 正在发送请求到 SAM 服务器...")
response = requests.post(URL, data=payload, files=files)

if response.status_code == 200:
    with open(SAVE_PATH, "wb") as f:
        f.write(response.content)
    print(f"✅ 抠图成功！已保存至: {SAVE_PATH}")
else:
    print(f"❌ 请求失败: {response.text}")