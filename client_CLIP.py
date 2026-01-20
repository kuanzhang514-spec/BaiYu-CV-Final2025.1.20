import requests
import os
import json


def test_clip_server():
    # --- 配置区域 ---
    SERVER_IP = "这是我的服务器IP地址，我隐藏了"
    PORT = "8021"
    url = f"http://{SERVER_IP}:{PORT}/clip/score"

    # 替换为你 Windows 本地的图片路径
    image_path = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\catsleep.jpg"
    # 待验证的文本
    test_text = "a photo of cat,a photo of dog,a photo of pig"

    if not os.path.exists(image_path):
        print(f"❌ 找不到本地文件: {image_path}")
        return

    print(f"📡 正在连接服务器: {SERVER_IP}...")

    # 准备文件和数据
    files = {
        'imagefile': (os.path.basename(image_path), open(image_path, 'rb'), 'image/jpeg')
    }
    data = {
        'text': test_text
    }

    try:
        # 发送请求
        response = requests.post(url, files=files, data=data, timeout=10)

        # 检查响应
        # Windows 脚本打印部分
        if response.status_code == 200:
            res = response.json()
            print(f"\n🏆 最佳匹配: {res['best_match']}")
            for label, val in res['results'].items():
                print(f"  - {label}: 相似度={val['similarity']}")

        else:
            print(f"❌ 服务器返回错误: {response.status_code}")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器。请检查：")
        print(f"1. 服务器防火墙是否开放了 {PORT} 端口？")
        print(f"2. 服务器上的 uvicorn 是否正在运行？")
        print(f"3. 你的电脑和服务器是否在同一局域网内？")
    except Exception as e:
        print(f"❌ 发生意外错误: {e}")


if __name__ == "__main__":
    test_clip_server()
