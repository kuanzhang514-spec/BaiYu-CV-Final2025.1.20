# 我自己的SAM2
# sam2_client.py
import requests

# === 配置区（按你的实际情况修改）===
SERVER_URL = "http://这是我的服务器IP地址，我隐藏了:8022/segment_by_bbox"  # 你的服务器地址
IMAGE_PATH = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\shanzhu.png"               # ← 改成你的本地图片路径
BBOX = [0, 0, 1000, 1000]                                # ← 改成你要分割的区域 [x1, y1, x2, y2]
OUTPUT_PATH = "./segmented_output.png"                     # 输出结果保存路径
# ===================================

def main():
    print(f"📤 正在向 {SERVER_URL} 发送请求...")
    print(f"🖼️  图片: {IMAGE_PATH}")
    print(f"📦 BBox: {BBOX}")

    try:
        with open(IMAGE_PATH, "rb") as f:
            files = {"image": f}
            data = {"bbox": ",".join(map(str, BBOX))}  # 转成 "100,100,400,400"
            response = requests.post(SERVER_URL, files=files, data=data, timeout=30)

        if response.status_code == 200:
            with open(OUTPUT_PATH, "wb") as out_file:
                out_file.write(response.content)
            print(f"✅ 成功！分割结果已保存到: {OUTPUT_PATH}")
        else:
            print(f"❌ 请求失败 (HTTP {response.status_code}): {response.text}")

    except FileNotFoundError:
        print(f"❌ 图片未找到: {IMAGE_PATH}")
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器，请检查：")
        print("   - 服务器是否正在运行？")
        print("   - IP 和端口是否正确？")
        print("   - 防火墙是否放行 8022 端口？")
    except Exception as e:
        print(f"💥 发生未知错误: {e}")

if __name__ == "__main__":
    main()