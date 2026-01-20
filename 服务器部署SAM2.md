# 3号机部署SAM2

## ==主线1：==

==部署流程：==

```bah
cd /data/xulab/
mkdir sam_service
cd sam_service

# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装基础依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install fastapi uvicorn python-multipart opencv-python pillow matplotlib
pip install wheel
pip install --upgrade setuptools
```

下载 SAM 2 模型、权重文件、yaml文件

```ba
# 1.手动下载SAM2模型，sam2-main.zip ，放到/data/xulab/sam_service/
https://github.com/facebookresearch/sam2 # 下载链接
unzip sam2-main.zip  #解压
pip install --no-build-isolation -e ./sam2-main  #安装

# 2.下载yaml文件，放在configs目录下
mkdir -p configs
 wget https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2.1/sam2.1_hiera_t.yaml -O configs/sam2.1_hiera_t.yaml

# 3.手动下载 .pt文件再拖到目录下
https://huggingface.co/facebook/sam2.1-hiera-tiny/blob/main/sam2.1_hiera_tiny.pt
```

server_sam2.py 服务端代码:

```bash
# /data/xulab/sam_service/server_sam2.py
import torch
import numpy as np
import cv2
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import uvicorn
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

app = FastAPI(title="SAM 2 Segmentation Service", description="为可验证视觉问答提供证据抠图")

# --- 配置路径 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "sam2.1_hiera_tiny.pt")
CONFIG_NAME = "sam2.1_hiera_t"  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 打印配置文件和权重文件路径，用于调试
print(f"🔧 检查权重文件: {os.path.exists(CHECKPOINT_PATH)} | 路径: {CHECKPOINT_PATH}")

# --- 加载模型 ---
print(f"🔧 正在加载 SAM 2 (tiny) 到 {DEVICE}...")

try:
    import sam2
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # ✅ 正确获取 sam2 包路径
    sam2_package_path = sam2.__path__[0]
    sam2_config_dir = os.path.join(sam2_package_path, "configs", "sam2.1")

    print(f"📁 SAM2 config 目录: {sam2_config_dir}")
    assert os.path.isdir(sam2_config_dir), f"Config 目录不存在！请检查安装。"

    # 清理可能的重复初始化
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # 初始化 Hydra
    initialize_config_dir(config_dir=sam2_config_dir, version_base=None)

    # 加载模型
    sam2_model = build_sam2(CONFIG_NAME, CHECKPOINT_PATH, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)
    print("✅ SAM 2 模型加载成功！服务就绪。")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    raise

@app.post("/segment_by_bbox", summary="根据 BBox 抠图")
async def segment_by_bbox(
    image: UploadFile = File(..., description="原始图像"),
    bbox: str = Form(..., description='目标区域坐标，格式: "x1,y1,x2,y2"')
):
    """
    输入一张图 + 一个 bbox，返回该区域内分割出的对象（PNG 透明背景）
    用于 VQA 闭环中的「证据提取」步骤
    """
    try:
        # 1. 读取图像
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return {"error": "无效图像文件"}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # 2. 解析 bbox
        coords = [float(x.strip()) for x in bbox.split(",")]
        if len(coords) != 4:
            return {"error": "bbox 必须是 x1,y1,x2,y2"}
        x1, y1, x2, y2 = coords

        # 3. 设置图像 & 预测
        predictor.set_image(img_rgb)
        masks, scores, _ = predictor.predict(
            box=np.array([x1, y1, x2, y2], dtype=np.float32)[None, :],
            multimask_output=False
        )
        mask = masks[0].squeeze().astype(bool)

        # 4. 构建 RGBA 图像
        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[:, :, :3][mask] = img_rgb[mask]
        rgba[mask, 3] = 255  # 不透明

        # 5. 裁剪（安全边界）
        x1_c, y1_c = max(0, int(x1)), max(0, int(y1))
        x2_c, y2_c = min(W, int(x2)), min(H, int(y2))
        cropped = rgba[y1_c:y2_c, x1_c:x2_c]

        # 6. 编码为 PNG
        bgra = cv2.cvtColor(cropped, cv2.COLOR_RGBA2BGRA)
        success, buffer = cv2.imencode(".png", bgra)
        if not success:
            return {"error": "图像编码失败"}

        print(f"🖼️ 抠图成功 | 尺寸: {cropped.shape[:2]} | Mask 像素数: {mask.sum()}")

        return Response(content=buffer.tobytes(), media_type="image/png")

    except Exception as e:
        print(f"❌ 处理错误: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8022)
```

请求端：

```bash
# sam2_client.py
import requests

# === 配置区===
SERVER_URL = "http://这是我的服务器IP地址：端口号，这里我隐藏了/segment_by_bbox"  # 你的服务器地址
IMAGE_PATH = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\pingguo.jpg"               # ← 改成你的本地图片路径
BBOX = [100, 100, 1000, 1000]                              
OUTPUT_PATH = "./segmented_output.png"                     #结果保存路径
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
```

