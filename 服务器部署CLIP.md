# 3号机器上部署CLIP

现在部署好了：

demo1.py 这个是restful封装的CLIP推理模型

test_clip_win.py 这个是vscode访问脚本举例

==启动使用：==

```ba
# 服务端
cd /data/xulab/clip_service
source venv/bin/activate
python demo1.py

#请求端
python test_clip_win.py
```



==这是部署过程：==

```bash
# 进入目标目录
cd /data/xulab/
mkdir clip_service
cd clip_service

# 创建虚拟环境 (推荐使用 venv 或 conda)
python3 -m venv venv
source venv/bin/activate

# 安装必要依赖
# 注意：Tesla P40 是 Pascal 架构，建议安装与 CUDA 12.4 匹配的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 1. 确保安装了 git（如果没有的话）
sudo apt-get update && sudo apt-get install git -y
# 2. 安装 CLIP 及其核心依赖
pip install git+https://github.com/openai/CLIP.git
# 3. 安装之前报错的其他 Web 依赖
pip install fastapi uvicorn python-multipart
```



/data/xulab/clip_service 下编写 demo1.py 脚本加载模型并启动 FastAPI 服务

```py
import torch
import clip
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import io
import uvicorn
import torch.nn.functional as F  # 导入 F 以使用 softmax

app = FastAPI(title="CLIP Consistency Scoring Service")

# 1. 硬件检测与模型加载
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在加载模型至设备: {device}...")

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

@app.post("/clip/score")
async def get_clip_score(
    imagefile: UploadFile = File(...),
    text: str = Form(...) # 接收 "cat,dog,pig"
):
    try:
        image_data = await imagefile.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # 将字符串按逗号切分成列表
        labels = [t.strip() for t in text.split(",")]
        text_input = clip.tokenize(labels).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_input)

            # 归一化特征
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # --- 核心修改部分：计算相似度和概率 ---
            
            # 1. 计算原始余弦相似度 (Cosine Similarity)
            similarities = (image_features @ text_features.T).cpu().numpy()[0]
            
            # 2. 计算 Softmax 概率 (Probabilities)
            # 乘以 100 是 CLIP 官方推荐的缩放因子，用于放大分差使 Softmax 效果更明显
            logits_per_image = (image_features @ text_features.T) * 100
            probs = F.softmax(logits_per_image, dim=-1).cpu().numpy()[0]
            
            # -----------------------------------

        # 封装结果
        score_details = {}
        for i, label in enumerate(labels):
            score_details[label] = {
                "similarity": round(float(similarities[i]), 4),
                "probability": round(float(probs[i]), 4)
            }
        
        # 找出概率最大的索引
        best_idx = probs.argmax()

        return {
            "status": "success",
            "results": score_details,
            "best_match": labels[best_idx],
            "max_probability": round(float(probs[best_idx]), 4),
            "max_similarity": round(float(similarities[best_idx]), 4)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```



本地请求端的 python 脚本:

```pyh
import requests
import os
import json

def test_clip_server():
    # --- 配置区域 ---
    SERVER_IP = "这是我的服务器IP地址，这里我隐藏了"
    PORT = "8000"
    url = f"http://{SERVER_IP}:{PORT}/clip/score"
    
    # 替换为你 Windows 本地的图片路径
    # image_path = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\catsleep.jpg" 
    image_path = r"C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\Uploadimg\pingguo.jpg" 
    # 待验证的文本
    # test_text = "a photo of cat,a photo of dog,a photo of pig"
    test_text = "a photo of cat,a photo of dog,a photo of apple"

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
```



服务端输出：

```bash
python demo1.py 
正在加载模型至设备: cuda...
INFO:     Started server process [1356153]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     192.168.10.102:52493 - "POST /clip/score HTTP/1.1" 200 OK
INFO:     192.168.10.102:51205 - "POST /clip/score HTTP/1.1" 200 OK
```

请求端输出:

```bas
(qwen3_local) C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji>python test_clip_win.py
📡 正在连接服务器: 192.168.10.115...

🏆 最佳匹配: a photo of apple
  - a photo of cat: 相似度=0.1936
  - a photo of dog: 相似度=0.205
  - a photo of apple: 相似度=0.3042
```

