# 服务器搭建Qwen

 ==2026.1.7 多模态模型部署完成，文件说明 **必读必读必读**：==

(1) api_server_vl.py 这里面是完成的下载模型+启动模型封装成API接口的代码

**(2) demo2.py 这里是多模态的API ，启动这个可以实现多模态问答**

(3) demo11.py 这里存放的是仅下载模型的代码，从api_server_vl.py 的代码中抽取出来的（直接执行api_server_vl.py就行了，功能覆盖了）

(4) qwen_vl_env 这是包环境存放目录

(5) ==怎么使用这个API接口？==首先启动demo2.py，并且 client_test.py 文件里面存放了示例代码，用 vscode 编辑器执行client_test.py 代码可以灵活实现 单/多模态问答(文件注释里写了具体怎么使用)

**激活环境：source qwen_vl_env/bin/activate         **

**退出环境：deactivate**





# **开始部署：**

```bash
# 进入dada目录 使用 sudo 创建目录
sudo mkdir -p qwen-project
cd qwen-project
ll
```

```bash
# 创建环境
python3 -m venv qwen_vl_env
source qwen_vl_env/bin/activate

# 
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.47.0 accelerate modelscope fastapi uvicorn qwen_vl_utils
pip install "numpy<2"
pip install modelscope    #使用国内镜像源
```

## 1.下载模型的脚本

```bash
# 脚本
touch api_server_vl.py
nano api_server_vl.py
```

粘贴代码到 api_server_vl.py：

```python
cat ./api_server_vl.py
import os
os.environ['MODELSCOPE_CACHE'] = '/qwen-project/model_cache'

import torch
import base64
from modelscope import snapshot_download
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# 自动下载/加载模型
model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, 
    torch_dtype=torch.float32, 
    device_map="auto", 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

@app.post("/chat_vl")
async def chat_vl(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        image_url = data.get("image_url", None)

        # --- 核心逻辑：判断输入模式 ---
        if image_url and len(image_url) > 0:
            print("📸 模式：多模态问答")
            # 格式化 Base64
            if not (image_url.startswith("http") or image_url.startswith("/") or image_url.startswith("data:image")):
                image_url = f"data:image/jpeg;base64,{image_url}"
            
            content = [
                {"type": "image", "image": image_url},
                {"type": "text", "text": prompt},
            ]
        else:
            print("📝 模式：纯文字问答")
            content = [
                {"type": "text", "text": prompt},
            ]

        messages = [{"role": "user", "content": content}]

        # --- 关键：准备推理数据 ---
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        print("🧠 正在推理...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=2048,   # 允许生成的最大长度
                do_sample=True,         # 开启采样模式
                temperature=0.7,        # 随机度
                top_p=0.9,              # 核心采样
                repetition_penalty=1.1  # 防止重复
            )
        
        # 处理输出结果
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]

        print(f"✅ 推理完成，回复长度: {len(output_text)}")
        return {"response": output_text}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"response": f"Error: {str(e)}"}

if __name__ == "__main__":
    print("🚀 API 服务正在启动，监听端口 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```



```bash
python api_server_vl.py   #关闭梯子执行
```

**下载的模型存到 `qwen-project/model_cache` 下了**

显示1：

```bash
python api_server_vl.py
正在检查/下载 Qwen2-VL 模型 (来自 ModelScope 国内镜像)...
Downloading Model from https://www.modelscope.cn to directory: /qwen-project/model_cache/models/qwen/Qwen2-VL-7B-Instruct
2026-01-06 22:54:00,315 - modelscope - INFO - Got 17 files, start to download ...
Downloading [LICENSE]: 100%|███████████████████████████████████████████████████████████████████████████████████████████| 11.1k/11.1k [00:05<00:00, 1.91kB/s]
Downloading [configuration.json]: 100%|███████████████████████████████████████████████████████████████████████████████████| 76.0/76.0 [00:05<00:00, 12.8B/s]
Downloading [chat_template.json]: 100%|██████████████████████████████████████████████████████████████████████████████████| 1.03k/1.03k [00:05<00:00, 176B/s]
Downloading [generation_config.json]: 100%|█████████████████████████████████████████████████████████████████████████████████| 244/244 [00:05<00:00, 40.9B/s]
Downloading [config.json]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1.17k/1.17k [00:05<00:00, 200B/s]
Downloading [merges.txt]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 1.59M/1.59M [00:06<00:00, 264kB/s]
Downloading [preprocessor_config.json]: 100%|████████████████████████████████████████████████████████████████████████████████| 347/347 [00:00<00:00, 351B/s]
Downloading [model.safetensors.index.json]: 100%|██████████████████████████████████████████████████████████████████████| 55.1k/55.1k [00:01<00:00, 55.1kB/s]
Downloading [README.md]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 17.3k/17.3k [00:00<00:00, 21.3kB/s]
Downloading [tokenizer_config.json]: 100%|█████████████████████████████████████████████████████████████████████████████| 4.09k/4.09k [00:01<00:00, 3.85kB/s]
Downloading [vocab.json]: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 2.65M/2.65M [00:03<00:00, 834kB/s]
Downloading [tokenizer.json]: 100%|████████████████████████████████████████████████████████████████████████████████████| 6.70M/6.70M [00:05<00:00, 1.24MB/s]
Downloading [model-00005-of-00005.safetensors]: 100%|██████████████████████████████████████████████████████████████████| 1.02G/1.02G [07:27<00:00, 2.44MB/s]
Downloading [model-00003-of-00005.safetensors]: 100%|██████████████████████████████████████████████████████████████████| 3.60G/3.60G [21:39<00:00, 2.97MB/s]
Downloading [model-00004-of-00005.safetensors]: 100%|██████████████████████████████████████████████████████████████████| 3.60G/3.60G [21:46<00:00, 2.96MB/s]
Downloading [model-00002-of-00005.safetensors]: 100%|██████████████████████████████████████████████████████████████████| 3.60G/3.60G [22:30<00:00, 2.86MB/s]
Downloading [model-00001-of-00005.safetensors]: 100%|██████████████████████████████████████████████████████████████████| 3.63G/3.63G [23:35<00:00, 2.75MB/s]
Processing 17 items: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 17.0/17.0 [23:35<00:00, 83.2s/it]
2026-01-06 23:17:35,474 - modelscope - INFO - Download model 'qwen/Qwen2-VL-7B-Instruct' successfully.█████████████████| 3.60G/3.60G [21:39<00:00, 3.36MB/s]
✅ 模型准备就绪，路径: /qwen-project/model_cache/models/qwen/Qwen2-VL-7B-Instruct                            | 908M/3.63G [07:33<25:52, 1.90MB/s]
🚀 正在加载模型到显存 (FP32 模式以确保 P40 兼容性)...██████████████████████████████████████████████████▌               | 2.78G/3.63G [21:44<06:23, 2.39MB/s]
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.▉               | 2.80G/3.63G [21:52<04:37, 3.20MB/s]
`Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the `config` argument. All other arguments will be removed in v4.46nloading [model-00001-of-00005.safetensors]: 100%|█████████████████████████████████████████████████████████████████▉| 3.63G/3.63G [23:35<00:00, 10.8MB/s]
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:15<00:00,  3.06s/it]
✅ 服务即将启动，端口: 8000
INFO:     Started server process [1544677]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

显示2：

```bash
nvidia-smi
Tue Jan  6 23:19:45 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla P40                      Off |   00000000:17:00.0 Off |                    0 |
| N/A   29C    P0             47W /  250W |    6731MiB /  23040MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla P40                      Off |   00000000:18:00.0 Off |                    0 |
| N/A   27C    P0             49W /  250W |    9155MiB /  23040MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   2  Tesla P40                      Off |   00000000:65:00.0 Off |                    0 |
| N/A   29C    P0             49W /  250W |    9155MiB /  23040MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   3  Tesla P40                      Off |   00000000:B3:00.0 Off |                    0 |
| N/A   28C    P0             48W /  250W |    7635MiB /  23040MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1250      G   /usr/lib/xorg/Xorg                              4MiB |
|    0   N/A  N/A   1544677      C   python                                       6724MiB |
|    1   N/A  N/A      1250      G   /usr/lib/xorg/Xorg                              4MiB |
|    1   N/A  N/A   1544677      C   python                                       9148MiB |
|    2   N/A  N/A      1250      G   /usr/lib/xorg/Xorg                              4MiB |
|    2   N/A  N/A   1544677      C   python                                       9148MiB |
|    3   N/A  N/A      1250      G   /usr/lib/xorg/Xorg                              4MiB |
|    3   N/A  N/A   1544677      C   python                                       7628MiB |
+-----------------------------------------------------------------------------------------+


解释多卡显存均匀分布：
你看 nvidia-smi 中的 Memory-Usage，4 张显卡分别占用了约 6.7GB, 9.1GB, 9.1GB, 7.6GB。
这说明 device_map="auto" 完美生效了，它把 7B 模型（FP32 精度）平摊到了你所有的显卡上，每张卡都压力不大，运行会非常稳定。
```

仅显示启动成功，没有测试过能不能用。



## 2.多模态模型

### api端 demo2.py

```python
import os
os.environ['MODELSCOPE_CACHE'] = '/qwen-project/model_cache'

import torch
import base64
from modelscope import snapshot_download
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

# 自动下载/加载模型
model_dir = snapshot_download("qwen/Qwen2-VL-7B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, 
    torch_dtype=torch.float32, 
    device_map="auto", 
    trust_remote_code=True
)

# 限制最大像素为 512x512 左右的大小（262144像素）
# 这将极大地减少显存占用，同时不影响识别效果
min_pixels = 256 * 28 * 28
max_pixels = 512 * 28 * 28 
processor = AutoProcessor.from_pretrained(
    model_dir, 
    trust_remote_code=True, 
    min_pixels=min_pixels, 
    max_pixels=max_pixels
)

@app.post("/chat_vl")
async def chat_vl(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        image_url = data.get("image_url", "")

        # 构造 content
        if image_url:
            # 如果是 Base64 且缺前缀，补全它
            if not (image_url.startswith("http") or image_url.startswith("data:image")):
                image_url = f"data:image/jpeg;base64,{image_url}"
            content = [
                {"type": "image", "image": image_url},
                {"type": "text", "text": prompt}
            ]
            print("📸 正在处理多模态任务...")
        else:
            content = [{"type": "text", "text": prompt}]
            print("📝 正在处理纯文本任务...")

        messages = [{"role": "user", "content": content}]

        # 推理预处理 (注意：这里千万别 print messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to("cuda")

        # --- 生成设置 ---
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=2048, 
                do_sample=True, 
                temperature=0.7
            )
        
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        return {"response": output_text}

    except Exception as e:
        return {"response": f"Error: {str(e)}"}

if __name__ == "__main__":
    print("🚀 API 服务正在启动，监听端口 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### vscode 客户端 client_test.py：

```python
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

def call_qwen(prompt, image_path=None, server_ip="192.168.10.115"):
    """
    通用请求函数：
    - 如果 image_path 为 None，自动切换为单模态（文字）
    - 如果 image_path 有值，自动切换为多模态（文字+图片）
    """
    url = f"http://{server_ip}:8000/chat_vl"
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
    SERVER_IP = "我的服务器IP地址，这里我隐藏了"
    # IMG_PATH = r'C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\catsleep.jpg'  # 猫
    # IMG_PATH = r'C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\hongyu.jpg'  # 红鱼
    IMG_PATH = r'C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji\dihuangkaijia.jpg'  # 帝皇铠甲

    # 使用示例：

    # 1. 单模态：直接不传 image_path 参数
    # call_qwen("背诵李白古诗", server_ip=SERVER_IP)

    # 2. 多模态：传入图片路径
    call_qwen("描述一下这张图片", image_path=IMG_PATH, server_ip=SERVER_IP)
    
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
```

### 问答演示

```bash
python ./demo2.py
Downloading Model from https://www.modelscope.cn to directory: /qwen-project/model_cache/models/qwen/Qwen2-VL-7B-Instruct
The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is ignored.
`Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the `config` argument. All other arguments will be removed in v4.46
Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:15<00:00,  3.18s/it]
🚀 API 服务正在启动，监听端口 8000...
INFO:     Started server process [1567398]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
📸 正在处理多模态任务...
INFO:     192.168.10.120:54753 - "POST /chat_vl HTTP/1.1" 200 OK
📸 正在处理多模态任务...
INFO:     192.168.10.120:63754 - "POST /chat_vl HTTP/1.1" 200 OK
📸 正在处理多模态任务...
INFO:     192.168.10.120:49689 - "POST /chat_vl HTTP/1.1" 200 OK
📝 正在处理纯文本任务...
INFO:     192.168.10.120:51198 - "POST /chat_vl HTTP/1.1" 200 OK

```

```bash
(qwen3_local) C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji>python client_test.py
📝 [单模态模式] 纯文字发送

🤖 Qwen2-VL 回答：
----------------------------------------
好的，以下是李白的几首古诗：

1. 《静夜思》
床前明月光，疑是地上霜。
举头望明月，低头思故乡。

2. 《望庐山瀑布》
日照香炉生紫烟，遥看瀑布挂前川。
飞流直下三千尺，疑是银河落九天。

3. 《早发白帝城》
朝辞白帝彩云间，千里江陵一日还。
两岸猿声啼不住，轻舟已过万重山。

4. 《将进酒》
君不见黄河之水天上来，奔流到海不复回。
君不见高堂明镜悲白发，朝如青丝暮成雪。
人生得意须尽欢，莫使金樽空对月。
天生我材必有用，千金散尽还复来。
烹羊宰牛且为乐，会须一饮三百杯。

5. 《夜泊牛渚怀古》
牛渚西江夜，青天无片云。
登舟望秋月，空忆谢将军。
余亦能高咏，斯人不可闻。
明朝挂帆席，枫叶落纷纷。

希望这些古诗能帮助你背诵。
----------------------------------------

(qwen3_local) C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji>python client_test.py
📸 [多模态模式] 正在处理图片: hongyu.jpg

🤖 Qwen2-VL 回答：
----------------------------------------
这张图片展示了一群红色的鱼在水中游动。这些鱼看起来非常鲜艳，身体呈现出明亮的红色，鳍和尾巴也是红色的。鱼群紧密地聚集在一起，似乎在寻找食物或相互互动。背景是浅蓝色的，可能是水族箱的 背景。整体画面给人一种生动和充满活力的感觉。
----------------------------------------

(qwen3_local) C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji>python client_test.py
📸 [多模态模式] 正在处理图片: catsleep.jpg

🤖 Qwen2-VL 回答：
----------------------------------------
这张图片展示了一只橙色的猫咪，它正躺在一个蓝色的垫子上，身体放松，似乎在休息。猫咪的头靠在垫子上，尾巴自然地垂在垫子的一侧。猫咪的旁边有一个白色的碗，碗里装满了猫粮。背景中可以看到 一些绿色的植物，阳光透过树叶洒在猫咪和垫子上，营造出一种温暖和宁静的氛围。图片的左下角有“vivo X90 ZEISS”的字样，表明这张照片可能是用vivo X90手机拍摄的，并且使用了蔡司镜头。
----------------------------------------

(qwen3_local) C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji>python client_test.py
📸 [多模态模式] 正在处理图片: dihuangkaijia.jpg

🤖 Qwen2-VL 回答：
----------------------------------------
这张图片展示了一个穿着华丽盔甲的角色，盔甲上覆盖着一层薄薄的雪。角色的盔甲主要是金色和银色的金属材质，带有复杂的图案和装饰。角色的头部有一个类似猫耳的设计，眼睛部分是红色的，显得非 常威严。背景模糊，似乎是在户外，可能是在雪地里。
----------------------------------------
```



## 3.最终包依赖：

```bash
xulab@xulab-EG341W-G21:~$ pip list
Package                        Version
------------------------------ ----------------
annotated-types                0.7.0
anyio                          4.12.0
appdirs                        1.4.4
apturl                         0.5.2
argcomplete                    3.6.3
asgiref                        3.11.0
async-timeout                  5.0.1
attr                           0.3.1
attrs                          25.4.0
azure-core                     1.37.0
azure-storage-blob             12.27.1
bcrypt                         3.2.0
black                          26.1a1
bleach                         5.0.1
blessings                      1.7
blinker                        1.4
boto3                          1.42.21
botocore                       1.42.21
Brlapi                         0.8.3
cachetools                     6.2.4
certifi                        2020.6.20
cffi                           2.0.0
chardet                        4.0.0
charset-normalizer             3.4.4
click                          8.0.3
colorama                       0.4.4
command-not-found              0.3
croniter                       6.0.0
cryptography                   46.0.3
cupshelpers                    1.0
datamodel-code-generator       0.26.1
dbus-python                    1.2.18
defer                          1.0.6
defusedxml                     0.7.1
distro                         1.7.0
distro-info                    1.1+ubuntu0.2
Django                         5.1.15
django-annoying                0.10.6
django-cors-headers            4.7.0
django-csp                     3.7
django-debug-toolbar           3.2.1
django-environ                 0.10.0
django-extensions              3.2.3
django-filter                  24.3
django-migration-linter        5.2.0
django-model-utils             4.1.1
django-ranged-fileresponse     0.1.2
django-rq                      3.1
django-storages                1.12.3
django-user-agents             0.4.0
djangorestframework            3.15.2
djangorestframework_simplejwt  5.5.1
dnspython                      2.8.0
drf-dynamic-fields             0.3.0
drf-flex-fields                0.9.5
drf-generators                 0.3.0
drf-spectacular                0.28.0
duplicity                      0.8.21
email-validator                2.3.0
exceptiongroup                 1.3.1
expiringdict                   1.2.2
Faker                          40.1.0
fasteners                      0.14.1
filelock                       3.20.2
fsspec                         2025.12.0
future                         0.18.2
genson                         1.3.0
google-api-core                2.28.1
google-auth                    2.45.0
google-cloud-appengine-logging 1.7.0
google-cloud-audit-log         0.4.0
google-cloud-core              2.5.0
google-cloud-logging           3.13.0
google-cloud-storage           2.19.0
google-crc32c                  1.8.0
google-resumable-media         2.8.0
googleapis-common-protos       1.72.0
gpustat                        0.6.0
grpc-google-iam-v1             0.14.3
grpcio                         1.76.0
grpcio-status                  1.76.0
h11                            0.16.0
httpcore                       1.0.9
httplib2                       0.20.2
httpx                          0.28.1
idna                           3.3
ijson                          3.4.0.post0
importlib_metadata             8.7.1
inflect                        5.6.2
inflection                     0.5.1
isodate                        0.7.2
isort                          5.13.2
jeepney                        0.7.1
Jinja2                         3.1.6
jiter                          0.12.0
jmespath                       1.0.1
joblib                         1.5.3
jsf                            0.11.2
jsonschema                     4.25.1
jsonschema-specifications      2025.9.1
keyring                        23.5.0
label-studio                   1.22.0
label-studio-sdk               2.0.16
language-selector              0.1
launchdarkly-server-sdk        8.2.1
launchpadlib                   1.10.16
lazr.restfulclient             0.14.4
lazr.uri                       1.0.6
lockfile                       0.12.2
louis                          3.20.0
lxml                           6.0.2
lxml_html_clean                0.4.3
macaroonbakery                 1.3.1
Mako                           1.1.3
markdown-it-py                 4.0.0
MarkupSafe                     2.0.1
mdurl                          0.1.2
monotonic                      1.6
more-itertools                 8.10.0
mpmath                         1.3.0
mypy_extensions                1.1.0
netifaces                      0.11.0
networkx                       3.4.2
ninja                          1.13.0
nltk                           3.9.2
numpy                          2.2.6
nvidia-cublas-cu11             11.11.3.6
nvidia-cuda-cupti-cu11         11.8.87
nvidia-cuda-nvrtc-cu11         11.8.89
nvidia-cuda-runtime-cu11       11.8.89
nvidia-cudnn-cu11              8.7.0.84
nvidia-cufft-cu11              10.9.0.58
nvidia-curand-cu11             10.3.0.86
nvidia-cusolver-cu11           11.4.1.48
nvidia-cusparse-cu11           11.7.5.86
nvidia-ml-py3                  7.352.0
nvidia-nccl-cu11               2.20.5
nvidia-nvtx-cu11               11.8.86
oauthlib                       3.2.0
olefile                        0.46
openai                         1.109.1
opencv-python-headless         4.12.0.88
opentelemetry-api              1.39.1
ordered-set                    4.0.2
packaging                      25.0
pandas                         2.3.3
paramiko                       2.9.3
pathspec                       0.12.1
pexpect                        4.8.0
pillow                         12.1.0
pip                            22.0.2
platformdirs                   4.5.1
proto-plus                     1.27.0
protobuf                       6.33.2
psutil                         5.9.0
psycopg                        3.3.2
psycopg-binary                 3.3.2
ptyprocess                     0.7.0
pyarrow                        22.0.0
pyasn1                         0.6.1
pyasn1_modules                 0.4.2
pyboxen                        1.3.0
pycairo                        1.20.1
pycparser                      2.23
pycups                         2.0.1
pydantic                       2.12.5
pydantic_core                  2.41.5
Pygments                       2.19.2
PyGObject                      3.42.1
PyJWT                          2.10.1
pymacaroons                    0.13.0
PyNaCl                         1.5.0
pyparsing                      2.4.7
pyRFC3339                      1.1
python-apt                     2.4.0+ubuntu4.1
python-dateutil                2.9.0.post0
python-debian                  0.1.43+ubuntu1.1
python-json-logger             2.0.4
pytokens                       0.3.0
pytz                           2022.1
pyxdg                          0.27
PyYAML                         6.0.3
redis                          5.2.1
referencing                    0.37.0
regex                          2025.11.3
reportlab                      3.6.8
requests                       2.32.5
requests-file                  3.0.1
requests-mock                  1.12.1
rich                           14.2.0
rpds-py                        0.30.0
rq                             2.6.1
rsa                            4.9.1
rstr                           3.2.2
rules                          3.4
s3transfer                     0.16.0
screen-resolution-extra        0.0.0
SecretStorage                  3.3.1
semver                         3.0.4
sentry-sdk                     2.48.0
setuptools                     80.9.0
six                            1.16.0
smart_open                     7.5.0
sniffio                        1.3.1
sqlparse                       0.5.5
ssh-import-id                  5.11
sympy                          1.14.0
systemd-python                 234
tldextract                     5.3.1
toml                           0.10.2
tomli                          2.3.0
torch                          2.3.0+cu118
torchaudio                     2.3.0+cu118
torchvision                    0.18.0+cu118
tqdm                           4.67.1
triton                         2.3.0
typing_extensions              4.15.0
typing-inspection              0.4.2
tzdata                         2025.3
ua-parser                      1.0.1
ua-parser-builtins             202601
ubuntu-drivers-common          0.0.0
ubuntu-pro-client              8001
ufw                            0.36.1
ujson                          5.11.0
unattended-upgrades            0.1
uritemplate                    4.2.0
urllib3                        2.6.2
usb-creator                    0.3.7
user-agents                    2.2.0
uuid_utils                     0.12.0
wadllib                        1.3.6
webencodings                   0.5.1
wheel                          0.40.0
wrapt                          2.0.1
xdg                            5
xkit                           0.0.0
xmljson                        0.2.1
zipp                           3.23.0
xulab@xulab-EG341W-G21:~$ 
```

# ==以上部署完成==
