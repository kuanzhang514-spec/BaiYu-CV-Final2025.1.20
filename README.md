readme 仓库日期写错了，是26年
### 1.实验环境基于实验室服务器,搭建配置模型的文档是服务器部署.md系列文档
所有文件里的IP地址都被我隐藏了，模型分别部署到了互相独立的python环境中，
考虑到解耦服务端与请求端，服务端部署Qwen CLIP SAM模型分别给封装成restful接口，本地请求端只需要请求接口就能使用模型功能，
纯python接口效率不好，后续可以使用java/go重写接口并模型部署成微服务形式提高并发

### **2.图片样例在“样本样例”目录，包括了原图与证据图（SAM经Qwen回答坐标分割后的图像）**

### 3.代码文件说明
**主实验文件：** \
1.基础闭环系统：main_experimentOfMyVQA.py \
2.自适应阈值VQA闭环系统：main_experimentOfMyVQA11.py \
4.基线系统（Qwen2-VL Only）：main_experimentOfMyVQA3.py \
5.无CLIP验证的闭环系统：main_experimentOfMyVQA4.py  \
6.又重新设计了加速策略,使用多线程加速策略：main_experimentOfMyVQA55.py\
6.又重新设计了加速策略,多级缓存加速策略：main_experimentOfMyVQA555.py \
1.TextVQA 数据集实验 200个样本：main_experimentOfTextVQA.py \
1.DocVQA 数据集实验随机 200个样本：main_experimentOfDocVQA.py 

**测试文件：** \
demo测试一下闭环方案行不行：main_vqa_loop.py     demo1.py   \
单独CLIP请求服务端测试代码：client_CLIP.py \
单独Qwen请求服务端测试代码：client_Qwen.py     demo2.py   \
单独SAM方案一测试：client_sam.py  \
单独SAM方案二测试（我自己的）：client_sam copy.py 

### 4.没上传的图片文件说明：
TextVQA DocVQA数据集的图片文件没上传，太大了，可以去huggingface下载

<img width="343" height="95" alt="image" src="https://github.com/user-attachments/assets/d9f005d7-a6a9-44ba-afc3-c666f9c719b1" />

<img width="208" height="97" alt="image" src="https://github.com/user-attachments/assets/a998862e-d11b-48c5-bb0d-972ed7b70784" />

所有的运行结果文件夹都应该有一个sam_segments 目录保存图片的证据图，我也没上传，太大了

<img width="1655" height="974" alt="image" src="https://github.com/user-attachments/assets/bb68bfc1-d5c4-49e4-aab3-a02de8afc8cf" />


### 5.本地请求端包环境
```bash
(qwen3_local) C:\Users\kuanzhang\Desktop\courseB\fuwuqisanhaoji>conda list
# packages in environment at C:\Users\kuanzhang\.conda\envs\qwen3_local:
#
# Name                    Version                   Build  Channel
accelerate                1.12.0                   pypi_0    pypi
aiohappyeyeballs          2.6.1                    pypi_0    pypi
aiohttp                   3.13.3                   pypi_0    pypi
aiosignal                 1.4.0                    pypi_0    pypi
altair                    6.0.0                    pypi_0    pypi
annotated-types           0.7.0                    pypi_0    pypi
anyio                     4.12.0                   pypi_0    pypi
async-timeout             5.0.1                    pypi_0    pypi
attrs                     25.4.0                   pypi_0    pypi
bitsandbytes              0.49.0                   pypi_0    pypi
blinker                   1.9.0                    pypi_0    pypi
bzip2                     1.0.8                h2bbff1b_6
ca-certificates           2025.12.2            haa95532_0
cachetools                6.2.4                    pypi_0    pypi
certifi                   2025.11.12               pypi_0    pypi
charset-normalizer        3.4.4                    pypi_0    pypi
click                     8.3.1                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
contourpy                 1.3.2                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
datasets                  4.5.0                    pypi_0    pypi
dill                      0.4.0                    pypi_0    pypi
distro                    1.9.0                    pypi_0    pypi
exceptiongroup            1.3.1                    pypi_0    pypi
expat                     2.7.3                h885b0b7_4
filelock                  3.20.0                   pypi_0    pypi
fonttools                 4.61.1                   pypi_0    pypi
frozenlist                1.8.0                    pypi_0    pypi
fsspec                    2025.10.0                pypi_0    pypi
gitdb                     4.0.12                   pypi_0    pypi
gitpython                 3.1.46                   pypi_0    pypi
h11                       0.16.0                   pypi_0    pypi
httpcore                  1.0.9                    pypi_0    pypi
httpx                     0.28.1                   pypi_0    pypi
huggingface-hub           0.36.0                   pypi_0    pypi
idna                      3.11                     pypi_0    pypi
iniconfig                 2.3.0                    pypi_0    pypi
jinja2                    3.1.6                    pypi_0    pypi
jiter                     0.12.0                   pypi_0    pypi
jsonpatch                 1.33                     pypi_0    pypi
jsonpointer               3.0.0                    pypi_0    pypi
jsonschema                4.26.0                   pypi_0    pypi
jsonschema-specifications 2025.9.1                 pypi_0    pypi
kiwisolver                1.4.9                    pypi_0    pypi
langchain                 1.2.0                    pypi_0    pypi
langchain-core            1.2.6                    pypi_0    pypi
langgraph                 1.0.5                    pypi_0    pypi
langgraph-checkpoint      3.0.1                    pypi_0    pypi
langgraph-prebuilt        1.0.5                    pypi_0    pypi
langgraph-sdk             0.3.1                    pypi_0    pypi
langsmith                 0.6.0                    pypi_0    pypi
libexpat                  2.7.3                h885b0b7_4
libffi                    3.4.4                hd77b12b_1
libzlib                   1.3.1                h02ab6af_0
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.10.8                   pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
multidict                 6.7.0                    pypi_0    pypi
multiprocess              0.70.18                  pypi_0    pypi
narwhals                  2.15.0                   pypi_0    pypi
networkx                  3.4.2                    pypi_0    pypi
numpy                     2.2.6                    pypi_0    pypi
openai                    2.15.0                   pypi_0    pypi
openssl                   3.0.18               h543e019_0
optimum                   2.1.0                    pypi_0    pypi
orjson                    3.11.5                   pypi_0    pypi
ormsgpack                 1.12.1                   pypi_0    pypi
packaging                 25.0                     pypi_0    pypi
pandas                    2.3.3                    pypi_0    pypi
pillow                    12.0.0                   pypi_0    pypi
pip                       25.3               pyhc872135_0
pluggy                    1.6.0                    pypi_0    pypi
propcache                 0.4.1                    pypi_0    pypi
protobuf                  6.33.4                   pypi_0    pypi
psutil                    7.2.0                    pypi_0    pypi
pyarrow                   22.0.0                   pypi_0    pypi
pydantic                  2.12.5                   pypi_0    pypi
pydantic-core             2.41.5                   pypi_0    pypi
pydeck                    0.9.1                    pypi_0    pypi
pygments                  2.19.2                   pypi_0    pypi
pyparsing                 3.3.1                    pypi_0    pypi
pytest                    9.0.2                    pypi_0    pypi
python                    3.10.19              h981015d_0
python-dateutil           2.9.0.post0              pypi_0    pypi
python-dotenv             1.2.1                    pypi_0    pypi
pytz                      2025.2                   pypi_0    pypi
pyyaml                    6.0.3                    pypi_0    pypi
referencing               0.37.0                   pypi_0    pypi
regex                     2025.11.3                pypi_0    pypi
requests                  2.32.5                   pypi_0    pypi
requests-toolbelt         1.0.0                    pypi_0    pypi
rpds-py                   0.30.0                   pypi_0    pypi
safetensors               0.7.0                    pypi_0    pypi
sentencepiece             0.2.1                    pypi_0    pypi
setuptools                80.9.0          py310haa95532_0
six                       1.17.0                   pypi_0    pypi
smmap                     5.0.2                    pypi_0    pypi
sniffio                   1.3.1                    pypi_0    pypi
sqlite                    3.51.0               hda9a48d_0
streamlit                 1.53.1                   pypi_0    pypi
sympy                     1.13.1                   pypi_0    pypi
tenacity                  9.1.2                    pypi_0    pypi
tk                        8.6.15               hf199647_0
tokenizers                0.22.1                   pypi_0    pypi
toml                      0.10.2                   pypi_0    pypi
tomli                     2.4.0                    pypi_0    pypi
torch                     2.5.1+cu121              pypi_0    pypi
torchaudio                2.5.1+cu121              pypi_0    pypi
torchvision               0.20.1+cu121             pypi_0    pypi
tornado                   6.5.4                    pypi_0    pypi
tqdm                      4.67.1                   pypi_0    pypi
transformers              4.57.3                   pypi_0    pypi
typing-extensions         4.15.0                   pypi_0    pypi
typing-inspection         0.4.2                    pypi_0    pypi
tzdata                    2025.3                   pypi_0    pypi
ucrt                      10.0.22621.0         haa95532_0
urllib3                   2.6.2                    pypi_0    pypi
uuid-utils                0.12.0                   pypi_0    pypi
vc                        14.3                h2df5915_10
vc14_runtime              14.44.35208         h4927774_10
vs2015_runtime            14.44.35208         ha6b5a95_10
watchdog                  6.0.0                    pypi_0    pypi
wheel                     0.45.1          py310haa95532_0
xxhash                    3.6.0                    pypi_0    pypi
xz                        5.6.4                h4754444_1
yarl                      1.22.0                   pypi_0    pypi
zlib                      1.3.1                h02ab6af_0
zstandard                 0.25.0                   pypi_0    pypi

```bash

