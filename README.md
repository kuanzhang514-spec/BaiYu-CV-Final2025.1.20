readme
### 1.实验环境基于实验室服务器,搭建配置模型的文档是服务器部署.md系列文档
所有文件里的IP地址都被我隐藏了，
解耦服务端与请求端，服务端部署Qwen CLIP SAM模型分别给封装成restful接口，本地请求端只需要请求接口就能使用模型功能，
纯python接口效率不好，后续可以使用java/go重写接口并模型部署成微服务形式提高并发

### **2.图片样例在“样本样例”目录，包括了原图与证据图（SAM经Qwen回答坐标分割后的图像）**

### 3.代码文件说明
**主实验文件：**
1.基础闭环系统：main_experimentOfMyVQA.py \
2.自适应阈值VQA闭环系统：main_experimentOfMyVQA11.py \
4.基线系统（Qwen2-VL Only）：main_experimentOfMyVQA3.py \
5.对比实验：无CLIP验证的闭环系统：main_experimentOfMyVQA4.py  \
6.又重新设计了加速策略,使用多线程加速策略：main_experimentOfMyVQA55.py\
6.又重新设计了加速策略,多级缓存加速策略：main_experimentOfMyVQA555.py \
1. TextVQA 数据集实验 200个样本：main_experimentOfTextVQA.py \
1. DocVQA 数据集实验随机 200个样本：main_experimentOfDocVQA.py \

**测试文件：**
demo测试一下闭环方案行不行：main_vqa_loop.py     demo1.py   \
单独CLIP请求服务端测试代码：client_CLIP.py \
单独Qwen请求服务端测试代码：client_Qwen.py     demo2.py   \
单独SAM方案一测试：client_sam.py  \
单独SAM方案二测试（我自己的）：client_sam copy.py \

### 4.没上传的图片文件说明：
TextVQA DocVQA数据集的图片文件我又没上传，太大了，可以去huggingface下载

<img width="343" height="95" alt="image" src="https://github.com/user-attachments/assets/d9f005d7-a6a9-44ba-afc3-c666f9c719b1" />

<img width="208" height="97" alt="image" src="https://github.com/user-attachments/assets/a998862e-d11b-48c5-bb0d-972ed7b70784" />

所有的运行结果文件夹都应该有一个sam_segments 目录保存图片的证据图，我也没上传，太大了

<img width="1655" height="974" alt="image" src="https://github.com/user-attachments/assets/bb68bfc1-d5c4-49e4-aab3-a02de8afc8cf" />




