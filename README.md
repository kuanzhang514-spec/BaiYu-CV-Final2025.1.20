readme
### 1.实验环境基于实验室服务器
所有文件里的IP地址都被我隐藏了，
解耦服务端与请求端，服务端部署Qwen CLIP SAM模型分别给封装成restful接口，本地请求端只需要请求接口就能使用模型功能，
纯python接口效率不好，后续可以使用java/go重写接口并模型部署成微服务形式提高并发

### **2.图片样例在“样本样例”目录，包括了原图与证据图（SAM经Qwen回答坐标分割后的图像）**

### 3.没上传的图片文件说明：
TextVQA DocVQA数据集的图片文件我又没上传，太大了，可以去huggingface下载

<img width="343" height="95" alt="image" src="https://github.com/user-attachments/assets/d9f005d7-a6a9-44ba-afc3-c666f9c719b1" />

<img width="208" height="97" alt="image" src="https://github.com/user-attachments/assets/a998862e-d11b-48c5-bb0d-972ed7b70784" />

所有的运行结果文件夹都应该有一个sam_segments 目录保存图片的证据图，我也没上传，太大了

<img width="1655" height="974" alt="image" src="https://github.com/user-attachments/assets/bb68bfc1-d5c4-49e4-aab3-a02de8afc8cf" />




