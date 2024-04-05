# RAG

## 介绍

ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B-32k 是 ChatGLM3 系列中的开源模型。

M3E 是 Moka Massive Mixed Embedding 的缩写.此模型由 MokaAI 训练，开源和评测，训练脚本使用 [uniem](https://github.com/wangyuxinwhy/uniem/blob/main/scripts/train_m3e.py) ，评测 BenchMark 使用 [MTEB-zh](https://github.com/wangyuxinwhy/uniem/tree/main/mteb-zh)

-----

ChatGLM3 开源模型旨在与开源社区一起推动大模型技术发展，恳请开发者和大家遵守[开源协议](MODEL_LICENSE)，勿将开源模型和代码及基于开源项目产生的衍生物用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。目前，本项目团队未基于 **ChatGLM3 开源模型**开发任何应用，包括网页端、安卓、苹果 iOS 及 Windows App 等应用。

尽管模型在训练的各个阶段都尽力确保数据的合规性和准确性，但由于 ChatGLM3-6B 模型规模较小，且模型受概率随机性因素影响，无法保证输出内容的准确。同时模型的输出容易被用户的输入误导。**本项目不承担开源模型和代码导致的数据安全、舆情风险或发生任何模型被误导、滥用、传播、不当利用而产生的风险和责任。**

## 友情链接
对 ChatGLM3 进行加速的开源项目：
* [chatglm.cpp](https://github.com/li-plus/chatglm.cpp): 类似 llama.cpp 的量化加速推理方案，实现笔记本上实时对话
* [ChatGLM3-TPU](https://github.com/sophgo/ChatGLM3-TPU): 采用TPU加速推理方案，在算能端侧芯片BM1684X（16T@FP16，内存16G）上实时运行约7.5 token/s

## 使用方式

以ChatGLM3的本地化部署，共有clip,streamlit,gradio,openai-style四种demo和调试方式；
基于p-tuning v2微调技术，对于特定任务进行微调，例如，单论对话情境下，AdvertiseGen数据集训练赋予模型由商品标签自动生成对商品描述，也支持多轮对话微调；
利用ChatGLM3-6B-32k结合M3E表示模型检索增强生成，用于根据更新的时装周刊数据库对模型检索增强。

### 环境安装
首先需要下载本仓库：
```shell
git https://github.com/152-zz/CHAT2clothes.git
cd ChatGLM3
```

然后使用 pip 安装依赖：
```
pip install -r requirements.txt
```
在ChatGLM3下创建两个文件夹：
chatglm3-6b-32k：地址 https://www.modelscope.cn/ZhipuAI/chatglm3-6b-32k.git

M3E-base: 地址 https://www.modelscope.cn/Jerry0/m3e-base.git

+ `transformers` 库版本应该 `4.30.2` 以及以上的版本 ，`torch` 库版本应为 2.0 及以上的版本，以获得最佳的推理性能。
+ 为了保证 `torch` 的版本正确，请严格按照 [官方文档](https://pytorch.org/get-started/locally/) 的说明安装。
+ `gradio` 库版本应该为 `3.x` 的版本。

+ 还原rag的database部分需要加载PyPDF2之类的包，以及部分时尚杂志，版权问题，暂不公开

### 综合 

可自行参考https://github.com/THUDM/ChatGLM3和https://huggingface.co/moka-ai/m3e-base
