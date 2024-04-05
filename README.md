# ChatGLM3 & nlp_gte_sentence-embedding_chinese-large

## 介绍

ChatGLM3 是智谱AI和清华大学 KEG 实验室联合发布的新一代对话预训练模型。ChatGLM3-6B-32k 是 ChatGLM3 系列中的开源模型。

nlp_gte_sentence-embedding_chinese-large 文本表示是自然语言处理(NLP)领域的核心问题, 其在很多NLP、信息检索的下游任务中发挥着非常重要的作用。近几年, 随着深度学习的发展，尤其是预训练语言模型的出现极大的推动了文本表示技术的效果, 基于预训练语言模型的文本表示模型在学术研究数据、工业实际应用中都明显优于传统的基于统计模型或者浅层神经网络的文本表示模型

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
利用ChatGLM3-6B-32k结合GTE中文通用表示模型检索增强生成(需先执行download.sh下载数据集)

### 环境安装
首先需要下载本仓库：
```shell
git https://github.com/152-zz/GARMLGTAHC.git
cd ChatGLM3
```

然后使用 pip 安装依赖：
```
pip install -r requirements.txt
```
在ChatGLM3下创建两个文件夹：
chatglm3-6b-32k：地址 https://www.modelscope.cn/ZhipuAI/chatglm3-6b-32k.git
nlp_gte_sentence-embedding_chinese-large: 地址https://www.modelscope.cn/iic/nlp_gte_sentence-embedding_chinese-large.git


+ `transformers` 库版本应该 `4.30.2` 以及以上的版本 ，`torch` 库版本应为 2.0 及以上的版本，以获得最佳的推理性能。
+ 为了保证 `torch` 的版本正确，请严格按照 [官方文档](https://pytorch.org/get-started/locally/) 的说明安装。
+ `gradio` 库版本应该为 `3.x` 的版本。

### 综合 Demo

提供了一个集成以下三种功能的综合 Demo，运行方法请参考 [综合 Demo](composite_demo/README.md)

- Chat: 对话模式，在此模式下可以与模型进行对话。
- Tool: 工具模式，模型除了对话外，还可以通过工具进行其他操作。
    <img src="resources/tool.png" width="400">
- Code Interpreter: 代码解释器模式，模型可以在一个 Jupyter 环境中执行代码并获取结果，以完成复杂任务。
    <img src="resources/heart.png" width="400">



#### 从本地加载模型

从 Hugging Face Hub 下载模型需要先[安装Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)，然后运行
git clone https://huggingface.co/THUDM/chatglm3-6b


### 模型微调

请参考对话模型微调 [ChatGLM3-6B 微调示例](finetune_chatmodel_demo/README.md),或基座模型微调 [ChatGLM3-6B-base 微调示例](finetune_basemodel_demo/README.md)。
请注意，不同的微调脚本对应的模型并不相同，请根据需要选择对应的模型。

### 网页版对话 Demo

![web-demo](resources/web-demo.gif)
可以通过以下命令启动基于 Gradio 的网页版 demo：
```shell
python web_demo_gradio.py
```

![web-demo](resources/web-demo2.png)

可以通过以下命令启动基于 Streamlit 的网页版 demo：
```shell
streamlit run web_demo_streamlit.py
```

网页版 demo 会运行一个 Web Server，并输出地址。在浏览器中打开输出的地址即可使用。 经测试，基于 Streamlit 的网页版 Demo 会更流畅。

### 命令行对话 Demo

![cli-demo](resources/cli-demo.png)

运行仓库中 [cli_demo.py](basic_demo/cli_demo.py)：

```shell
python cli_demo.py
```

程序会在命令行中进行交互式的对话，在命令行中输入指示并回车即可生成回复，输入 `clear` 可以清空对话历史，输入 `stop` 终止程序。

### LangChain Demo

代码实现请参考 [LangChain Demo](langchain_demo/README.md)。

#### 工具调用

关于工具调用的方法请参考 [工具调用](tools_using_demo/README.md)。 

#### OpenAI API Demo

感谢 [@xusenlinzy](https://github.com/xusenlinzy) 实现了 OpenAI 格式的流式 API 部署，可以作为任意基于 ChatGPT 的应用的后端，比如 [ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web)。可以通过运行仓库中的[openai_api.py](openai_api_demo/openai_api.py) 进行部署：
```shell 
cd openai_api_demo 
python openai_api.py
```
同时，我们也书写了一个示例代码，用来测试API调用的性能。可以通过运行仓库中的[openai_api_request.py](openai_api_demo/openai_api_request.py) 进行测试
+ 使用Curl进行测试
```shell
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d "{\"model\": \"chatglm3-6b\", \"messages\": [{\"role\": \"system\", \"content\": \"You are ChatGLM3, a large language model trained by Zhipu.AI. Follow the user's instructions carefully. Respond using markdown.\"}, {\"role\": \"user\", \"content\": \"你好，给我讲一个故事，大概100字\"}], \"stream\": false, \"max_tokens\": 100, \"temperature\": 0.8, \"top_p\": 0.8}"
````
+ 使用Python进行测试
```shell
cd openai_api_demo
python openai_api_request.py
```
如果测试成功，则模型应该返回一段故事。

## 低成本部署

### 模型量化

默认情况下，模型以 FP16 精度加载，运行上述代码需要大概 13GB 显存。如果你的 GPU 显存有限，可以尝试以量化方式加载模型，使用方法如下：

```python
model = AutoModel.from_pretrained("THUDM/chatglm3-6b",trust_remote_code=True).quantize(4).cuda()
```

模型量化会带来一定的性能损失，经过测试，ChatGLM3-6B 在 4-bit 量化下仍然能够进行自然流畅的生成。

### CPU 部署

如果你没有 GPU 硬件的话，也可以在 CPU 上进行推理，但是推理速度会更慢。使用方法如下（需要大概 32GB 内存）
```python
model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).float()
```

### Mac 部署

对于搭载了 Apple Silicon 或者 AMD GPU 的 Mac，可以使用 MPS 后端来在 GPU 上运行 ChatGLM3-6B。需要参考 Apple 的 [官方说明](https://developer.apple.com/metal/pytorch) 安装 PyTorch-Nightly（正确的版本号应该是2.x.x.dev2023xxxx，而不是 2.x.x）。

目前在 MacOS 上只支持[从本地加载模型](README.md#从本地加载模型)。将代码中的模型加载改为从本地加载，并使用 mps 后端：

```python
model = AutoModel.from_pretrained("your local path", trust_remote_code=True).to('mps')
```

加载半精度的 ChatGLM3-6B 模型需要大概 13GB 内存。内存较小的机器（比如 16GB 内存的 MacBook Pro），在空余内存不足的情况下会使用硬盘上的虚拟内存，导致推理速度严重变慢。

### 多卡部署

如果你有多张 GPU，但是每张 GPU 的显存大小都不足以容纳完整的模型，那么可以将模型切分在多张GPU上。首先安装 accelerate: `pip install accelerate`，然后即可正常加载模型。

### TensorRT-LLM Demo

ChatGLM3-6B已经支持使用 TensorRT-LLM 工具包进行加速推理，模型推理速度得到多倍的提升。具体使用方法请参考 [TensorRT-LLM Demo](tensorrt_llm_demo/tensorrt_llm_cli_demo.py) 和 官方技术文档。


## 引用

如果你觉得工作有帮助的话，请考虑引用下列论文。

```
@article{zeng2022glm,
  title={Glm-130b: An open bilingual pre-trained model},
  author={Zeng, Aohan and Liu, Xiao and Du, Zhengxiao and Wang, Zihan and Lai, Hanyu and Ding, Ming and Yang, Zhuoyi and Xu, Yifan and Zheng, Wendi and Xia, Xiao and others},
  journal={arXiv preprint arXiv:2210.02414},
  year={2022}
}
```
```
@inproceedings{du2022glm,
  title={GLM: General Language Model Pretraining with Autoregressive Blank Infilling},
  author={Du, Zhengxiao and Qian, Yujie and Liu, Xiao and Ding, Ming and Qiu, Jiezhong and Yang, Zhilin and Tang, Jie},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={320--335},
  year={2022}
}
```
