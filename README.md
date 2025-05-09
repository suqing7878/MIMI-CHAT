## MIMI CHAT: A User-friendly AI Chat Tool ʕ •ᴥ•ʔ
### 项目简介
MIMI CHAT 是一个基于 Flask 后端构建的 AI 聊天聚合平台。与 OpenWeb UI 等项目不同，本项目实现了完全的前后端分离，提供极高的后端自由度，支持用户自定义任意扩展，例如 RAG、Web 搜索等功能。这使得平台更灵活，适用于各种高级应用场景，如联网搜索、文档解析等。
### 运行指南
以下是快速启动 MIMI CHAT 的步骤。确保您的环境已安装 Python 和 Git。
1. 克隆仓库：
`git clone https://github.com/suqing7878/MIMI-CHAT.git`
2. 进入项目目录：
`cd MIMI-CHAT`
3. 安装依赖：
`pip install -r requirements.txt`
4. 运行应用：
`python app.py`
5. 服务器端运行
如果需要在特定 IP 和端口上运行应用，请编辑 `app.py` 文件中的 `app.run()` 语句，例如：
`app.run(host='xxx.xxx.x.xx', port=xxxx, debug=False)`
请将 `xxx.xxx.x.xx` 和 `xxxx` 替换为您的实际 IP 和端口号。
6. API 密钥配置
为了启用 AI 功能，您需要在 `MIMI-CHAT/api_key.py` 文件中添加相应的 API 密钥。

### 支持功能
MIMI CHAT 提供了丰富的功能，支持多种 AI 模型和扩展。以下是主要功能列表：
- **全模型联网搜索**：支持 function call 机制，包含国内源（如博查）和国际源（如 EXA、Jina），实现高效的实时搜索。
- **全模型文档解析**：不依赖模型自身的文档理解能力，支持多种格式，包括 Word、PPT、Excel、CSV、TXT 等，确保准确解析。
- **历史聊天记录自适应切换**：允许无缝切换模型，例如从 GPT 到 Gemini，或从单模态到多模态模型，保持聊天连续性。
- **文生图与图像编辑**：兼容最新模型，如 GPT-image-1、Gemini、FLUX 等，支持生成和编辑图像。
- **历史文件管理**：包括图像、文档等文件的下载、删除和重命名操作，便于用户管理聊天历史。
- **模型兼容性**：支持所有 OpenAI 兼容模型；对于非兼容模型，后端可高度自定义以实现扩展。
- **PC-手机端兼容**：支持PC和手机端的双端同步，专门适配的不同型号手机端的使用。
### Web UI
![Image 1](https://github.com/suqing7878/MIMI-CHAT/blob/main/image/1.png?raw=true)

![Image 2](https://github.com/suqing7878/MIMI-CHAT/blob/main/image/2.png?raw=true)
### 贡献与反馈
欢迎提交 issue 或 pull request 以改进项目。如果您有任何问题，请通过 GitHub 联系作者。
