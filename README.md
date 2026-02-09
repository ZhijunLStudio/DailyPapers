# DailyPapers

🤖 自动抓取 HuggingFace Daily Papers，通过 AI 智能筛选并同步到 Zotero 的论文管理工具。

## ✨ 功能特性

- **自动抓取**: 每日自动从 HuggingFace Daily Papers 获取最新论文
- **AI 智能筛选**: 基于个人研究兴趣，使用 LLM 自动筛选感兴趣的论文
- **智能分类**: 根据 Zotero 现有目录结构，自动将论文归类到合适文件夹
- **元数据提取**: 自动提取论文标题、作者、摘要等元数据
- **Zotero 同步**: 自动上传论文 PDF、阅读笔记到 Zotero
- **本地备份**: 同时保存到本地指定目录，支持网盘同步

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

复制配置模板并修改：

```bash
cp config.yaml.example config.yaml
```

编辑 `config.yaml`：

```yaml
# Zotero 配置
zotero:
  library_id: 'YOUR_LIBRARY_ID'
  library_type: 'user'
  api_key: 'YOUR_API_KEY'

# OpenAI 配置
openai:
  api_key: 'YOUR_API_KEY'
  base_url: 'https://api.openai.com/v1'
  model: 'gpt-4'

# 本地存储配置
local_storage:
  base_dir: '/path/to/your/papers'

# 筛选偏好
preferences:
  interest: >
    你的研究兴趣...
  ignore: >
    你不想看的方向...
```

### 3. 运行

```bash
# 抓取今天的论文
python main.py

# 抓取指定日期的论文
python main.py --date 2024-01-15

# 指定日期范围
python main.py --start-date 2024-01-01 --end-date 2024-01-31
```

## 🔧 代理配置

如果你的网络需要代理，设置环境变量：

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
python main.py
```

## 📁 项目结构

```
DailyPapers/
├── main.py              # 主程序入口
├── config.yaml          # 配置文件（需自行创建）
├── requirements.txt     # 依赖列表
├── src/
│   ├── hf_scraper.py    # HuggingFace 论文抓取
│   ├── llm_agent.py     # AI 分析模块
│   ├── zotero_ops.py    # Zotero 操作
│   └── utils.py         # 工具函数
└── papers/              # 本地论文存储目录
```

## 📝 配置说明

### Zotero API Key 获取

1. 登录 [Zotero](https://www.zotero.org/)
2. 进入 [Settings → Feeds/API](https://www.zotero.org/settings/keys)
3. 创建新密钥，勾选读写权限
4. 获取 Library ID：在 [Settings → User ID](https://www.zotero.org/settings/keys) 页面

### 研究兴趣配置

在 `config.yaml` 的 `preferences` 部分：

- `interest`: 你感兴趣的研究方向，AI 会优先筛选这些论文
- `ignore`: 你不想看的方向，AI 会自动过滤

支持自然语言描述，越详细筛选效果越好。

## ⚠️ 注意事项

- **API 限制**: Zotero API 有速率限制，大量论文请分批处理
- **存储空间**: 请确保本地存储目录有足够空间
- **版权问题**: 请遵守论文版权，仅用于个人学习研究

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 License

MIT License
