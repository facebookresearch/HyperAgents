# HyperAgents 项目分析

## 项目概览

**项目名称**: HyperAgents  
**作者**: Jenny Zhang, Bingchen Zhao, Wannan Yang, Jakob Foerster, Jeff Clune, Minqi Jiang, Sam Devlin, Tatiana Shavrina  
**组织**: Facebook Research (Meta AI)  
**仓库**: https://github.com/facebookresearch/Hyperagents  
**论文**: https://arxiv.org/abs/2603.19461  
**博客**: https://ai.meta.com/research/publications/hyperagents/  
**类型**: AI 代理研究 / 自改进代理系统  
**许可**: CC BY-NC-SA 4.0

### 核心价值
HyperAgents 是一个**自指、自改进的代理系统**，可以优化任何可计算的任务。
- 元代理 (Meta Agent) 改进任务代理 (Task Agent)
- 任务代理在特定领域执行任务
- 通过进化算法进行迭代优化

---

## 项目结构

```
Hyperagents/
├── agent/                      # 基础模型调用代码
├── analysis/                   # 绘图和分析脚本
├── baselines/                  # 基线算法实现
├── domains/                    # 各领域任务代码
├── utils/                      # 通用工具代码
├── generate_loop.py            # 主算法入口点
├── meta_agent.py               # 元代理主实现
├── task_agent.py               # 任务代理主实现
├── run_meta_agent.py           # 运行元代理并获取 diff
├── run_task_agent.py           # 运行任务代理
├── select_next_parent.py       # 选择下一个父代
├── ensemble.py                 # 集成方法
├── setup_initial.sh            # 初始化代理
├── requirements.txt            # Python 依赖
├── requirements_dev.txt        # 开发依赖
├── Dockerfile                  # Docker 容器配置
├── outputs_os_parts.zip        # 实验日志（多部分压缩）
├── outputs_os_parts.z01-z08   # 分卷压缩文件
├── README.md
├── LICENSE.md
├── CONTRIBUTING.md
└── CODE_OF_CONDUCT.md
```

---

## 技术架构

### 双层代理系统

```
┌─────────────────────────────────────────────────────────────┐
│                     HyperAgents 系统                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Meta Agent (元代理)                                   │ │
│  │  • 分析任务代理的性能                                   │ │
│  │  • 生成改进建议                                         │ │
│  │  • 创建任务代理的新版本                                 │ │
│  │  • 选择最优父代进行进化                                 │ │
│  └───────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Task Agent (任务代理)                                 │ │
│  │  • 在特定领域执行任务                                   │ │
│  │  • 生成可执行代码                                       │ │
│  │  • 与环境交互                                           │ │
│  │  • 输出性能指标                                         │ │
│  └───────────────────────────────────────────────────────┘ │
│                            │                                  │
│                            ▼                                  │
│  ┌───────────────────────────────────────────────────────┐ │
│  │  Domains (领域)                                         │ │
│  │  • 不同的任务环境                                       │ │
│  │  • 评估指标                                             │ │
│  └───────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 核心文件说明

| 文件 | 用途 |
|------|------|
| `meta_agent.py` | 元代理主实现 - 改进任务代理 |
| `task_agent.py` | 任务代理主实现 - 执行领域任务 |
| `generate_loop.py` | 主循环入口 - 运行完整进化算法 |
| `run_meta_agent.py` | 单独运行元代理并获取代码 diff |
| `run_task_agent.py` | 单独运行任务代理 |
| `select_next_parent.py` | 进化选择 - 选择下一个父代 |
| `ensemble.py` | 集成方法 - 组合多个代理 |
| `setup_initial.sh` | 初始化第一代代理 |

---

## 关键依赖

```txt
openai
anthropic
google-generativeai
graphviz
numpy
...
```

**支持的模型**:
- OpenAI GPT 系列
- Anthropic Claude 系列
- Google Gemini 系列

---

## 安装与配置

### 1. API 密钥配置

创建 `.env` 文件：
```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...
```

### 2. 系统依赖

```bash
sudo dnf install -y python3.12-devel
sudo dnf install -y graphviz graphviz-devel cmake ninja-build 
sudo dnf install -y bzip2-devel zlib-devel ncurses-devel libffi-devel
```

### 3. Python 环境

```bash
python3.12 -m venv venv_nat
source venv_nat/bin/activate
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

### 4. Docker 构建

```bash
docker build --network=host -t hyperagents .
```

### 5. 初始化代理

```bash
bash ./setup_initial.sh
```

---

## 运行 HyperAgents

### 主循环

```bash
python generate_loop.py --domains <domain>
```

输出默认保存在 `outputs/` 目录。

### 单独运行组件

```bash
# 运行元代理
python run_meta_agent.py

# 运行任务代理
python run_task_agent.py

# 选择下一个父代
python select_next_parent.py
```

---

## 实验日志

实验日志存储为多部分 ZIP 压缩包。解压方法：

```bash
zip -s 0 outputs_os_parts.zip --out unsplit_logs.zip
unzip unsplit_outputs.zip
```

---

## 安全警告

> [!WARNING]  
> 此仓库涉及执行**不受信任的、模型生成的代码**。
> 
> 虽然在当前设置和使用的模型下，此类代码不太可能执行明显的恶意操作，但由于模型能力或对齐的限制，它仍可能表现出破坏性。
> 
> 使用此仓库即表示您承认并接受这些风险。

---

## 学术引用

如果此项目对您有帮助，请考虑引用：

```bibtex
@misc{zhang2026hyperagents,
      title={Hyperagents}, 
      author={Jenny Zhang and Bingchen Zhao and Wannan Yang and Jakob Foerster and Jeff Clune and Minqi Jiang and Sam Devlin and Tatiana Shavrina},
      year={2026},
      eprint={2603.19461},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.19461}, 
}
```

---

## 项目亮点

1. **自改进系统** - 代理可以改进自身的代码
2. **双层架构** - 元代理 + 任务代理分离
3. **进化算法** - 通过选择和变异迭代优化
4. **多模型支持** - OpenAI、Anthropic、Google 模型
5. **完整实验日志** - 包含可复现的实验数据
6. **Docker 支持** - 容器化部署
7. **学术开源** - 配套 arXiv 论文

---

## 进一步探索

- 阅读 arXiv 论文: https://arxiv.org/abs/2603.19461
- 查看 Meta AI 博客: https://ai.meta.com/research/publications/hyperagents/
- 查看 `domains/` 了解支持的任务领域
- 查看 `agent/` 了解基础模型调用实现
- 查看 `analysis/` 了解数据分析和可视化方法
- 解压 `outputs_os_parts.zip` 查看实验日志
