# HPV LangGraph OpenHarness Quickstart

## 1) 项目简介
本项目用于 HPV 前兆二分类（`0=健康/远离故障`，`1=退化/接近故障`）：
- 流程：`observe -> hypothesize -> validate -> reflect`
- 由 OpenHarness 调用大模型生成工况与规则
- 输出训练/验证集文件夹级指标
- 支持独立 `test.py` 做人工测试阶段验证

## 2) 环境安装（必须）

### 2.1 Python 依赖
```bash
pip install -r requirements.txt
```

### 2.2 安装 OpenHarness CLI
推荐：
```bash
pip install git+https://github.com/HKUDS/OpenHarness.git
```

安装后确认：
```bash
openharness --help
```

## 3) 配置 API Key（必须）
在项目根目录创建 `.env`：

```env
OPENROUTER_API_KEY=你的OpenRouterKey
```

说明：
- 启动脚本会自动加载 `.env`
- 运行时会将 OpenRouter 以 OpenAI-Compatible 方式调用

## 4) 一键启动（Windows）
```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

## 5) 独立测试（test.py）
```powershell
$env:PYTHONPATH='src'
python test.py --config configs/hpv_openharness.yaml --data-root ./test
```

## 6) 目录说明
```text
configs/                    # 配置
src/hpv_agent/              # 核心代码
hpv/                        # 原始数据（不入库）
_outputs/                   # 运行产物
start.ps1                   # Windows 启动脚本
test.py                     # 独立测试脚本
```

## 7) Git 忽略与取消跟踪
`.gitignore` 已包含：
- `hpv/**`
- `.env`
- `*.parquet`

若这些文件曾被跟踪，需执行：

```bash
git rm -r --cached hpv
git rm --cached .env
git rm --cached *.parquet
```

子目录 parquet 更稳妥写法：
```bash
git rm -r --cached -- '*.parquet'
```

然后提交：
```bash
git commit -m "chore: ignore local data/env/parquet files"
```
