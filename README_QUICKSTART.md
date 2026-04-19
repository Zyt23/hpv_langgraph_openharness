# HPV LangGraph OpenHarness 项目说明（Quickstart）

## 1. 项目做什么
这个项目用于 HPV 故障前兆二分类（`0=健康/远离故障`，`1=退化/接近故障`）：
- 使用 LangGraph 组织 `observe -> hypothesize -> validate -> reflect` 循环
- 使用 OpenHarness 调用大模型生成工况与规则
- 在训练集/验证集做文件夹级评估
- 最后可用独立 `test.py` 在人工提供的测试目录评估

## 2. 启动前需要补的 API Key
在项目根目录创建/编辑 `.env`，至少包含：

```env
OPENROUTER_API_KEY=你的OpenRouterKey
```

说明：
- 启动脚本会自动读取 `.env`
- 运行时会自动兼容为 OpenAI 格式接口调用（OpenRouter base-url）

## 3. 一键启动（Windows）
```powershell
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

启动后会自动：
- 加载 `.env`
- 配置 OpenHarness 本地可写目录（项目内 `.openharness`）
- 启动主流程 `python -m hpv_agent.run_agent --config configs/hpv_openharness.yaml`

## 4. 手动测试（独立 test）
```powershell
$env:PYTHONPATH='src'
python test.py --config configs/hpv_openharness.yaml --data-root ./test
```

## 5. 目录概览
```text
configs/                      # 配置文件（模型、循环参数、数据路径）
src/hpv_agent/                # 核心代码（graph/workers/metrics/adapter）
hpv/                          # 原始数据目录（建议不入库）
_outputs/                     # 运行产物目录（每次 run 一个子目录）
start.ps1                     # Windows 启动脚本
test.py                       # 独立测试脚本
```

## 6. Git 忽略与已跟踪文件处理
本项目已在 `.gitignore` 中加入：
- `hpv/**`
- `.env`
- `*.parquet`

如果这些文件已经被 Git 跟踪过，仅改 `.gitignore` 不够，还要执行：

```bash
git rm -r --cached hpv
git rm --cached .env
git rm --cached *.parquet
```

如果有子目录里的 parquet，建议用更稳妥版本：

```bash
git rm -r --cached -- '*.parquet'
```

然后提交一次：

```bash
git commit -m "chore: ignore local data/env/parquet files"
```

## 7. 常见问题
- 看起来“卡住”：
  - 优先看终端日志中的 `[bootstrap]/[observe]/[hypothesize]/[validate]` 阶段输出
  - 通常是模型调用耗时，不是卡在 parquet 构造
- 结果很差：
  - 先检查 `round_xxx/selected_condition.json` 和 `candidate_rule.json` 是否落入 fallback
  - 再看 `observe_trace.json` / `hypothesize_trace.json` 是否有真实工具调用记录
