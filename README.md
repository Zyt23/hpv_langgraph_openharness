# HPV Benchmark Agent（LangGraph + OpenHarness，弱先验版）

这版是按你的新要求改过的：

- 不再把工程经验硬编码成候选工况族或规则先验
- 把工程师报告 / PDF / DOCX / Markdown 当作 **knowledge skill**
- Agent 先做：**从过去经验和样本 schema 中提出工况条件**
- 再做：**在该工况下提规则、验证、反思**
- 最终测试完全不用 Agent，只用固化好的规则包

## 核心改动

1. **不再内置强先验 condition families**
   - 旧版是从一组预设工况族里选
   - 新版由 Agent 输出 `ConditionSpec`（条件 DSL）

2. **文档不需要手工转 Markdown**
   - 程序会自动把 `pdf/docx/md/txt` 统一归一化到 `knowledge_cache/*.md`
   - Agent 通过 search/read skill 获取文档片段

3. **Benchmark 结构更通用**
   - `observe`：提出工况条件 `ConditionSpec`
   - `hypothesize`：提出 `RuleBundle`
   - `validate`：程序验证
   - `reflect`：判断回到 observe 还是只改规则
   - `final_test`：文件夹级指标，不调用 Agent

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uv tool install git+https://github.com/HKUDS/OpenHarness.git
PYTHONPATH=src python -m hpv_agent.run_agent --config configs/hpv_openharness.yaml
```

## test.py

```bash
PYTHONPATH=src python test.py \
  --config configs/hpv_openharness.yaml \
  --data-root ./test \
  --rule artifacts/<run_name>/best_rule.json
```

## 产物

```text
artifacts/<run_name>/
  aircraft_index.parquet
  split_manifest.json
  knowledge_index.json
  knowledge_cache/*.md
  best_rule.json
  training_eval.json
  holdout_eval.json
  metrics_table.txt
  round_001/
    observed_schema.json
    selected_condition.json
    observation_summary.json
    candidate_rule.json
    validation_report.json
    reflection.json
```
