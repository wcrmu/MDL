# Feature Pipeline Data Layout Migration Prompt

把下面这段 prompt 发给下游 agent，用于把已有 feature pipeline 迁移到 `data/pipelines` / `data/processed` 新布局。

```text
你要把已有的 MDL feature pipeline 迁移到新的仓库内 data layout。

背景：
MDL 现在统一把 feature pipeline 放在仓库内：
<project-root>/data/pipelines/<dataset_name>

feature pipeline 生成的训练输入统一放在：
<project-root>/data/processed/<dataset_name>

原始数据、本地软链或缓存放在：
<project-root>/data/raw/<dataset_name>

小型可提交测试样本放在：
<project-root>/data/fixtures/<dataset_name>

你必须先阅读：
- <project-root>/docs/feature_pipeline_development.md
- <project-root>/docs/feature_pipeline_agent_playbook.md
- <project-root>/docs/feature_engineering_checklist.md
- <project-root>/docs/feature_pipeline_data_layout_migration.md

路径：
- MDL 仓库根目录：<project-root>
- 当前 feature pipeline 根目录：<old-feature-pipeline-root>
- 数据集名称：<dataset_name>
- 新 feature pipeline 根目录：<project-root>/data/pipelines/<dataset_name>
- 新 processed 输出目录：<project-root>/data/processed/<dataset_name>

硬性边界：
1. 不要修改 <project-root>/src、<project-root>/scripts、<project-root>/configs、<project-root>/tests。
2. 只允许修改：
   - <project-root>/data/pipelines/<dataset_name>/**
   - <project-root>/data/processed/<dataset_name>/**
   - <project-root>/data/fixtures/<dataset_name>/**
3. 不要改变特征工程逻辑、label 逻辑、split 逻辑、vocab 逻辑、manifest 协议字段或训练数据语义。
4. 不要实现 DDP、HDFS streaming、rank/worker shard 分配或训练代码改造。
5. 如果 processed 数据很大，不要复制大文件，先输出 NEEDS_USER_DECISION。
6. 如果必须修改 MDL core 才能跑通，停止并输出 NEEDS_FRAMEWORK_CHANGE。

迁移规则：
- 外部目录 MDL_feature_pipelines/<dataset_name> -> data/pipelines/<dataset_name>
- pipeline 内部 processed/ -> data/processed/<dataset_name>
- pipeline 内部 raw/ -> data/raw/<dataset_name>，只保留软链或路径说明，不提交大文件
- reports/ 保留在 data/pipelines/<dataset_name>/reports/
- README、configs、scripts、tests、reports 中的路径全部同步
- <feature-pipeline-root>/processed -> data/processed/<dataset_name>
- MDL_feature_pipelines -> data/pipelines

执行步骤：
1. 检查 <old-feature-pipeline-root> 的文件结构和 git 状态。
2. 创建 <project-root>/data/pipelines/<dataset_name>。
3. 迁移 feature pipeline 代码、configs、scripts、tests、reports、README。
4. 更新 pipeline 配置，使输出目录默认为 <project-root>/data/processed/<dataset_name>。
5. 如果存在 raw 路径配置，改为指向 <project-root>/data/raw/<dataset_name> 或保留为用户传参。
6. 更新 README 和 reports 中所有旧路径。
7. 不要改 manifest.json 的协议字段。
8. 不要改 MDL core 训练/评估/reader 代码。
9. 跑旧路径残留检查：
   rg -n "MDL_feature_pipelines|<feature-pipeline-root>/processed|adapter|Adapter|ADAPTER" <project-root>/data/pipelines/<dataset_name>
10. 如果 feature pipeline 自带测试，运行：
    cd <project-root>/data/pipelines/<dataset_name>
    python -m pytest tests
11. 从 <project-root> 验证 processed 数据：
    python scripts/preprocess.py --data-dir data/processed/<dataset_name> --max-rows 1000
12. 从 <project-root> 做 smoke train：
    python scripts/train.py --data-dir data/processed/<dataset_name> --epochs 1 --batch-size 32 --max-steps 2 --eval-max-batches 2
13. 输出迁移报告。

迁移报告必须包含：
- 迁移了哪些目录
- processed 输出目录是什么
- raw 数据路径如何处理
- 替换了哪些旧路径/旧术语
- 是否还有旧路径残留
- 测试命令和结果
- 是否需要 NEEDS_USER_DECISION 或 NEEDS_FRAMEWORK_CHANGE

特别注意：
feature pipeline 只负责 raw -> processed manifest dataset。
不要在 feature pipeline 中实现训练时 HDFS streaming、DDP rank/world_size 分片、torchrun、checkpoint resume。
这些未来会由 MDL core 的 dataio/trainer 层统一处理。

本次任务只做目录迁移和路径更新。任何非路径/命名相关的代码逻辑改动都必须停止并询问。
```
