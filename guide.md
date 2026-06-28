你要把一个已经实现完成的 MDL dataset adapter 改名为 feature pipeline。

  背景：
  MDL 项目已经不再使用 adapter 这个命名。统一术语是 feature pipeline。
  目录建议从 MDL_adapters/<dataset_name> 迁移到 MDL_feature_pipelines/<dataset_name>。
  文档、README、脚本说明、报告文件名和变量名也要同步改名。

  路径：
  - MDL 仓库根目录：<project-root>
  - 当前 adapter 根目录：<old-adapter-root>
  - 新 feature pipeline 根目录：<new-feature-pipeline-root>
  - 数据集名称：<dataset-name>

  硬性边界：
  1. 不要修改 <project-root>/src、<project-root>/scripts、<project-root>/configs、<project-root>/tests。
  2. 只允许修改当前 feature pipeline 相关目录和文档。
  3. 不要改变数据转换逻辑、特征逻辑、manifest 字段语义或输出数据格式。
  4. 不要重新生成大规模 processed 数据，除非我明确要求。
  5. 如果发现路径、import 或运行方式不确定，先停止并输出 NEEDS_USER_DECISION。

  改名规则：
  - adapter -> feature pipeline
  - Adapter -> Feature Pipeline
  - MDL_adapters -> MDL_feature_pipelines
  - <adapter-root> -> <feature-pipeline-root>
  - adapter_design.md -> feature_pipeline_design.md
  - adapter validation -> feature pipeline validation
  - adapter CLI -> feature pipeline CLI
  - adapter README -> feature pipeline README
  - adapter output -> feature pipeline output
  - dataset adapter -> dataset-specific feature pipeline

  执行步骤：
  1. 检查 <old-adapter-root> 的 git 状态和文件结构。
  2. 如果目录仍在 MDL_adapters 下，把整个目录迁移到 MDL_feature_pipelines 下。
  3. 全局搜索以下旧词并替换：
     - adapter
     - Adapter
     - ADAPTER
     - MDL_adapters
     - adapter-root
     - adapter_design
     - adapter_development
     - adapter_agent
  4. 重命名报告文件：
     - reports/adapter_design.md -> reports/feature_pipeline_design.md
  5. 更新 README、configs、scripts、tests、reports 中的路径和命令。
  6. 保持 Python 包名、模块名、函数名稳定，除非它们明显包含 adapter 且改名不会破坏 import。
  7. 不要改 manifest.json 的协议字段。
  8. 跑检查：
     - rg -n "adapter|Adapter|ADAPTER|MDL_adapters|adapter-root|adapter_design|adapter_development|adapter_agent" <new-feature-pipeline-root>
     - python scripts/preprocess.py --help
     - 如果存在 adapter/feature pipeline 自测：pytest tests 或 python -m pytest tests
     - 从 <project-root> 运行：
       python scripts/preprocess.py --data-dir <new-feature-pipeline-root>/processed --max-rows 1000
  9. 输出改名报告，包含：
     - 迁移了哪些目录
     - 重命名了哪些文件
     - 替换了哪些术语
     - 是否还有旧命名残留
     - 验收命令和结果

  注意：
  如果搜索结果中出现 adapter 但属于第三方库、历史兼容说明、外部 URL，先不要替换，列入报告让我确认。

  如果你想让它更保守，可以加一句：

  本次任务只做命名迁移，不允许改任何特征工程逻辑或数据生成逻辑。任何非命名相关 diff 都必须回滚或停止询问。
