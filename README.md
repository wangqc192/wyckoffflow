# WyckoffFlow Lightning

本文档说明 WyckoffFlow 模板到晶体结构的生成、评估和导出流程。

本文档中的文件命名约定：CrystalFlow 使用 `diffcsp`；DiffCSP++ 使用 `diffcsppp`。
命令中的 `<model_path>`、`<dataset>`、`<input_csv>` 等表示需要替换为实际路径或名称的参数。

## 模型评估

评估模型并计算指标：

~~~bash
python scripts/evaluate.py \
  --model_path <model_path> \
  --dataset <dataset>

uv run python scripts/compute_metrics_crystalflow.py \
  --root-path <crystalflow_dir> \
  --gt-file data/<dataset>/test.csv \
  --workers 32
~~~

## 一、已有 WyckoffFlow 模板 → 结构

### 1. WyckoffFlow 采样和模板导出

从公式和空间群开始采样：

~~~bash
uv run python scripts/sample_wy.py \
  --model_path <model_path> \
  --formula_file <formula_file> \
  --num_evals 20 \
  --batch_size 128 \
  --save_path outputs/flow_samples
~~~

输出 `outputs/flow_samples.pt`，再解码为可读模板：

~~~bash
uv run python scripts/extract_wyckoff_samples.py \
  --input_pt <flow_samples.pt> \
  --output_csv <flow_samples.csv>
~~~

CSV 主要包含：

- `sample_index`
- `target_index`
- `space_group`
- `formula`
- `target_formula`
- `wyckoff_occupancy`
- `count`

示例：

~~~csv
target_index,space_group,formula,target_formula,wyckoff_occupancy,count
0,216,Ga4Te4,Ga4Te4,216_Ga1x4d_Te1x4a,1
~~~

该步骤完成 Wyckoff 图解码、完整晶胞组成检查和组成守恒，并保留每一个采样结果，
不对重复模板去重；`count` 对每一行固定为 `1`。公式使用完整的
conventional-cell 计量，不会自动约分，例如 `Ga4Te4` 不会变成 `GaTe`。

### 2. WyckoffFlow 模板 → CrystalFlow

> 本节命令全部用于 CrystalFlow；`diffcsp` 仅保留在相关参数名和输出文件名中。

#### 先设置环境变量

以下路径适用于当前仓库和机器；`CRYSTALFLOW_GPU_ID` 可改为 `0`、`1`、`2` 或 `3`：

~~~bash
export CRYSTALFLOW_PYTHON=/home/wangqc/miniconda3/envs/crystalflow/bin/python
export CRYSTALFLOW_GPU_ID=0
export CRYSTALFLOW_REPO=/home/wangqc/DiffCSP
export CRYSTALFLOW_CHECKPOINT=/home/wangqc/DiffCSP/ckpt/CSP-mp20-sym
~~~

#### 第一步：转换为 CrystalFlow 输入

~~~bash
uv run python scripts/prepare_crystalflow_templates.py \
  --input <flow_samples.csv> \
  --output <crystalflow_dir>/wyckoff_info.csv \
  --manifest <crystalflow_dir>/manifest.csv
~~~

生成：

- `wyckoff_info.csv`：CrystalFlow 输入格式；
- `manifest.csv`：保留原始模板索引、`target_index`、模板顺序、目标信息及结构对应关系。

`wyckoff_info.csv` 示例：

~~~csv
formula,num_evals,pressure,wyckoff
Ga4Te4,1,0,216_Ga1x4d_Te1x4a
~~~

#### 第二步：CrystalFlow 生成结构

~~~bash
scripts/run_crystalflow_template_shard.sh \
  <crystalflow_dir>
~~~

脚本自动读取并生成：

- `<crystalflow_dir>/wyckoff_info.csv`：输入模板；
- `<crystalflow_dir>/diffcsp_queries.json`：结构化 query；
- `<crystalflow_dir>/samples.pt`：结构采样结果；
- `<crystalflow_dir>/sample.log`：日志文件。

Python、GPU、CrystalFlow 仓库和检查点路径从上面的环境变量读取。

默认参数：

- checkpoint：`/home/wangqc/DiffCSP/ckpt/CSP-mp20-sym`
- `N=100`：扩散/ODE 积分步数，不是每个模板生成 100 个结构；
- 坐标退火：开启，`anneal_slope=5`；
- `batch_size=128`。

当前流程通常为每个模板生成一个结构候选，输出分数坐标、原子类型、晶格参数、晶格长度、角度和原子数。

评估生成结构：

~~~bash
uv run python scripts/compute_metrics_crystalflow.py \
  --root-path <crystalflow_dir> \
  --manifest <crystalflow_dir>/manifest.csv \
  --gt-file data/<dataset>/test.csv \
  --sample-files <crystalflow_dir>/samples.pt \
  --workers 32 \
  --ltol 0.3 \
  --stol 0.5 \
  --angle-tol 10
~~~

该路径用于评估：模板命中率 → 结构有效率 → StructureMatcher 命中率。

### 3. WyckoffFlow 模板 → DiffCSP++

DiffCSP++ 使用结构化 query JSON，而不是简单的 `wyckoff_occupancy` CSV。例如：

~~~json
[
  {
    "spacegroup_number": 216,
    "wyckoff_letters": ["4a", "4d"],
    "atom_types": ["Te", "Ga"]
  }
]
~~~

准备批量评估输入：

~~~bash
python scripts/run_nextcrystal_diffcsppp.py prepare \
  --evaluation-csv <evaluation.csv> \
  --test-csv data/<dataset>/test.csv \
  --manifest-csv <diffcsppp_dir>/manifest.csv \
  --selected-json <diffcsppp_dir>/selected.json
~~~

`evaluation.csv` 至少需要包含：

- `material_index`
- `candidate_index`
- `space_group_rank`
- `template_rank`
- `target_structure_sequence`
- `generated_structure_sequence`

`prepare` 会检查候选槽位和组成，并生成 `manifest.csv` 与 `selected.json`。

#### DiffCSP++ 结构采样

固定使用四个分片、批大小 128：

~~~bash
for shard in 0 1 2 3; do
  python scripts/run_nextcrystal_diffcsppp.py sample \
    --diffcsppp-repo <diffcsppp_repo> \
    --checkpoint-dir <diffcsppp_checkpoint> \
    --selected-json <diffcsppp_dir>/selected.json \
    --output <diffcsppp_dir>/sample_shard${shard}.pt \
    --shard-index ${shard} \
    --num-shards 4 \
    --batch-size 128 \
    --device cuda \
    --overwrite
done
~~~

固定配置：

- `batch_size=128`
- `num_shards=4`
- `step_lr=1e-5`

四个分片按全局 query 顺序切分，每个输出都保存全局 `input_indices`。因此合并时不依赖分片内索引，
即使某个分片为空，也会生成合法的空 payload。

#### DiffCSP++ 结构评估

`compute_metrics_diffcsppp.py` 只保留 CSP 重建评估；它按 PT 中的全局 `input_indices` 读取
拼接的坐标和原子类型，并按 manifest 的 `target_index` 聚合候选：

~~~bash
uv run python scripts/compute_metrics_diffcsppp.py \
  --root-path <diffcsppp_dir> \
  --manifest <diffcsppp_dir>/manifest.csv \
  --gt-file data/<dataset>/test.csv \
  --sample-files <diffcsppp_dir>/samples_shard0.pt \
                 <diffcsppp_dir>/samples_shard1.pt \
                 <diffcsppp_dir>/samples_shard2.pt \
                 <diffcsppp_dir>/samples_shard3.pt \
  --workers 32 \
  --ltol 0.3 \
  --stol 0.5 \
  --angle-tol 10
~~~

也可以直接传入包含 `samples_shard*.pt` 的目录。`--tasks csp` 为兼容 DiffCSP++ 命令行保留，
当前不支持 `gen` 任务。`--ltol`、`--stol` 和 `--angle-tol` 分别控制 StructureMatcher 的晶格长度、
距离和角度容差，默认值为 `0.3`、`0.5` 和 `10`；`--angle_tol` 也可作为 `--angle-tol` 的别名。
两个评估脚本共用 `scripts/eval_utils.py`，但 CrystalFlow 和 DiffCSP++ 的 PT 数据布局分别由各自脚本解析。

#### NextCrystal 结构评估

`compute_metrics_nextcrystal.py` 使用和上面两个脚本完全相同的 CSP 重建口径，读取
`~/NextCrystal/outputs/mp_20/sample_structures/{query_index}.cif`。默认直接从
`postprocessed_assignments_from_top5_sg.csv` 和 `mp_test.json`（也支持 `mp_test.csv`）
重建候选到 MP20 输入的对应关系，不依赖额外的 `manifest.csv`：

~~~bash
uv run python scripts/compute_metrics_nextcrystal.py \
  --root-path ~/NextCrystal/outputs/mp_20 \
  --gt-file data/mp20/test.csv \
  --samples-dir ~/NextCrystal/outputs/mp_20/sample_structures \
  --workers 32 \
  --ltol 0.3 \
  --stol 0.5 \
  --angle-tol 10
~~~

如果已有预构建的 manifest，也可以通过 `--manifest` 传入；如果输入文件使用非默认文件名，
可以显式指定：

~~~bash
uv run python scripts/compute_metrics_nextcrystal.py \
  --root-path ~/NextCrystal/outputs/mp_20 \
  --assignment-csv <postprocessed_assignments.csv> \
  --query-file <mp_test.json-or-csv> \
  --gt-file data/mp20/test.csv \
  --workers 32
~~~

脚本按照 NextCrystal 的有效 assignment 展平顺序建立一基的 `query_index`，并按原始 MP20
输入累计 `candidate_rank`；这与 NextCrystal manifest 的候选分组一致。它保留没有候选结构的
目标样本，分母仍是完整 MP20 test 集；每个目标的多个候选取最佳匹配。组成有效性、结构有效性、
`primitive_cell=True`、`scale=True` 以及 StructureMatcher 容差均与 CrystalFlow/DiffCSP++
评估一致。默认输出为 `eval_metrics.json` 和 `eval_details.csv`。

## 二、当前推荐的端到端入口

从一个完整化学式直接生成结构：

~~~bash
uv run python scripts/generate_structure.py \
  --formula Ga4Te4 \
  --flow-checkpoint <flow_checkpoint> \
  --output-dir <output_dir> \
  --structure-backend diffcsppp
~~~

该流程依次执行：空间群预测、Wyckoff 模板生成和结构生成。

### 第 1 步：NextCrystal 预测空间群

输出：

- `nextcrystal_input.csv`
- `space_groups.csv`

默认预测 Space-group Top-5，并检查空间群的 Wyckoff multiplicity 是否能够实现目标完整组成。

### 第 2 步：WyckoffFlow 生成模板

默认参数：

- Space-group Top-K：5
- templates per space group：4
- template pool size：16
- sampling mode：`n-shot`

理论上每个材料最多生成 `5 × 4 = 20` 个模板；实际数量可能因组成不可行
或可行模板不足而减少。重复模板也会保留，不会在导出时去重。

输出：

- `templates.csv`：模板及层级信息；
- `templates.pt`：WyckoffFlow 原始图样本；
- `diffcsp_templates.csv`：CrystalFlow 文本输入；
- `diffcsp_queries.json`：两个后端共用的结构化 query；
- `run.json`：公式、checkpoint、随机种子和采样参数。

### 第 3 步：调用结构生成后端

#### CrystalFlow 后端

~~~bash
uv run python scripts/generate_structure.py \
  --formula Ga4Te4 \
  --flow-checkpoint <flow_checkpoint> \
  --output-dir <crystalflow_output_dir> \
  --structure-backend diffcsp
~~~

内部调用 `scripts/run_crystalflow_symmetry.py`，读取 `diffcsp_queries.json`，展开对称操作并生成晶格、
分数坐标和原子类型，最后写出标准化的 `diffcsp_sample.pt`。

#### DiffCSP++ 后端

~~~bash
uv run python scripts/generate_structure.py \
  --formula Ga4Te4 \
  --flow-checkpoint <flow_checkpoint> \
  --output-dir <diffcsppp_output_dir> \
  --structure-backend diffcsppp
~~~

内部固定调用四次 `scripts/run_nextcrystal_diffcsppp.py sample`，每次使用：

- `--batch-size 128`
- `--num-shards 4`

输出四个分片：

- `diffcsppp_sample_shard0.pt`
- `diffcsppp_sample_shard1.pt`
- `diffcsppp_sample_shard2.pt`
- `diffcsppp_sample_shard3.pt`

## 三、统一导出结构

CrystalFlow 和 DiffCSP++ 最终都通过 `write_structure_files(...)` 导出结构。该函数根据 `input_index`
将样本与 `templates.csv` 对齐，并使用 `pymatgen.Structure` 重建结构：

~~~text
lattice = lengths + angles
species = atom_types
coords = frac_coords
coords_are_cartesian = False
~~~

输出：

~~~text
structures.csv
cif/candidate_000_sg216.cif
cif/candidate_001_sg216.cif
poscar/candidate_000_sg216.vasp
poscar/candidate_001_sg216.vasp
~~~

`candidate_000` 表示模板候选编号，`sg216` 表示生成空间群。

## 四、索引约定

整个链路使用同一套索引：

~~~text
templates.csv 第 i 行
        ↕
diffcsp_queries.json 第 i 个 query
        ↕
结构采样 payload 的 input_index = i
        ↕
最终 structures.csv 第 i 个候选
~~~

DiffCSP++ 的各个分片虽然分别运行，但保存的是全局 `input_index`。比较 CrystalFlow 和 DiffCSP++ 时，
应固定同一份 `templates.csv` 和 `diffcsp_queries.json`，再用相同的模板索引比较 `diffcsp` 与 `diffcsppp` 的结果。

## 五、流程边界与评估

当前流程结束于：

~~~text
结构模型输出晶格和坐标 → 导出 CIF/POSCAR
~~~

暂不自动包含：

- 几何优化；
- DFT 弛豫；
- 能量排序；
- 结构稳定性预测；
- 额外的对称性修复。

评估分为两层：

1. **模板层**：使用 `eval_gwa.py` 检查空间群和 Wyckoff 模板，等价模板视为匹配；
2. **结构层**：使用 `compute_metrics_crystalflow.py` 评估 CrystalFlow，或 `compute_metrics_diffcsppp.py` 评估 DiffCSP++，检查组成、
   结构有效性和 StructureMatcher 匹配。

完整流程为：

~~~text
WyckoffFlow/NextCrystal 生成模板
        ↓
G-W-A、空间群和组成评估
        ↓
CrystalFlow 或 DiffCSP++ 生成结构
        ↓
结构有效性检查
        ↓
StructureMatcher 结构命中率
~~~
