# WyckoffFlow Lightning

本文档说明 WyckoffFlow 模板到晶体结构的生成、评估和导出流程。

本文档中的文件命名约定：CrystalFlow 使用 `diffcsp`；DiffCSP++ 使用 `diffcsppp`。
命令中的 `<model_path>`、`<dataset>`、`<input_csv>` 等表示需要替换为实际路径或名称的参数。

## 实验目录命名

训练目录自动包含实验类型、实验标签、数据配置、网络结构和关键训练参数，例如：

```text
outputs/2026-09-28/09-56-08_discrete_flow_baseline_mp20_crystal_gnn_h256_l4_bs256_lr0.0001_wd0.0_ep1000_s42/
```

`h` / `l` 分别是网络隐藏维度 / 层数，`bs` 是单设备训练 batch size，
`lr` / `wd` 是学习率 / weight decay，`ep` 是最大训练轮数，`s` 是 seed。
空间群分类使用 `sg_mlp` 并显示组成编码维度 `c`；联合训练额外显示
SG / Flow 损失权重 `sgw` / `floww`。这些数值随实际配置自动更新。

用 `run_tag` 标明实验目的或未列入目录名的消融参数，默认为空：

```bash
uv run python -m models.run \
  run_tag=no_film model.decoder.composition_film=false

uv run python -m models.run --multirun \
  run_tag=capacity model.decoder.num_gnn_layers=3,6 model.decoder.hidden_dim=256,512
```

标签建议只使用字母、数字、下划线和连字符。多参数扫描中，每个子目录采用
`序号_完整实验名`。目录名展示关键参数，全部生效配置保存在 `hparams.yaml`，
命令行覆盖保存在 `.hydra/overrides.yaml`；其他消融开关需通过标签说明。
仍可用 `hydra.run.dir=...` 显式指定输出路径。

## 训练中的验证集模板重建率

`uv run python -m models.run` 默认在完成第 100、200、300…个 epoch 时，
对完整验证集计算模板重建率。使用真实空间群和完整组成、50 步、20 次 n-shot
采样，同一批 flow 轨迹的末步 logits 分别进行组成守恒解码和无计数约束的
categorical 采样。按 `eval_gwa.py` 相同的等价模板口径记录：

| 指标 | 计数守恒开启 | 计数守恒关闭 |
| --- | --- | --- |
| GWA@1 | `val/gwa_top1` | `val/gwa_top1_no_composition` |
| GWA@20 | `val/gwa_top20` | `val/gwa_top20_no_composition` |
| 组成正确率 | `val/composition_accuracy` | `val/composition_accuracy_no_composition` |

组成正确率为完整晶胞各元素计数正确的样本数 / 请求的样本总数；空模板、错误组成
及漏生成均计入未命中，不筛选或补采样。关闭计数守恒仍保留化学式条件和元素掩码。
这里评估的是 Wyckoff 模板重建，不调用结构生成模型。

每次评估保存到训练目录的 `reconstruction/epoch_0099/` 等目录中
（目录编号为零基 epoch，`0099` 对应第 100 轮）：

- `summary.json`：重建率、轮次、随机种子和采样参数；
- `details.csv`：逐材料命中情况；
- `samples.csv`：所有生成候选，保留重复模板；
- `no_composition/`：关闭计数守恒的同名三份文件；根目录 `summary.json` 也包含这组指标。

最佳检查点的文件名包含零基 epoch 编号，例如第 100 轮对应 `0099`：

- `checkpoints/best_epoch_0099.ckpt`：按验证 loss 选择；
- `checkpoints/best_gwa_epoch_0099.ckpt`：按开启计数守恒时的验证 GWA@20 选择；
- `checkpoints/best_gwa_no_composition_epoch_0099.ckpt`：按关闭计数守恒时的验证 GWA@20 选择。

默认每类只保留一个最佳检查点；出现更优结果时，保存新文件并移除该类旧的最佳文件。
每 100 轮的 `epoch_*.ckpt` 和用于断点续训的 `last.ckpt` 也照常保存。
两组评估共用一次 flow 前向过程；额外的无约束采样不改变后续 flow 的随机数。
评估固定随机种子，
结束后恢复训练 RNG 状态。多 GPU 训练时由 rank 0 评估完整验证集并同步指标。

参数可通过 Hydra 覆盖，例如：

```bash
uv run python -m models.run \
  train.reconstruction.every_n_epochs=100 \
  train.reconstruction.num_samples=20 \
  train.reconstruction.flow_steps=50
```

`train.reconstruction.batch_size=128` 限制每批轨迹总数（含重复采样），
`train.reconstruction.cpu_workers=8` 控制解码进程数。
使用 `train.reconstruction.enabled=false` 可关闭；空间群分类实验默认关闭。
修改配置后需启动或恢复训练，已经运行的训练进程不会自动加载新回调。

画训练和重建曲线：

```bash
# 单个实验：自动合并其 resume_*/logs/metrics.csv
uv run python scripts/plot_loss.py outputs/2026-09-24/09-56-08_discrete_flow

# 批量处理 outputs 下的实验（含 old），每个实验保存到 logs/loss.png
uv run python scripts/plot_loss.py outputs
```

脚本按已有指标显示训练/验证 loss、GWA@1/@K 和组成正确率，区分开启/关闭
计数守恒；联合模型还显示任务 loss、空间群准确率和预测空间群后的重建结果。
重建指标优先使用 `reconstruction/epoch_*/summary.json` 中的等价模板评估结果，
缺少 summary 时使用 CSV 指标；只在实际评估轮次画点，不填充未评估轮次。
横轴为已完成轮数（零基 epoch + 1），准确率显示为百分比。loss 纵轴按各条
曲线的最小值到 95% 分位数取并集，并留 5% 边距；初期高值可能超出显示范围，
以突出主要训练阶段的变化。
也可传入 `logs` 目录或 `metrics.csv` 只读取该份日志；单个实验可用
`--output path/to/figure.png` 指定图片路径。绘图不依赖 LaTeX 或 SciencePlots。

## 图网络配置与消融

参考 DiffCSP，decoder 使用独立的 Hydra 配置组
`conf/model/decoder/`，默认是 `wyckoff_gnn.yaml`。网络层数、隐藏维度、
组成编码、FiLM 和 MLP 等参数放在 `model.decoder` 中；source、损失权重等
直接放在 `model` 中，例如
`model.zero_df_loss_weight=2.0` 和 `model.inf_df_loss_weight=1.0`。

~~~bash
uv run python -m models.run \
  model/decoder=wyckoff_gnn \
  model.decoder.num_gnn_layers=6 \
  model.decoder.hidden_dim=512

# 对层数和 FiLM 做组合消融
uv run python -m models.run --multirun \
  model.decoder.num_gnn_layers=2,3,6 \
  model.decoder.composition_film=true,false
~~~

新增图网络时，在 `conf/model/decoder/<name>.yaml` 中配置该网络的 `_target_`
和参数，再用 `model/decoder=<name>` 切换。flow 自动向构造函数传入
`num_elements`、`max_num_atoms`、`conditional_composition` 和 `continuous_time=True`。
decoder 的 `forward(data, time)` 应返回 `(zero_logits, inf_logits)`，形状分别为
`[零自由度节点数, num_elements + 1]` 和
`[非零自由度节点数, num_elements, max_num_atoms + 1]`。

`mlp_hidden_layers` 直接表示隐藏层数，不包括最终 Linear 输出层。
当前默认值为 3，保持原网络深度。历史配置采用“额外隐藏层数”语义，复用时需将
旧值加 1；旧 checkpoint 内嵌的对应配置（`sg_head`、`decoder` 或独立 SG 的顶层
`mlp_hidden_layers`）也需同步修改或在加载时覆盖，仅修改 `conf/` 不会更新这些值。

新增的 `crystal_gnn` 使用 Wyckoff 对称性描述符、组成残差、多头图注意力和
跨元素共享计数预测头。用 `model/decoder=crystal_gnn` 启动独立训练；
设计、消融和评估口径见 [CrystalGNN 说明](docs/crystal_gnn.md)。

使用 `uv run python -m models.run experiment=joint` 可联合训练空间群预测与
CrystalGNN 占位流，共享静态成分编码器。训练、联合评估和仅给定成分的采样命令见
[联合训练说明](docs/joint_training.md)。

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
  --num-samples 20 \
  --batch_size 128 \
  --flow_steps 100 \
  --save_path outputs/flow_samples
~~~

采样步数由采样命令指定，默认 100，不保存到模型配置或训练检查点中。
`sample_wy.py` 使用 `--flow_steps`，`generate_structure.py` 使用 `--flow-steps`；
`sample_and_eval_gwa.sh` 的第四个位置参数为采样步数。

采样过程统一由 `models/sampling.py` 组织：构造条件与 flow 轨迹 → 模型返回
末步 logits → 解码模板。模型不保存采样次数或组成守恒开关。

| 采样参数 | 含义 |
| --- | --- |
| `--num-samples` | 每个公式/空间群条件请求的输出样本数 |
| `--sampling-mode n-shot` | 每个样本运行一条随机 flow 轨迹，再做随机解码 |
| `--sampling-mode top-n` | 每个条件运行一条随机 flow 轨迹，再搜索多个候选 |
| `--sampling-mode greedy` | 每个样本独立初始化，逐步取 decoder 的 argmax，再做确定性解码 |
| `--enforce-composition` / `--no-enforce-composition` | 是否使用组成守恒解码，默认开启；top-n 必须开启 |
| `--fixed-site-beam-size` | 固定 Wyckoff 位点的搜索束宽，默认 `max(256, 8*num_samples)` |

评估模型自身的随机生成效果，使用 `--sampling-mode n-shot --no-enforce-composition`。
该组合保留随机 flow 轨迹，末步按各变量的 categorical 分布抽样，不做组分守恒的
DP/束搜索。化学式条件和元素种类掩码仍然保留，各元素的原子总数由模型自行预测。
一键采样评估脚本也支持该组合，例如使用真实空间群评估 MP20 测试集：

~~~bash
bash scripts/sample_and_eval_gwa.sh \
  <model_path> 20 eval/nshot_no_composition 50 \
  example/input_test_origin.csv data/mp20/test.csv \
  --sampling-mode n-shot --no-enforce-composition
~~~

导出和 GWA 评估保留组分错误及空模板，不筛选或补采样；空模板的 `formula` 为空，
`wyckoff_occupancy` 仅记录空间群编号，作为未命中计入评估。

内部用 `num_trajectories` 表示 flow 轨迹数，和输出样本数区分。
`--num_evals`、`--count_conserving`、`--topn-beam-size` 仍可作为上述参数的旧别名。
CrystalFlow 输入 CSV 中的 `num_evals` 属于外部格式，沿用其原名称。

`--reuse-logits` 只运行解码，不加载模型；未指定的模式、样本数和束宽沿用缓存。
top-n 可以重新指定 `--num-samples`，输出元数据记录实际生效值；n-shot 和 greedy 的轨迹数
已由缓存确定，改变样本数需要重新运行 flow。

解码结果统一为 `DecodingResult(samples, infeasible_graph_indices)`，不原地修改
flow 状态；不可行输入按索引返回并跳过，不拆批重新运行模型。
top-n 保留 decoder 排名；n-shot 和 greedy 的模板频次排序由结构生成流程单独处理。
top-n 是基于末步 logits 的候选搜索，前面的 flow 仍然随机；有限束宽也不保证
得到完整生成分布的精确前 N 名。

greedy 在每一步直接取经过元素掩码的 decoder logits 的 argmax，不使用随机跳转。
最后开启组成约束时，使用无随机扰动的 DP/束搜索选择一个组成守恒模板；关闭约束时
直接逐变量取 argmax。`--num-samples` 对应独立初态的数量，允许重复；当前默认
`flow_source: zeros` 的初态相同，因此同一条件会得到重复结果，通常设为 1。
使用 `uniform` 或 `marginal` 先验时，初态仍有随机性。三个采样入口均支持
`--sampling-mode greedy`，例如：

~~~bash
uv run python scripts/sample_wy.py \
  --model_path <model_path> \
  --formula_file <formula_file> \
  --sampling-mode greedy \
  --num-samples 1 \
  --save_path outputs/greedy_samples
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
- `candidate_rank`
- `decoder_log_score`（有约束解码的末步类别对数分数，替代 `candidate_probability`）
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

组成守恒在采样解码阶段完成（默认开启）。导出步骤解码 Wyckoff 图，并保留每一个采样结果，
不对重复模板去重；`count` 对每一行固定为 `1`。公式使用完整的
conventional-cell 计量，不会自动约分，例如 `Ga4Te4` 不会变成 `GaTe`。

`decoder_log_score` 是给定末步 flow 状态时各类别 log probability 的和，
不代表对所有 flow 轨迹积分后的模板概率；使用对数避免很小的概率下溢。
输出中的 `gpu_flow_time` 和 `cpu_decode_time` 分别记录 flow 与解码耗时。

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
