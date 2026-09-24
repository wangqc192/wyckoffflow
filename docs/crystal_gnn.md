# CrystalGNN：面向 Wyckoff 模板生成的图网络

实现位于 `models/pl_models/crystal_gnn.py`，配置为
`conf/model/decoder/crystal_gnn.yaml`。通过现有 decoder 接口接入
`DiscreteFlowModule`，需要从头训练新网络。

## 设计依据

图中的节点代表 Wyckoff 轨道类型，而不是带三维坐标的原子。因此网络利用
空间群、Wyckoff 对称性、重数和化学组成，不引入不存在的原子距离或键角。

| 改动 | 目的 |
| --- | --- |
| 元素 embedding 与向量计数 embedding 联合编码 | 避免先将每种轨道计数压缩成单个标量；显式表达元素与计数的关系 |
| Wren 的 444 维 `bra-alg-off` 描述符 | 补充可跨空间群共享的晶体学特征，与空间群、位置、自由度、重数 embedding 一起输入 |
| 各元素已分配原子数与剩余原子数 | 让各位点协调满足完整组成；按常规晶胞重数累计，保留负残差以识别过量占据 |
| 多头图注意力与重数关系偏置 | 按接收节点归一化消息；边特征包括重数比、最大公约数比例、自由度和自环标志 |
| 条件归一化、可学习残差缩放、dropout | 将时间和组成注入每层，稳定残差更新并提供可调正则化 |
| 跨元素共享计数预测头 | 针对每个位点—元素组合预测轨道计数，使较少见的正计数类别共享训练信号；仅计算组成中存在的元素 |

原子计数关系为：

```text
当前原子数[e] = Σ位点 multiplicity[位点] × 当前轨道计数[位点, e]
剩余原子数[e] = 目标原子数[e] − 当前原子数[e]
```

固定位置的轨道计数是元素 one-hot，空位为全零；可变位置使用当前 flow 状态的
元素计数向量。剩余原子数是软特征，不用它锁定或排除位点，因此仍可纠正当前
错误的占据。数据中的 `x_0_dof`、`x_inf_dof` 是状态来源，不依赖可能尚未更新的 `x`。

输出继续为 `(zero_logits, inf_logits)`。不在组成中的元素通道是占位值，
调用方必须像现有 `DiscreteFlowModule` 一样应用元素掩码；最终组成守恒由现有
解码器保证。Wren 特征来自项目已有的 `wyckoffflow/aviary` 依赖并保存进检查点。

节点重排等变性与原点变换等价性是不同的性质。本网络没有强制所有等价原点表示
输出相同概率；继续使用数据集的等价表示增强，并按 `scripts/eval_gwa.py`
相同口径评估等价模板命中。

## 启动训练

```bash
uv run python -m models.run model/decoder=crystal_gnn
```

默认 4 层、hidden=256、元素维度 128、8 个注意力头、dropout=0.1。
在默认 100 种元素、最大轨道计数 54 的配置下，可训练参数约 520 万，
原 `wyckoff_gnn` 约 1,677 万；主要节省来自跨元素共享输出头。
其余训练、损失、采样和验证配置继承现有默认值。旧 `gnn.py` 检查点不适用于
这个新架构；请启动独立训练目录，不要用旧检查点作为 `resume_from`。

对照旧网络时保持数据、种子、训练轮数、优化器、损失权重和采样预算一致：

```bash
uv run python -m models.run --multirun \
  model/decoder=wyckoff_gnn,crystal_gnn
```

新网络提供三个独立特征开关，便于逐项消融，例如：

```bash
uv run python -m models.run \
  model/decoder=crystal_gnn \
  model.decoder.use_composition_residual=false
```

另外两个开关是 `use_symmetry_features`、`use_edge_bias`。
关闭组成残差时仍保留目标组成与节点自身当前占据，只移除显式全局分配量和残差。
可通过 `model.decoder.dropout=0.0` 检查正则化的影响。

## 如何判断效果

主要比较固定生成配置下验证集的 GWA@1、GWA@20（包含等价模板），并以
`best_gwa.ckpt` 为生成指标选模结果。辅助观察非空位点、计数 ≥ 2 类别准确率
和验证 CE。已有检查点诊断表明，CE 变差可以与 GWA 改善同时发生，因此不能
仅凭 CE 宣称新网络更好。

测试覆盖 230 个空间群、无固定位置的图、节点重排、批次间隔离、常规晶胞原子
计数、过量占据、时间端点、组成绝对数量、混合精度反向传播、采样与守恒解码、
检查点恢复及消融开关：

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  uv run --no-sync pytest -q \
  tests/test_crystal_gnn.py tests/test_training_config.py tests/test_checkpoint.py
```

本次 CPU 验证共通过 21 项相关测试，包括 bfloat16 反向传播。在默认网络配置下，
用 32 个真实 MP20 训练材料做 20 步优化，固定噪声评估的 loss 从 3.5108 降至
1.5703；3 步 flow 后的 8 个解码样本均满足目标组成。记录保存在
`artifacts/2026-09-24_crystal_gnn/smoke.json`。这是小批次可训练性检查，
不是验证集性能对照；完整重训后的 GWA 提升幅度尚未验证。
