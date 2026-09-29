# 空间群与 Wyckoff 占位联合训练

`experiment=joint` 使用 CrystalGNN，占位流与空间群分类共享一份静态成分编码器。

```text
完整晶胞组成 → 静态成分编码器 → 整体成分特征 → 空间群分类头
                            ↘ 元素与整体特征 → CrystalGNN 占位流
                                                ↑
                                  当前占位、动态成分余量、空间群、时间
```

共享编码器包含元素 embedding、静态元素特征网络和整体成分网络。动态余量、
时间编码、图注意力和占位输出头属于占位分支。空间群头只读取静态成分特征及
候选空间群的公开 Wyckoff 目录，不读取真实群的图或占位。两项损失共同更新
共享编码器，占位流训练使用真实空间群。

```text
L = sg_loss_weight × L_sg + flow_loss_weight × L_flow
```

空间群概率使用现有成分可行性 mask；最终占位仍由现有计数守恒解码器约束。
输入需要使用常规晶胞的完整组成，不能将约化式直接当作完整晶胞计数。

## 训练

```bash
uv run python -m models.run experiment=joint
```

默认使用 CrystalGNN 的 4 层、256 维隐藏特征、128 维元素 embedding、8 个注意力头。
`model.sg_head.mlp_hidden_layers=3` 表示 3 个隐藏层，另有 1 个 Linear 输出层。
空间群直接分类 MLP 和组成—空间群兼容性评分 MLP 的每个隐藏层使用
`Linear → LayerNorm → SiLU → Dropout(0.1)`，最终输出层仍为 Linear。
对应配置为 `model.sg_head.layer_norm=true` 和 `model.sg_head.dropout=0.1`；
共享成分编码器与候选群 Wyckoff 位置编码器不使用这两个设置。
可分别关闭以做消融：

```bash
uv run python -m models.run experiment=joint \
  model.sg_head.layer_norm=false model.sg_head.dropout=0.0
```

默认空间群权重为 0.1，占位流权重为 1.0，即
`L = 0.1 × L_sg + L_flow`。可以分别配置，例如将空间群权重恢复为 1 做对照：

```bash
uv run python -m models.run experiment=joint \
  model.sg_loss_weight=1.0 model.flow_loss_weight=1.0
```

`model.flow_encoder_grad_scale` 单独控制 Flow 损失传回共享组成编码器的梯度，
默认 1.0 保持原行为。设为 0.1 时保留 10%，设为 0 时组成编码器只由 SG
损失更新；两种设置下 SG 头和 Flow 解码器仍同时训练。缩放覆盖共享的元素
embedding、元素 token 和 pooled 特征，不改变特征前向值、数值损失权重或
Flow 解码器自身收到的梯度。此选项用于检验共享训练的影响，不能预先认定
Flow 梯度有害。旧 checkpoint 缺少此项时默认使用 1.0。

```bash
uv run python -m models.run experiment=joint model.flow_encoder_grad_scale=0.1
```

默认使用原优化器配置（当前全局 weight decay 为 0）。若要测试整个 Joint
模型的权重衰减，并排除 bias、LayerNorm 等一维参数：

```bash
uv run python -m models.run experiment=joint \
  optim.weight_decay=0.01 model.decay_matrix_weights_only=true
```

此选项启用独立参数分组：所有维度不少于 2 的可训练参数使用
`optim.weight_decay`，覆盖共享成分编码器、空间群分支和 Flow 分支的
Linear 权重、embedding 等矩阵参数；bias、LayerNorm 等一维参数的衰减为 0。
该分组设置会保存在 checkpoint 中，续训时应保持
`model.decay_matrix_weights_only` 与原训练一致。旧 checkpoint 缺少此项时
默认不启用分组，保留原优化器结构。

训练记录 `sg_loss`、`flow_loss`、`loss`、`spg_top1` 和 `spg_top5`。
其中 `sg_loss`、`flow_loss` 保留未加权值，`loss` 为加权总损失。
旧的独立模型配置及其 checkpoint 仍按原架构加载；不能将独立模型的
checkpoint 用作联合模型的 `resume_from`。
旧 Joint checkpoint 若未保存 `sg_head.layer_norm`、`sg_head.dropout`，加载时
分别按 `false`、`0.0` 处理，保持原模型结构和预测。新增 LayerNorm 的 Joint
需要开启新的训练；续训旧 Joint 时使用与其匹配的旧配置，或显式设置
`model.sg_head.layer_norm=false model.sg_head.dropout=0.0`。
续训时的损失权重也应按原 checkpoint 的配置设置。

## 验证

每次周期重建先使用真实空间群评估占位，再用预测群评估完整任务。
默认联合评估取 5 个可行群，在总共 20 个模板的预算内分配，每群 4 个。
可行群少于 5 个时，在选中的群之间重新分配预算；余数优先分给排名靠前的群。
重复模板占用预算，匹配按 `eval_gwa.py` 的等价模板规则计算。

| 指标 | 含义 |
| --- | --- |
| `val/spg_top1`、`val/spg_top5` | 空间群分类命中率 |
| `val/gwa_top20` | 真实空间群条件下的模板命中率 |
| `val/joint_gwa_top20` | 预测空间群后的完整模板命中率 |
| `val/joint_gwa_top1` | 最高概率空间群的第一个生成模板是否命中 |

联合结果写入 `reconstruction/epoch_XXXX/joint/`，最佳联合命中率对应
`checkpoints/best_joint_gwa_epoch_XXXX.ckpt`。候选按空间群概率、群内采样顺序排列，
没有将 flow 末步分数作为跨群联合概率排序。

相关设置为 `train.reconstruction.predicted_space_groups` 和
`train.reconstruction.num_samples`；前者设为 0 时只运行真实群条件评估。

## 采样

用一个联合 checkpoint 从成分预测空间群并生成占位：

```bash
uv run python -m scripts.sample_wy \
  --model_path <joint_run_or_checkpoint> \
  --formula Li4O4 \
  --space-group-top-k 5 \
  --num-samples 4 \
  --flow_steps 50 \
  --save_path outputs/joint_samples
```

采样 CLI 的 `--num-samples` 是每个选中群的样本数，因此此例最多生成 20 个模板；
可行群不足时 CLI 不补齐群数。`--space-group-top-k` 与显式指定空间群互斥。
也可以使用原有的 `--space_group 194` 进行给定群的占位采样。
生成文件兼容现有模板导出和下游结构生成流程。
