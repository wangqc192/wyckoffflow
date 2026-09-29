# 从组成学习空间群

`models/pl_models/chemical_sg.py` 提供独立的神经网络空间群分类器，可与已有占位流 checkpoint 配合采样。空间群概率来自网络参数；不使用空间群频率表、相似组成检索或经验先验融合。加载器会拒绝带有这类先验的历史实验 checkpoint。

输入是常规晶胞的完整元素计数。固定的元素描述符（matscholar200、cgcnn92）作为化学输入特征，经可学习的残差网络输出 230 个群的分数。可选条件分支学习原子计数与化学特征的关系；可选原型辅助分类只在训练时读取结构标签，推理不需要真实结构、原型或空间群。

`models/pl_models/neural_fusion.py` 支持由折外预测训练的融合网络。同一完整组成的训练样本固定进入同一折，融合网络没有用专家在其训练样本上的记忆分数拟合。还可汇总网络预测的原型类别概率，按类别所属空间群参与最终预测；原型到空间群的映射是类别定义，不是经验频率表。推理不读取真实原型。`ChemicalSpaceGroupPredictor` 的 `prototype_weight` 可在训练时直接优化这两路概率的组合。

默认保留原有的组成可行性 mask，它根据公开 Wyckoff 重数与输入计数排除无法满足组成的群，未使用数据集类别频率。`use_feasibility=False` 可训练完全不读取该 mask 的对照模型。实验报告必须区分这两种设置。

采样示例（总共至多 5 × 4 = 20 个模板）：

```bash
uv run --no-sync python -m scripts.sample_wy \
  --model_path outputs/2026-09-24/09-56-08_discrete_flow/checkpoints/epoch_0299.ckpt \
  --space-group-model-path artifacts/2026-09-24_sg_improvement/candidate_neural_hierarchy.pt \
  --formula Li4O4 --space-group-top-k 5 \
  --num-samples 4 --sampling-mode top-n \
  --save_path outputs/chemical_sg_samples
```

`--num-samples` 是每个空间群的数量。GWA 匹配使用 `scripts/eval_gwa.py` 的等价模板约定，不能用模板字符串相等替代。

2026-09-24 实验脚本和完整记录位于 `artifacts/2026-09-24_sg_improvement/`。当前冻结候选 `candidate_neural_hierarchy.pt` 使用原 MP20 的 27,136 条训练数据，在 9,047 条验证数据上经实际 FP32 加载器复核为 Top-1 75.94%、Top-5 93.57%；结果和校验和见 `candidate_neural_hierarchy_verified.json`，尚未评估此候选的独立测试集。此前冻结的 `candidate_neural_v2.pt` 的独立测试 Top-5 为 92.99%，不能将该成绩归给新模型。

`candidate_neural_prototype_fusion.pt` 使用五折共 55 个分支，在同一完整验证集上为 Top-5 93.58%，只多命中 1 条且推理更重，因此采样示例仍使用较轻候选。第一轮完整端到端 GWA@20 为 89.74%（8119/9047），经 `gwa_full_neural/verified.json` 独立审计确认；该轮使用的是 `candidate_neural_fusion.pt` 与冻结的 epoch 398 占位网络。

额外监督数据实验单独保存在 `neural_external_*` 等目录，只使用 MPTS 的训练划分，并排除全部 MP20 train/val/test 材料 ID 和约分组成。当前冻结候选未使用这些额外样本。`external_train_preparation.json`、`external_material_ids.json` 记录标注口径和各实验材料清单。

端到端方法试验可内部提出更多群和模板，再按神经网络空间群概率与占位网络分数排序，最终总预算仍为 20；上面的采样 CLI 示例是固定 5 × 4 分配，不等同于这套全局排序流程。`full_gwa.py`、`audit_full_gwa.py` 提供完整验证和独立审计入口，检查组成守恒、等价模板、目标覆盖及预算。

Top-5 95% 和端到端 GWA@20 90% 仍是优化目标，不能将 Top-10、真实空间群条件下的 GWA 或小样本结果当作达标证据。`excluded_prior_results/` 为不符合当前要求的历史结果。
