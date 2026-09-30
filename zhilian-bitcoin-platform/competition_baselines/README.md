# 六个新增基线

本目录只包含以下六个方法的统一协议复现：

- MDST-GNN
- TFGAT-DCPLU
- Elliptic++-HGT
- FG-EGCN
- GPN
- NSGCN-LSTM

不包含 HistGradientBoosting 和 Bit-CHetG。时间划分固定为训练 1–30、验证
31–35、测试 36–49；未知标签节点仅作为图上下文，不进入监督损失或指标。
阈值和早停只使用验证集，测试集不参与模型选择。

运行全部五种子实验：

```powershell
python run.py --models all --force
python summarize.py
python validate.py
```

快速检查：

```powershell
python run.py --models mdst_gnn --seeds 20260806 --device cpu --smoke --force
python -m pytest tests -q
```

完整结果写入 `../artifacts/competition_baselines_5seed/`。
