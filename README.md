
# SMILES Block Diffusion – Minimal Codebase (v4)

> **这版修复点**：彻底清掉所有 `...` 占位/截断；示例数据不再含省略号；Tokenizer/训练/采样/配置/测试全量实现；增加 `pyproject.toml` 便于 `pip install -e .`。  
> 训练目标=**MDLM（SUBS）** 的简化 NELBO，只在**被遮蔽位**计交叉熵并乘 \( \alpha'(t)/(1-\alpha_t) \)；采样**never‑remask** + **前缀冻结**（论文附录 B.3，式(19)）。 fileciteturn0file0

## 目录
```
.
├─ configs/default.yaml
├─ examples/
│  └─ toy_data/         # 64 长度玩具样本（无省略号）
├─ src/smiles_bd/
│  ├─ tokenizer_smiles.py
│  ├─ schedule.py
│  ├─ model.py
│  ├─ diffusion.py
│  ├─ data.py
│  ├─ config.py
│  ├─ utils.py
│  ├─ train.py
│  └─ sample.py
├─ tests/
│  ├─ test_tokenizer.py
│  ├─ test_dataset.py
│  ├─ test_training_and_sampling.py
│  └─ test_train_loop_smoke.py
├─ pyproject.toml
├─ requirements.txt
└─ pytest.ini
```

## 安装
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .   # 可选，支持 `python -m smiles_bd.train`
```

## 数据格式（与你的前提完全一致）
- 每行：`SMI_A[EOS]SMI_A'[EOS]...`，尾部用 `[PAD]` 补到 **1024**；
- **不自动注入**任何 special token；**不 wrap/不切块**；
- `attention_mask`==1 仅在非 `[PAD]` 处。

## 训练
```bash
python -m smiles_bd.train --config ./configs/default.yaml \
  --override paths.train_path=/abs/train.txt \
             paths.valid_path=/abs/valid.txt \
             paths.vocab_path=/abs/vocab.txt \
             model.max_len=1024 train.epochs=10 train.batch_size=8
```
- 遮罩率：`U[β,ω]`（默认 `0.3~0.8`），降低梯度方差（§5）；  
- 目标：只在被遮蔽位计交叉熵，并乘 \( \alpha′(t)/(1-α_t) \)（附录 B.3，式(19)）。 fileciteturn0file0  
- Checkpoint 会附带 `meta`（形状等），采样端自动匹配。

## 采样（前缀冻结 + `[EOS]` 切分）
```bash
python -m smiles_bd.sample \
  --ckpt ./checkpoints/model.pt \
  --config ./configs/default.yaml \
  --override sample.steps=32 sample.top_p=0.9 \
  --prefix "C1=CC=CC=C1"
```
- 将前缀放到最前并冻结；其后初始化为 `[MASK]`，每步揭示一部分未揭示位；**禁止**生成 `[MASK]/[PAD]`；按 `[EOS]` 切候选；  
- **never‑remask** ⇒ NFEs 上界为序列长度（§6.2）。 fileciteturn0file0

## 默认配置（`configs/default.yaml`）
- `model`: `max_len=1024, d_model=512, n_heads=8, n_layers=8, dropout=0.1`
- `train`: `lr=3e-4, beta=0.3, omega=0.8, grad_clip=1.0`
- `sample`: `steps=24, top_p=0.9`

## 设计取舍（为什么适配你的任务）
- **整段（block_size=length）MDLM**：退化为“单块扩散”，不需块掩码/专用内核；训练与推理更稳（§3.2）。  
- **SUBS（零掩码概率 + carry‑over unmasking）**：训练只在 mask 位；采样一旦揭示不回退（附录 B.3）。  
- **Clipped schedule**：避免极端遮蔽率，实证上与 PPL 改善相关（§5 表2）。  
以上均直接对应原论文推导与实验。 fileciteturn0file0

## 质量自检（tests/ 已覆盖）
- Tokenizer：`[EOS]/[PAD]/[MASK]` 映射正确；`decode` 丢 PAD；
- Dataset：`attention_mask.sum()==非PAD数`；
- 训练步：只在遮蔽位计损；`num_masked>0`；loss 有限；
- 采样：前缀不改写；后缀不生成 PAD；可按 `[EOS]` 切分候选；
- Smoke：10 步内 loss 下降（或不升），保障闭环可跑。

---

**如需把示例 L=64 替换为真实 1024，只需：**  
- 把 `configs/default.yaml` 的路径指向你的 `train.txt/valid.txt/vocab.txt`；  
- 将 `model.max_len=1024`；其余保持即可。
