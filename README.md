
# SMILES Block Diffusion – Minimal Codebase (v4)

> 目标：把同指纹/同性质簇内的多条 SMILES 串接成 **定长 1024** 的长序列，在**整段**上用**离散扩散（masked diffusion, SUBS）**做“**掩码→还原**”学习；采样时把 A 放在最前作为**前缀并冻结**，对后续的 `[MASK]` 做扩散还原，并按 `[EOS]` 切出若干候选 SMILES。

本仓库是从 *Block Diffusion* 思想中**抽取并重构**的极简实现，专为上面的化学任务而裁剪：
- 仅保留 **整段（block_size = model.length）** 的 masked diffusion 训练与**前缀填充**采样；
- **不注入**任何特殊符号，也**不 wrap/切块**；数据应当已经按 `smiles-A [EOS] smiles-A' [EOS] ... [PAD] ...` 处理为 **1024** 长度；
- 直接使用你提供的 `vocab.txt`（SMILES 词表）与 regex 分词器。

与原论文/代码库的关系与取舍
----------------------------

- 我们遵循 *Block Diffusion Language Models (BD3-LMs)* 的**整段**设定（把 block size 设为上下文长度），训练目标采用其在 **附录 B.3** 给出的**简化 NELBO**：只对被掩码的位置做交叉熵，并使用随时间的权重（等价于 α'(t)/(1-α_t) 的比例因子）以贴近理论推导。参见论文 *Eq. (19)* 与算法 1/2 的思路（训练/采样），以及 **“clipped schedule”** 降低方差的建议（Sec. 5, Tab. 2 & Fig. 2）。fileciteturn0file0  
- 我们移除了与本任务无关的特性（多数据集封装、AR/SEDD 基线、Hydra/Lightning、大量回调、KV/FlexAttention 优化等）。在**整段**设定下，专门的块级注意力掩码退化为**标准自注意力**，因此不再需要复杂 kernel（论文 *Suppl. B.6–B.7*）fileciteturn0file0。
- 采样使用**前缀冻结 + 迭代解码**：一次生成若干高置信 token，且**不会再重掩码**（SUBS 参数化“carry‑over unmasking”），与文献在 masked diffusion 中“已揭示 token 不再被重新遮蔽”的假设一致（附录 B.3）。fileciteturn0file0
- 训练时使用**Clipped Linear** 噪声日程（`mask_rate ~ U[β, ω]`），依据论文对**训练方差最小化**的讨论（Sec. 5.1–5.3），默认 `β=0.3, ω=0.8`，可按需要调参。fileciteturn0file0

> 这份代码是**工程化的最小骨架**：Torch 原生实现，只有 4 个核心文件（Tokenizer / Schedule / Transformer Denoiser / Diffusion 训练&采样），便于你直接集成到现有化学工作流中。

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
- 目标：只在被遮蔽位计交叉熵，并乘 \( \alpha′(t)/(1-α_t) \)（附录 B.3，式(19)）。
- Checkpoint 会附带 `meta`（形状等），采样端自动匹配。

> **注意**：为了保证只在真实 token 上学习，loss 仅在 `[MASK]` 位置计算；`[PAD]` 位置永远不参与。

## 采样（前缀冻结 + `[EOS]` 切分）
```bash
python -m smiles_bd.sample \
  --ckpt ./checkpoints/model.pt \
  --config ./configs/default.yaml \
  --override sample.steps=32 sample.top_p=0.9 \
  --prefix "C1=CC=CC=C1"
```
- 将前缀放到最前并冻结；其后初始化为 `[MASK]`，每步揭示一部分未揭示位；**禁止**生成 `[MASK]/[PAD]`；按 `[EOS]` 切候选；  
- **never‑remask** ⇒ NFEs 上界为序列长度（§6.2）。 
- 迭代式解码：每一步在当前高置信位置解码一批 token（**不会再被重掩码**），直到没有 `[MASK]` 或达到步数。
- 生成序列会按 `[EOS]` **切分**出多个候选 SMILES（来自与 A 同簇的长序列后缀），用于后续性质筛选。



- `test_tokenizer.py`：验证 regex 分词 & vocab 装载；
- `test_dataset.py`：验证样本加载为固定长度张量，且 attention mask 仅在非 `[PAD]` 上为 1；
- `test_training_and_sampling.py`：跑一个最小训练步并做前缀采样，检查前缀未被篡改且能按 `[EOS]` 切分。


## 默认配置（`configs/default.yaml`）
- `model`: `max_len=1024, d_model=512, n_heads=8, n_layers=8, dropout=0.1`
- `train`: `lr=3e-4, beta=0.3, omega=0.8, grad_clip=1.0`
- `sample`: `steps=24, top_p=0.9`

## 设计取舍（为什么适配你的任务）
- **整段（block_size=length）MDLM**：退化为“单块扩散”，不需块掩码/专用内核；训练与推理更稳（§3.2）。  
- **SUBS（零掩码概率 + carry‑over unmasking）**：训练只在 mask 位；采样一旦揭示不回退（附录 B.3）。  
- **Clipped schedule**：避免极端遮蔽率，实证上与 PPL 改善相关（§5 表2）。  
以上均直接对应原论文推导与实验。

1. **整段扩散（block_size = length）**  
   你的目标是在**整段 1024** token 上进行“掩码→还原”，而非分块。BD3-LM 在这种设定下退化为单块扩散，注意力掩码也退化成常规自注意力（无需专门块掩码/缓存），训练和推理由此**更简单可靠**（论文 Sec. 3.2 & Suppl. B.6）。
2. **SUBS 参数化 + 不重掩码**  
   这使得训练和采样语义一致：学的是“在上下文条件下覆盖 `[MASK]` 的真实 token”，采样时一旦揭示就固定，贴合你的**前缀冻结 + 后缀填充**需求（附录 B.3）。
3. **Clipped Schedule 降低方差**  
   依据论文第 5 节的分析，遮蔽比例避免极端值能显著降低 NELBO 的方差，并与困惑度相关（Tab. 2, Fig. 2）；这里默认 `U[0.3, 0.8]`，并允许你按数据特征微调。
4. **数据面向工程**  
   你已经把多个同簇 SMILES 串接为 1024，并用 `[EOS]`/`[PAD]` 管理边界；本实现**严格遵守**“不再注入特殊符号 & 不 wrap”的约束，避免破坏你的显式边界设计。


## 测试
```bash
pytest -q
```

- Tokenizer：`[EOS]/[PAD]/[MASK]` 映射正确；`decode` 丢 PAD；
- Dataset：`attention_mask.sum()==非PAD数`；
- 训练步：只在遮蔽位计损；`num_masked>0`；loss 有限；
- 采样：前缀不改写；后缀不生成 PAD；可按 `[EOS]` 切分候选；
- Smoke：10 步内 loss 下降（或不升），保障闭环可跑。

---

