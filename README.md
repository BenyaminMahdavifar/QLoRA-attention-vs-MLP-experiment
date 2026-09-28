# QLoRA Attention vs MLP Layer Experiment

> An experiment on **where adaptation happens inside a Transformer**.

## Core Question

When we fine-tune a language model, **which part of the Transformer actually needs to change?**

A Transformer layer contains different computational pathways. Two of the most important are:

- **Self-Attention** — controls how token representations interact with each other.
- **MLP / Feed-Forward Network** — transforms the representation of each token after contextual information has been gathered.

Instead of adapting the whole model, this experiment isolates these pathways and asks:

> **What happens when LoRA is allowed to modify only Attention or only MLP layers?**

The goal is not to claim that one pathway is universally better. The goal is to understand **where task adaptation is taking place**.

---

## Experimental Design

To make the comparison meaningful, the two experiments keep the main setup fixed and change only the LoRA target modules.

| Component | Attention-only | MLP-only |
|---|---:|---:|
| Base model | Qwen3-0.6B | Qwen3-0.6B |
| Adaptation | LoRA / PEFT | LoRA / PEFT |
| Dataset | GAIA | GAIA |
| Attention modules | Trainable | Frozen |
| MLP modules | Frozen | Trainable |
| Main variable | Attention adaptation | MLP adaptation |

This is essentially an **architectural ablation experiment**: restrict learning to one pathway and observe what changes.

---

# Experiment 1 — Attention Only

In the attention-only experiment, LoRA is applied to the four main attention projections:

```text
model.layers.*.self_attn.q_proj
model.layers.*.self_attn.k_proj
model.layers.*.self_attn.v_proj
model.layers.*.self_attn.o_proj
```

The MLP pathway remains frozen.

### What is being tested?

Conceptually, this experiment asks:

> **How much task adaptation can be represented by changing the mechanism that determines how tokens interact?**

The model is allowed to modify:

- Query projections
- Key projections
- Value projections
- Output projection

but not the MLP projections.

### Adapter

[Qwen3-0.6B Attention-only LoRA Adapter](https://huggingface.co/MLBenyamin/Qwen3-0.6B-attention-layer-EX-gaia-lora-adapter)

---

# Experiment 2 — MLP Only

In the MLP-only experiment, LoRA is applied to:

```text
model.layers.*.mlp.gate_proj
model.layers.*.mlp.up_proj
model.layers.*.mlp.down_proj
```

The attention pathway remains frozen.

### What is being tested?

Conceptually, this experiment asks:

> **How much task adaptation can be represented by changing the transformation of token representations after contextual information has been gathered?**

The model is allowed to modify the MLP pathway while leaving attention unchanged.

### Adapter

[Qwen3-0.6B MLP-only LoRA Adapter](https://huggingface.co/MLBenyamin/Qwen3-0.6B-mlp-layer-EX-gaia-lora-adapter)

---

# Why Separate Attention and MLP?

A Transformer layer is not a single operation.

At a high level:

```text
Token representations
        │
        ▼
   Self-Attention
        │
        ▼
Contextual representations
        │
        ▼
       MLP
        │
        ▼
Updated representations
```

This creates a useful conceptual separation:

### Attention

Primarily answers:

> **Which information should interact with which other information?**

### MLP

Primarily answers:

> **How should the representation be transformed once the relevant information is available?**

These are not completely independent functions, and this distinction should not be interpreted as a strict biological or functional separation. The experiment uses the distinction as a practical way to isolate architectural pathways.

---

# Results

The following results were obtained from the experiments.

| Experiment | Base Eval Loss | Fine-tuned Eval Loss | Base Overlap Match Accuracy | Fine-tuned Overlap Match Accuracy |
|---|---:|---:|---:|---:|
| Attention-only | 3.22 | 2.96 | 6.67% | 0.00% |
| MLP-only | 3.20 | 2.93 | 6.67% | 0.00% |

## An Important Distinction

The decrease in evaluation loss should **not** automatically be interpreted as improved task-level performance.

Loss and task accuracy measure different things.

A model can optimize its training/evaluation objective while still failing a particular exact-match or generation-based evaluation.

This is one of the important observations of the experiment:

> **Optimization behavior and task-level behavior are not necessarily the same thing.**

The results therefore need to be interpreted as an architectural experiment rather than as evidence that one pathway is universally superior.

---

# Target Modules

The relevant Transformer modules can be grouped as follows:

| Module | Pathway | Used in |
|---|---|---|
| `q_proj` | Self-Attention | Attention-only |
| `k_proj` | Self-Attention | Attention-only |
| `v_proj` | Self-Attention | Attention-only |
| `o_proj` | Self-Attention | Attention-only |
| `q_norm` | Attention normalization | Neither |
| `k_norm` | Attention normalization | Neither |
| `gate_proj` | MLP | MLP-only |
| `up_proj` | MLP | MLP-only |
| `down_proj` | MLP | MLP-only |

The normalization layers were intentionally not included in the two isolated experiments.

---

# Reproducing the Experiment

Install the main dependencies:

```bash
pip install torch transformers peft datasets accelerate bitsandbytes
```

The experiment uses:

- **Qwen3-0.6B**
- **LoRA / PEFT**
- **QLoRA-style quantized fine-tuning setup**
- **GAIA**
- Isolated Attention or MLP target modules

---

# Adapters

### Attention-only

[MLBenyamin/Qwen3-0.6B-attention-layer-EX-gaia-lora-adapter](https://huggingface.co/MLBenyamin/Qwen3-0.6B-attention-layer-EX-gaia-lora-adapter)

### MLP-only

[MLBenyamin/Qwen3-0.6B-mlp-layer-EX-gaia-lora-adapter](https://huggingface.co/MLBenyamin/Qwen3-0.6B-mlp-layer-EX-gaia-lora-adapter)

---

# What This Experiment Is Really Trying to Understand

The central question is not:

> "Is Attention better than MLP?"

That would require a much broader and more controlled experimental design.

The more useful question is:

> **If we give the optimizer access to only one computational pathway, what kind of adaptation can that pathway represent?**

This reframes LoRA from simply being a parameter-efficient fine-tuning technique into an experimental tool for studying **where model adaptation occurs**.

---

# Limitations

This experiment has several important limitations.

### 1. Small base model

The experiment uses Qwen3-0.6B. Results may change for larger architectures.

### 2. Single task setting

The evaluation is based on the selected GAIA setup. Different tasks may require different forms of adaptation.

### 3. LoRA configuration matters

Rank, alpha, dropout, learning rate, quantization, and other training choices can affect the result.

### 4. Architecture matters

The functional roles of Attention and MLP are architecture-dependent. The same conclusions should not automatically be generalized to every Transformer variant.

### 5. Evaluation methodology matters

Loss, exact-match accuracy, generation quality, and other metrics can tell different stories.

Therefore, these results should be treated as **evidence from a controlled experiment**, not as a universal statement about Transformer architecture.

---

# Future Experiments

The natural next step is to expand the ablation matrix.

### Attention + MLP

Allow both pathways to adapt and compare against the isolated experiments.

### Layer-wise adaptation

Instead of adapting all Attention or all MLP layers:

- early layers only
- middle layers only
- late layers only
- individual layer sweeps

### Projection-level ablations

For Attention:

- Q only
- K only
- V only
- O only
- Q + K
- Q + V
- K + V

For MLP:

- gate only
- up only
- down only
- gate + up
- up + down

### LoRA rank

Test whether different pathways require different adaptation capacity:

```text
rank = 1
rank = 2
rank = 4
rank = 8
rank = 16
...
```

### Parameter efficiency

Compare:

```text
trainable parameters
        vs.
performance
        vs.
evaluation loss
```

This can help answer whether a pathway provides more useful adaptation per trainable parameter.

---

# Research Direction

A broader version of this experiment can be viewed as:

```text
Transformer
    │
    ├── Attention
    │     ├── Q
    │     ├── K
    │     ├── V
    │     └── O
    │
    └── MLP
          ├── Gate
          ├── Up
          └── Down
```

Instead of asking only:

> **How well can the model be fine-tuned?**

we can ask:

> **Where does the model store the changes required for a new task?**

That question motivates the experiments in this repository.

---

# References

- [Qwen](https://github.com/QwenLM/Qwen3)
- [Hugging Face PEFT](https://github.com/huggingface/peft)
- [GAIA Benchmark](https://huggingface.co/gaia-benchmark)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

---

# License

This repository is intended for research and experimentation. Please check the licenses of the underlying model, datasets, and libraries before redistribution or commercial use.
