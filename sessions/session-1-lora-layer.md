# Session 1: a LoRA layer, written by you

Time: about 2.5h. Code: `lora.py` and `tests/test_lora.py`.

By the end you have a `LoRALinear` module and an `apply_lora` function. Both are tested, and GPT-2 runs with LoRA inside its attention layers.

## 0. Before you start (15 min)

1. Set up the environment and packages (see `GUIDE.md`).
2. Load `openai-community/gpt2` with `transformers` and print the modules of the first block:
   ```python
   for name, module in model.transformer.h[0].named_modules():
       print(name, type(module).__name__)
   ```
3. Look at `attn.c_attn`.
   - What type is it?
   - What is the shape of its `.weight`?
   - Compare with what `nn.Linear(768, 2304)` would store.

You should find that GPT-2 uses a `Conv1D` layer. Despite the name, it is a plain linear layer, but it stores its weight as **(in, out)** = (768, 2304), while `nn.Linear` stores **(out, in)**. Its forward is `x @ W + b`. Read the class in the reading list (item 7, about 20 lines). This detail matters later, so understand it now.

## 1. Read first (30–40 min)

Read the **explanations**. Skip the code in the tutorials until you've written your own version.

1. **The LoRA paper** (Hu et al., 2021): <https://arxiv.org/abs/2106.09685>
   - Read the abstract, section 1 and section 4 (the method, and how it applies to Transformers).
   - Skim sections 7.1–7.2: which weight matrices to adapt, and what rank is needed.
   - Questions to answer for yourself:
     - Why is B initialised to zero?
     - What does alpha / r do?
     - Why is there no extra inference cost?
2. **Hugging Face PEFT, LoRA conceptual guide:** a short, practical overview of the knobs (r, alpha, target modules, merging). <https://huggingface.co/docs/peft/main/en/conceptual_guides/lora>
3. **Sebastian Raschka, *Build a Large Language Model (From Scratch)*, Appendix E** (the notebook in the book's repo). A clear, step-by-step build of a LoRA layer in plain PyTorch. **Read the text; don't read the code cells yet.** <https://github.com/rasbt/LLMs-from-scratch/tree/main/appendix-E>
4. **Sebastian Raschka, "Improving LoRA: Implementing Weight-Decomposed Low-Rank Adaptation (DoRA) from Scratch."** Its first half implements LoRA from scratch with diagrams. DoRA is a bonus for later. <https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch>

**Optional:** why low rank works at all.

5. **Aghajanyan et al., 2020, "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning."** Fine-tuning needs far fewer directions than the model has parameters. <https://arxiv.org/abs/2012.13255>

## 2. Write `LoRALinear` (45 min)

**What it does:** it wraps an existing layer `base` (either `nn.Linear` or GPT-2's `Conv1D`). It freezes `base` and adds two new trainable matrices.

**Things to decide yourself:**
- **Shapes:** pick a convention and stick to it. The usual one is `A: (r, in_features)` and `B: (out_features, r)`.
- **How to read `in_features` and `out_features`:** `nn.Linear` has them as attributes. For `Conv1D`, get them from the weight shape, and remember it's (in, out).
- **Initialisation:** A small and random (the paper uses Gaussian; the reference code uses Kaiming-uniform), B all zeros.
- **Scaling:** `alpha / r`, stored once.
- **Forward:** `base(x) + scaling * (x @ A.T @ B.T)`.
  - Notice that the LoRA path never touches `base`'s weight. That's why the same wrapper works for `Linear` and `Conv1D`.
  - The weight layout only matters when you **merge** (session 3).
- **Freezing:** set `requires_grad = False` on the base parameters only. New `nn.Parameter`s are trainable by default.
- **Optional:** dropout on `x` in the LoRA path only (the paper's `lora_dropout`). Start with 0.

**Small things that cause bugs:**
- Use `nn.Parameter`, not plain tensors, or the optimizer never sees A and B.
- Create A and B with the same dtype and device as `base.weight`.
- Don't freeze A and B by accident with a blanket "freeze everything" loop that runs *after* you wrap.

## 3. Tests (30 min)

Write these with `pytest`, using a small random `nn.Linear(16, 32)` so they run in milliseconds.

1. **Starts identical:** for random `x`, `LoRALinear(base)(x)` equals `base(x)` exactly. Why exactly and not just approximately?
2. **Only LoRA trains:** the names of the parameters with `requires_grad=True` are exactly your A and B.
3. **Parameter count:** trainable count = `r × (in + out)`.
4. **Gradients, the interesting one.** Do one forward and backward pass with any loss. **Before you run it, predict:** which of A and B gets a non-zero gradient on the very first step? Then check.
   - Hint: B = 0, and the gradient of A goes through B.
   - This is why one of the two must start at zero but not both.
5. **It learns:**
   - Make a target that is `base(x)` plus a fixed random rank-2 update.
   - Train only the LoRA parameters with Adam for a few hundred steps, with r = 4.
   - The MSE should drop close to zero. With r = 1 it shouldn't get as low. Why?

## 4. Insert it into GPT-2 (30 min)

Write `apply_lora(model, targets=("c_attn",), r=8, alpha=16)`:

1. Freeze every parameter in the model.
2. Walk `model.named_modules()`. For every module whose name ends with a target name, replace it with `LoRALinear(module, r, alpha)`.
   - To replace a submodule, find its **parent** module and use `setattr(parent, child_name, new_module)`. You can't change a module while you're iterating over `named_modules()`, so collect the names first.
3. Return the model.

**Done when:**
- **Same logits:** with `model.eval()`, the logits for a fixed input are identical before and after `apply_lora`.
- **Your hand calculation matches:** the trainable count for `c_attn` with r = 8 is 12 blocks × 8 × (768 + 2304) = **294,912**. That's about 0.24% of GPT-2's 124,439,808 parameters.
- **Your wrapper is in place:** `print(model.transformer.h[0].attn)` shows your `LoRALinear` around `c_attn`.
- **Gradients reach LoRA:** one training step on a sentence (language-model loss) gives gradients on the LoRA parameters and on nothing else.

**Stretch, if you have time:** add `c_proj` (attention output) and the MLP layers to `targets`, and predict the new parameter count before checking it.

## 5. After you're done: compare with the reference code (15 min)

Now read how others wrote it, and list the differences from your version:

6. **microsoft/LoRA, `loralib/layers.py`:** the paper authors' implementation. Look at the `Linear` class: initialisation, `fan_in_fan_out`, `merge`. <https://github.com/microsoft/LoRA/blob/main/loralib/layers.py>
7. **Hugging Face `Conv1D`** (in `pytorch_utils.py`): confirm the weight layout you found in step 0. <https://github.com/huggingface/transformers/blob/main/src/transformers/pytorch_utils.py>
8. **PEFT's LoRA layer:** the production version. It's much longer, and seeing what it adds is the lesson: multiple adapters, dropout, merging, other layer types. <https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py>
9. **nanoGPT `model.py`:** a short, readable GPT-2. It's useful for seeing where `c_attn`, `c_proj` and `c_fc` sit in a block. <https://github.com/karpathy/nanoGPT/blob/master/model.py>
10. **Raschka's Appendix E code cells:** now you can read them.

## Questions you should be able to answer after this session

Say these out loud. They're common interview questions.

- What does LoRA train, and what stays frozen? Why does it save memory during training? Think about the optimizer states.
- Why is B initialised to zero, and why not both A and B?
- What does alpha / r do, and why divide by r?
- Why does LoRA add no inference cost after merging?
- Which layers would you put LoRA on, and how would you choose r?
- When would you fine-tune all the weights instead of using LoRA?
