# Guide: LoRA + int8 on GPT-2, in 4 sessions

About 10 hours in total: 4 sessions of about 2.5 hours, plus an optional fifth. Each session ends with something that works and a test that proves it.

## Rules

- **You write the core code yourself:** the LoRA layer, the code that inserts it into the model, the training loop, the merge and the quantizer.
- **AI is your tutor, not your typist.** When you're stuck, ask for a hint and say which level you want:
  - **Hint 1:** where to look (a concept, or a page from the reading list).
  - **Hint 2:** the approach, in words.
  - **Hint 3:** pseudo-code for the tricky part only.

  Letting AI write boilerplate is fine: data loading, argument parsing, plots, timing code.
- **Don't move on until the "Done when" test passes.** A bug in session 1 is cheap to find; in session 3 it's expensive.
- **Read reference implementations after your own attempt,** not before. Comparing your version with the authors' is where most of the learning happens.
- **Write numbers down** as you go, in `EXPERIMENTS.md`: setting, trainable parameters, time, memory, accuracy. They become the README and your interview answers.

## Setup (once)

- Python 3.12 and a virtual environment (`uv` or `venv`).
- Packages: `torch` (the CPU build is fine), `transformers`, `datasets`, `peft` (session 2 only, to compare against), `pytest`.
- Model: GPT-2 small from Hugging Face (`openai-community/gpt2`, 124M parameters). It fits easily on a 16 GB laptop CPU.
- If training on CPU is too slow in session 2, use Colab's free GPU for that session only.

---

## Session 1: a LoRA layer, written by you (about 2.5h)

**Goal:** a `LoRALinear` module that wraps a frozen layer and adds a trainable low-rank update. Then a function that inserts it into GPT-2's attention.

**The idea:** keep the pretrained weight W frozen. Next to it, train two thin matrices, A (r × in) and B (out × r):

```
output = W x + (alpha / r) · B A x
```

B starts at zero, so at the start the model is exactly the original model.

**Deliverables:**
- `lora.py`
- `tests/test_lora.py`
- a function `apply_lora(model, targets, r, alpha)`

**Done when:**
- Right after wrapping, outputs are identical to the original model's, both for a single layer and for the whole GPT-2.
- Only the LoRA parameters have `requires_grad=True`.
- The trainable parameter count equals your hand calculation (for `c_attn` with r = 8: 294,912).
- A toy task shows the loss going down.

Detailed guide: [`sessions/session-1-lora-layer.md`](sessions/session-1-lora-layer.md).

## Session 2: fine-tune and measure (about 2.5h)

**Goal:** show, with a number, that your LoRA changes the model's behaviour, and that it matches the reference library.

**Task:** SST-2 sentiment (movie reviews, positive or negative). Turn it into next-token prediction:

```
Review: <text>
Sentiment: positive | negative
```

To predict, compare the model's score for " positive" against " negative" at the last position. Accuracy on the SST-2 validation split (872 examples) is the metric.

**Steps:**
1. Measure the zero-shot accuracy of the base model. It's the baseline, and it will be weak.
2. Write your own training loop:
   - Train on a subset (start with 2,000 examples, short sequences).
   - Compute the loss only on the label token.
   - Use AdamW on the LoRA parameters only.
3. Repeat with the `peft` library, using the same r, alpha, target modules, data, steps and seed. GPT-2's attention uses a `Conv1D` layer with transposed weights, so `peft` needs `fan_in_fan_out=True`. Find out why.
4. Optional: try r = 1, 4, 8, 16 and see where more rank stops helping.

**Done when:**
- Your LoRA clearly beats zero-shot on validation.
- Your LoRA lands within about 1–2 accuracy points of `peft` with the same settings.
- The numbers are in `EXPERIMENTS.md`.

## Session 3: merge, then compress to int8 (about 2.5h)

**Goal:** a smaller model, with a measured trade-off.

**Steps:**
1. **Merge:** add `(alpha / r) · B A` into W, so LoRA costs nothing at inference. Watch out for the `Conv1D` transpose.
   - Done when: merged and unmerged outputs match within about 1e-5.
2. **Your int8 quantizer:** weight-only, symmetric, with one scale per output channel:
   ```
   scale = max|W_row| / 127
   q = round(W / scale), clipped to [-127, 127]
   ```
   Store `q` as int8 and `scale` as fp32. In the forward pass, dequantize and multiply.
   - Also try a single scale for the whole matrix (per-tensor), and explain the accuracy difference. The answer is outliers.
3. **Measure each model:** fp32, merged fp32, your int8, and PyTorch's int8. For each one, record:
   - size of the weights on disk;
   - CPU latency for a fixed batch;
   - SST-2 validation accuracy.
4. **Compare with PyTorch's dynamic quantization:** `torch.ao.quantization.quantize_dynamic`, or the newer `torchao` library.
   - Note: it only targets `nn.Linear`, and GPT-2's attention uses `Conv1D`, so you'll need to convert those layers first.

**What to expect, and to understand:** your weight-only int8 cuts memory by about 4× but may not be faster. You still compute in fp32 after dequantizing. Real speed-ups need int8 kernels, which PyTorch's dynamic quantization uses. Explaining this difference is a strong interview answer.

**Done when:**
- The results table has all four rows.
- You can explain each number: why the size dropped, why the latency did or didn't change, and why accuracy barely moved (or why it did).

## Session 4: write it up (about 2.5h)

**Goal:** a repo a stranger understands in 2 minutes.

- **README:** what you built, the results table, one plot (accuracy vs rank, or size vs accuracy), and how to run it.
- **"What I learned":** 5–8 bullets, written as you'd say them in an interview.
- **Clean-up:** tests pass, one command per experiment, a pinned `requirements` or `pyproject`.
- **The CV line, with your real numbers:**
  > Implemented LoRA and int8 post-training quantization from scratch in PyTorch on GPT-2; LoRA reached X% on SST-2 (vs Y% zero-shot) training 0.24% of parameters, and int8 cut model size Z× with W points of accuracy change.

## Session 5 (optional): your own Transformer with GPT-2's weights

**Goal:** load GPT-2 small's real weights into your own Transformer implementation, and match Hugging Face's logits to within 1e-4.

**What it involves:**
- Turn an encoder-decoder into decoder-only.
- Pre-LN blocks and a final LayerNorm.
- Learned position embeddings.
- tanh-GELU.
- A fused QKV projection plus an output projection.
- Transposing the four `Conv1D` weights in each block.
- Debugging layer by layer, using `output_hidden_states=True`.

Do it when the four main sessions are done. It's the best way to learn GPT-2's internals, but it isn't needed for the LoRA and int8 story.

---

## Reading list for later sessions

- **QLoRA** (Dettmers et al., 2023): LoRA on top of a 4-bit quantized base model. It ties sessions 2 and 3 together. <https://arxiv.org/abs/2305.14314>
- **LLM.int8()** (Dettmers et al., 2022): why outliers break naive int8, and the fix. <https://arxiv.org/abs/2208.07339>
- **Hugging Face blog, "A Gentle Introduction to 8-bit Matrix Multiplication…":** the same ideas, explained more simply. <https://huggingface.co/blog/hf-bitsandbytes-integration>
- **Maarten Grootendorst, "A Visual Guide to Quantization":** pictures of symmetric vs asymmetric quantization, per-tensor vs per-channel, and PTQ vs QAT. <https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization>
- **PyTorch quantization docs:** <https://docs.pytorch.org/docs/stable/quantization.html>
- **torchao docs:** <https://docs.pytorch.org/ao/stable/index.html>
- **SST-2 dataset:** <https://huggingface.co/datasets/stanfordnlp/sst2>
