# lora-int8-lab

A learning project: fine-tune GPT-2 small with a **LoRA layer I write myself**, then **compress** the fine-tuned model to int8 with a quantizer I also write myself. At each step I measure what changed in accuracy, size and speed.

The goal is to learn. The code that matters (the LoRA layer, the training loop and the quantizer) is written by hand. AI is only a tutor and a reviewer.

## Plan

| Session | Topic | Status |
|---|---|---|
| 1 | Hand-written LoRA layer, tested, inserted into GPT-2 | not started |
| 2 | LoRA fine-tune on a measurable task, compared with the `peft` library | not started |
| 3 | Merge LoRA into the weights, then hand-written int8 quantization; size, latency and accuracy vs fp32 | not started |
| 4 | Results, write-up, lessons | not started |
| Optional extra | Load GPT-2's weights into my own Transformer implementation and match Hugging Face (not needed for sessions 1–4) | not started |

- [`GUIDE.md`](GUIDE.md): the overall guide (goals, "done when" tests, rules).
- [`sessions/`](sessions/): a detailed guide for each session, with reading lists.

## Results

To be filled in after session 3.

| Model | Trainable params | Size on disk | CPU latency (batch 8) | SST-2 val accuracy |
|---|---|---|---|---|
| GPT-2 small, zero-shot | – | | | |
| + LoRA (mine) | | | | |
| + LoRA (`peft`) | | | | |
| LoRA merged, int8 (mine) | – | | | |
| LoRA merged, int8 (PyTorch) | – | | | |
