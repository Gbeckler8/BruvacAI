# BruvacAI

*Built with Llama.*

A LoRA fine-tuned LLM that generates Magic: The Gathering flavor text from card data — built on Llama-3.2-1B-Instruct and trained on 23K+ cards via Scryfall. BruvacAI learns the genre's distinctive voice — terse, evocative prose and quotes drawn from Magic's worldbuilding — and generates new flavor text. Named after Bruvac, the Grandiloquent — a fitting namesake for a model whose whole job is producing grandiloquent prose.

🤗 **Model:** [Gbeckler/bruvac-ai](https://huggingface.co/Gbeckler/bruvac-ai)

🚀 **Live demo:** [Hugging Face Space](https://huggingface.co/spaces/Gbeckler/BruvacAI_Space)

---

## What it does

Give BruvacAI a card name, mana cost, type line, and (optionally) rules text, and it generates new flavor text in the style of real Magic: The Gathering cards.

## Sample outputs

Generated on held-out validation cards.

### Sinew Sliver
**Generated:** With the rise of Phyrexia, many Eldrazi, including the Slivers, became agents of change, seeking to preserve their homeworlds.
**Actual:** As the muscle cords of the creature twitched, Hanna saw an unsettling unanimity in the others' rippling flesh. She didn't know what it meant, but she urged Sisay to keep the ship at a safe distance.


### Tolarian Scholar
**Generated:** "Every book is a door to a greater library."
**Actual:** The Tolarian Academies embrace a tradition of study and research while discouraging the kinds of experiments that ruined the original island of Tolaria.


### Dissipate
**Generated:** "Magic is a waste of time." —Elspeth
**Actual:** "If you weren't born with it, you don't need it." —Grahilah, former trader of Amiqat

*(More examples in the [training notebook](./BruvacAI_full_loop_v1.ipynb) and live in the [demo](https://huggingface.co/spaces/Gbeckler/BruvacAI_Space).)*

---

## Build Process

### Data
- Sourced from the [Scryfall Bulk Data API](https://scryfall.com/docs/api/bulk-data) (Default Cards dataset)
- Filtered to cards with non-empty flavor text, then deduplicated on `(name, flavor_text)` pairs to preserve genuine flavor-text variants across reprints while collapsing exact duplicates
- Final dataset: ~27,230 unique (name, flavor_text) examples
- Train/val/test split: 85/10/5, split by **card name** (not by row) to prevent the same card's reprint variants from leaking across splits

### Model
- Base model: [Llama-3.2-1B-Instruct](https://huggingface.co/unsloth/Llama-3.2-1B-Instruct) (via Unsloth's 4-bit quantized build)
- Fine-tuning method: LoRA (rank 32, applied to all attention + MLP projection layers), trained with [Unsloth](https://github.com/unslothai/unsloth) for memory/speed efficiency on a free-tier Colab T4 GPU
- Rules text was prioritized below name/type in the prompt ordering, and randomly dropped in ~25% of training examples (dropout-style augmentation) so the model learns name/type alone are often sufficient to produce strong flavor text — rules text supplements rather than dominates

### Why Llama-3.2-1B specifically
Flavor text is a short-form, stylistically narrow task — complex reasoning is not a part of the task. Primarily, the model must absorb the stylistic choices of Wizards of the Coast, as well as some small instances of Magic: The Gathering world building. A 1B model has enough capacity for this scope without the compute/memory overhead larger models would add, which mattered given training ran on a free-tier T4 in Google Colab. It also kept iteration fast enough to debug the pipeline in reasonable time, and the ~23K-example training set is well-suited to LoRA fine-tuning at this scale without excessive overfitting risk.

### Epoch selection
Trained for 3 epochs; validation loss was lowest after epoch 1 and increased in epochs 2–3 (classic overfitting pattern — training loss kept dropping while validation loss rose). The epoch-1 checkpoint was selected as the final model.

| Epoch | Training Loss | Validation Loss |
|-------|---------------|------------------|
| 1     | 3.1127        | 3.1331           |
| 2     | 2.5522        | 3.1515           |
| 3     | 2.0527        | 3.3732           |

---

## Repository structure

```
/notebooks
  ├── BruvacAI_full_loop_v1.ipynb   — full data pipeline + training + evaluation
app.py                                — Gradio app (deployed to Hugging Face Spaces)
requirements.txt
LICENSE-LLAMA.txt                     — Llama 3.2 Community License (required redistribution notice)
README.md
```

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("Gbeckler/bruvac-ai")
base_model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
model = PeftModel.from_pretrained(base_model, "Gbeckler/bruvac-ai")
```

See `BruvacAI_inference.ipynb` for a full working example, or try the [live demo](https://huggingface.co/spaces/Gbeckler/BruvacAI_Space) — no setup required.

---

## Challenges encountered

Most of the friction in this project came from library version drift and infrastructure quirks rather than modeling decisions. However, they are nevertheless important to note to remember for fine-tuning work in the future.

- **Gradient checkpointing incompatibility**: `use_gradient_checkpointing` (both Unsloth's and PyTorch's native modes) conflicted with Unsloth's fused chunked cross-entropy loss, which uses `torch.func.grad_and_value` — a documented PyTorch limitation where saved-tensor hooks aren't supported inside `torch.func` transforms. Resolved by disabling checkpointing and compensating with a smaller batch size + higher gradient accumulation.
- **Packing-induced slowdown**: Enabling `packing=True` caused a ~10x throughput drop (0.07 it/s with 100% GPU utilization). Likely, this was due to kernel recompilation from inconsistent packed-batch shapes. Disabling packing brought throughput back to ~0.73 it/s.
- **`SFTConfig`/`TrainingArguments` and `padding_free` conflicts**: Several `trl`/`transformers` version-mismatch errors required explicitly constructing `SFTConfig` and setting `padding_free` to match whichever packing mode was active.
- **Dataset tokenization multiprocessing hang**: `num_proc > 1` occasionally stalled indefinitely in Colab; resolved via environment variables disabling multiprocessing plus a runtime restart.

## Limitations

- **No power/toughness conditioning**: the model was trained without creature stats, so it has no signal about a creature's relative size/power when generating flavor text.
- **1B parameter model**: capable of solid short-form stylistic output, but won't match larger models on coherence over longer generations or deeper lore consistency.
- **No rigorous quantitative evaluation**: flavor text quality is inherently subjective; evaluation here is qualitative (side-by-side comparison against real flavor text on held-out cards), not benchmark-driven.
- **CPU inference in the live demo**: the Hugging Face Space runs on free CPU hardware (no 4-bit quantization), so generation takes ~5–15 seconds per request.

## Future Work
- **Improved UI** - more intuitive frontend for players to use the Gradio demo more easily
### Documentation and Files
- **Inference Notebook** - to come!
- **Test Data Documentation** - The results shown earlier in this README are from the Validation Set! The test data set has **untapped** potential.
### Prompt Engineering
- **Power/toughness conditioning**: will be part of the system prompt in v2
- **Few-shot learning**: providing direct samples in the system prompt could improve model accuracy due to in-context learning principles

---

## License & Attribution

This model is a fine-tuned derivative of [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and is distributed under the terms of the [Llama 3.2 Community License Agreement](https://www.llama.com/llama3_2/license/). A copy of that license is included in this repository as `LICENSE-LLAMA.txt`.

> Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

Code in this repository (data pipeline, training scripts, Gradio app) is original work by the author and is provided as-is for portfolio/educational purposes.

## Acknowledgments

- Card data: [Scryfall](https://scryfall.com)
- Fine-tuning framework: [Unsloth](https://github.com/unslothai/unsloth) (Apache 2.0)
- Base model: [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) (Meta) — Built with Llama
- My brother: for inspiring in me the passion for Magic: The Gathering and similar card games
