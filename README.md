# dumb-llm

Train a GPT-2-scale language model from scratch. Three tools form the full pipeline: ingest web text, train a transformer, evaluate the result.

```
HuggingFace (SlimPajama)
    │
    ▼  pipeline ingest
data/tokenized/slimpajama-gpt2bpe/shard_NNNN.npy
    │
    ▼  python training/train.py --config configs/gpt2_small.yaml
out/gpt2-small/step-NNNNNNNN/checkpoint.pt
    │
    ▼  python -m eval.run_benchmarks / perplexity / generate
results/
```

## Setup

```bash
uv sync
```

---

## 1. Data Ingestion

**Package:** `dumb-llm-data/`  
**Command:** `pipeline ingest`

Downloads documents from HuggingFace in streaming mode, tokenizes them with GPT-2 BPE, and writes fixed-size shards to disk as `.npy` files. Raw text is also preserved as JSONL.

### Usage

```bash
pipeline ingest \
  --source slimpajama \
  --tokenizer gpt2bpe \
  --output-dir data/ \
  --seq-length 1024 \
  --shard-size 8192
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--source` | `slimpajama` | Data source. Currently `slimpajama` (streams `gmongaras/SlimPajama-627B_Reupload`). |
| `--tokenizer` | `gpt2bpe` | Tokenizer. Currently `gpt2bpe` (tiktoken GPT-2 BPE, vocab size 50,257). |
| `--output-dir` | `data/` | Root output directory. |
| `--max-documents` | all | Stop after this many documents. Omit to run until the stream is exhausted. |
| `--seq-length` | `1024` | Tokens per row in each shard. |
| `--shard-size` | `8192` | Rows (sequences) per `.npy` shard file. |
| `--raw-chunk-size` | `10000` | Documents per raw JSONL file. |

### How it works

1. Documents are streamed from HuggingFace one at a time — the full corpus never loads into memory.
2. Each document is tokenized and its tokens are appended to a flat buffer. Documents are concatenated end-to-end with no separators or padding.
3. When the buffer holds `shard-size × seq-length` tokens, it is sliced, reshaped to `(shard-size, seq-length)`, and saved as `shard_NNNN.npy` (`uint16`).
4. Raw documents are buffered and flushed to `chunk_NNNN.jsonl` every `raw-chunk-size` documents.
5. After the stream ends (or on Ctrl+C), any remaining complete sequences are written as a final shard. Trailing tokens that don't fill a complete sequence are discarded.
6. A `manifest.json` is written describing the full dataset.

Progress is printed in-place: `{docs} docs | {tokens} tokens | {shards} shards`.

### Output layout

```
data/
├── raw/
│   └── slimpajama/
│       ├── chunk_0000.jsonl
│       └── ...
└── tokenized/
    └── slimpajama-gpt2bpe/
        ├── shard_0000.npy      # uint16, shape (8192, 1024)
        ├── shard_0001.npy
        ├── manifest.json
        └── ...
```

### `manifest.json`

```json
{
  "tokenizer": "gpt2bpe",
  "seq_length": 1024,
  "dtype": "uint16",
  "vocab_size": 50257,
  "num_shards": 42,
  "total_sequences": 344064,
  "total_tokens": 352321536,
  "shard_files": [
    {"filename": "shard_0000.npy", "num_sequences": 8192},
    ...
  ]
}
```

---

## 2. Training

**Package:** `dumb-llm-training/`  
**Command:** `python training/train.py --config <yaml>`

Pretrains a GPT-2-scale decoder-only transformer on the shards produced by the ingestion step. Uses [litgpt](https://github.com/Lightning-AI/litgpt) for the model and Lightning Fabric for multi-GPU training (DDP or FSDP).

### Usage

```bash
# single GPU
python training/train.py --config configs/gpt2_small.yaml

# override any config field with dot notation
python training/train.py --config configs/gpt2_small.yaml \
  --training.max_iters=5000 \
  --training.batch_size=4

# resume from a checkpoint
python training/train.py --config configs/gpt2_small.yaml \
  --training.resume=out/gpt2-small/step-00005000
```

Multi-GPU training is handled automatically by Lightning Fabric — no special launcher is required for DDP.

### Preset configs

Three model sizes are provided in `dumb-llm-training/configs/`:

| Config | Parameters | Layers | Heads | `n_embd` | Default `max_iters` | Peak LR |
|---|---|---|---|---|---|---|
| `gpt2_small.yaml` | ~124M | 12 | 12 | 768 | 10,000 | 6e-4 |
| `gpt2_medium.yaml` | ~350M | 24 | 16 | 1,024 | 20,000 | 3e-4 |
| `gpt2_large.yaml` | ~774M | 36 | 20 | 1,280 | 30,000 | 2.5e-4 |

### Configuration reference

All fields can be overridden on the command line with `--section.field=value`.

**`model:`**

| Field | Default | Description |
|---|---|---|
| `n_layer` | 12 | Transformer layers |
| `n_head` | 12 | Attention heads |
| `n_embd` | 768 | Embedding dimension |
| `block_size` | 1024 | Context window length |
| `vocab_size` | 50257 | GPT-2 vocabulary size |
| `bias` | `false` | Bias in attention/MLP |
| `dropout` | 0.0 | Dropout rate |

**`training:`**

| Field | Default | Description |
|---|---|---|
| `max_iters` | 10000 | Total training steps |
| `batch_size` | 8 | Micro-batch size per GPU |
| `gradient_accumulation_iters` | 4 | Accumulation steps before optimizer update |
| `learning_rate` | 6e-4 | Peak learning rate |
| `min_lr` | 6e-5 | Floor LR after cosine decay |
| `weight_decay` | 0.1 | AdamW weight decay (applied to 2D params only) |
| `beta1` / `beta2` | 0.9 / 0.95 | AdamW betas |
| `grad_clip` | 1.0 | Gradient clipping max norm |
| `warmup_iters` | 200 | Linear warmup steps |
| `lr_decay_iters` | 10000 | Cosine decay endpoint |
| `eval_interval` | 250 | Validate every N steps |
| `eval_iters` | 50 | Validation batches per evaluation |
| `save_interval` | 1000 | Checkpoint every N steps |
| `log_interval` | 10 | Print loss every N steps |
| `log_dir` | `logs/` | TensorBoard log directory |
| `out_dir` | `out/` | Checkpoint directory |
| `precision` | `bf16-mixed` | Mixed precision mode |
| `strategy` | `ddp` | `ddp` or `fsdp` |
| `resume` | — | Path to checkpoint directory to resume from |

**`data:`**

| Field | Default | Description |
|---|---|---|
| `data_dir` | `data/tokenized/slimpajama-gpt2bpe` | Path to shard directory |
| `val_split` | 0.05 | Fraction of shards reserved for validation |
| `num_workers` | 4 | DataLoader worker processes |

### How it works

1. **Dataset**: shards are memory-mapped (`np.load(..., mmap_mode="r")`). Each 1024-token row produces 1,023 input/label pairs for next-token prediction. The last 5% of shards (alphabetically) are held out as a validation set.
2. **Optimizer**: AdamW with fused CUDA kernel. Weight decay is applied only to 2D weight matrices; biases and layer norms are exempt.
3. **LR schedule**: linear warmup → cosine decay → flat floor at `min_lr`.
4. **Gradient accumulation**: gradients are accumulated for `gradient_accumulation_iters` steps before each optimizer update. DDP gradient syncs are deferred during accumulation to avoid unnecessary communication.
5. **Checkpoints**: saved to `{out_dir}/step-{N:08d}/checkpoint.pt` plus a `meta.json` with iteration metadata. Training state (iter number, best val loss) is saved to `training_state.pt` on exit.
6. **Ctrl+C**: triggers a checkpoint save before exiting.

### Monitoring

```bash
tensorboard --logdir logs/
```

Metrics logged: `train/loss`, `train/lr`, `train/tokens_per_sec`, `val/loss`.

### Checkpoint layout

```
out/
└── gpt2-small/
    ├── step-00001000/
    │   ├── checkpoint.pt   # model + optimizer state dicts
    │   └── meta.json       # {"iter_num": 1000, "model": {...}, "training": {...}}
    ├── step-00002000/
    └── training_state.pt   # {"iter_num": N, "best_val_loss": X, ...}
```

---

## 3. Evaluation

**Package:** `dumb-llm-eval/`

Three independent evaluation tools. All accept a checkpoint directory and a model config YAML.

### 3a. Benchmarks

```bash
python -m eval.run_benchmarks \
  --checkpoint out/gpt2-small/step-00010000 \
  --config dumb-llm-training/configs/gpt2_small.yaml
```

Runs the model against standard NLP benchmarks using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Results are printed as a table and saved as JSON.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to checkpoint directory |
| `--config` | required | Path to model architecture YAML |
| `--tasks` | from `benchmarks.yaml` | Comma-separated lm-eval task names |
| `--benchmarks` | `configs/benchmarks.yaml` | YAML with default task list and settings |
| `--output` | `results/benchmarks.json` | Results output path |
| `--device` | `cuda` | Compute device |
| `--batch-size` | 1 | Inference batch size |
| `--num-fewshot` | from `benchmarks.yaml` | Few-shot examples per task |
| `--limit` | from `benchmarks.yaml` | Cap examples per task |

**Default benchmark suite** (`configs/benchmarks.yaml`), all 0-shot:

| Task | What it measures |
|---|---|
| `lambada_openai` | Long-range context: predict the final word of a passage |
| `hellaswag` | Commonsense sentence completion |
| `piqa` | Physical intuition question answering |
| `arc_easy` | Elementary science questions |
| `winogrande` | Coreference / pronoun resolution |

### 3b. Perplexity

```bash
python -m eval.perplexity \
  --checkpoint out/gpt2-small/step-00010000 \
  --config dumb-llm-training/configs/gpt2_small.yaml \
  --data /path/to/corpus.txt
```

Computes perplexity on an arbitrary text file using a sliding window. Useful for measuring fit on domain-specific corpora.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to checkpoint directory |
| `--config` | required | Path to model architecture YAML |
| `--data` | required | Text file to evaluate |
| `--output` | `results/perplexity.json` | Results output path |
| `--device` | `cuda` | Compute device |
| `--stride` | `block_size / 2` | Sliding window step (default 512) |

The sliding window ensures each token is scored exactly once with the maximum available left context. Output JSON includes `perplexity`, `cross_entropy`, `num_tokens`, `num_windows`, and the run parameters.

### 3c. Generation

```bash
python -m eval.generate \
  --checkpoint out/gpt2-small/step-00010000 \
  --config dumb-llm-training/configs/gpt2_small.yaml
```

Generates text from a fixed set of prompts for qualitative inspection. Running the same prompts across checkpoints makes it easy to track how coherence improves during training.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | required | Path to checkpoint directory |
| `--config` | required | Path to model architecture YAML |
| `--prompts` | `prompts/default.yaml` | YAML file containing the prompt list |
| `--output` | `results/generations.txt` | Text file to write generations to |
| `--device` | `cuda` | Compute device |
| `--max-tokens` | 200 | Maximum new tokens per prompt |
| `--temperature` | 0.8 | Sampling temperature (0 = greedy) |
| `--top-k` | 50 | Top-k filtering |

The default prompt set (`prompts/default.yaml`) covers general language modeling (news, story, technical writing), factual recall, reasoning, and cybersecurity-domain text. Generation stops on EOS or after `--max-tokens` tokens.

---

## Repository layout

```
dumb-llm/
├── pyproject.toml              # uv workspace root
├── dumb-llm-data/              # ingestion pipeline  (package: dumb-llm)
│   ├── pipeline/
│   │   ├── cli.py              # entry point: `pipeline ingest`
│   │   ├── config.py           # PipelineConfig dataclass
│   │   ├── sharding.py         # ShardWriter + RawChunkWriter
│   │   ├── sources/            # DataSource ABC + SlimPajama implementation
│   │   └── tokenization/       # Tokenizer ABC + GPT-2 BPE implementation
│   └── pyproject.toml
├── dumb-llm-training/          # pretraining loop  (package: dumb-llm-training)
│   ├── training/
│   │   ├── train.py            # entry point
│   │   ├── config.py           # ModelConfig, TrainingConfig, DataConfig
│   │   ├── data.py             # ShardDataset, create_datasets()
│   │   ├── model.py            # create_model(), configure_optimizer()
│   │   └── utils.py            # get_lr(), evaluate(), save/load_checkpoint()
│   ├── configs/
│   │   ├── gpt2_small.yaml
│   │   ├── gpt2_medium.yaml
│   │   └── gpt2_large.yaml
│   └── pyproject.toml
└── dumb-llm-eval/              # evaluation tools  (package: dumb-llm-eval)
    ├── eval/
    │   ├── run_benchmarks.py   # lm-eval harness integration
    │   ├── perplexity.py       # sliding-window perplexity
    │   ├── generate.py         # text generation
    │   └── model_loader.py     # shared checkpoint loading + LitGPTLM wrapper
    ├── configs/
    │   ├── benchmarks.yaml     # default task list
    │   ├── gpt2_small.yaml
    │   ├── gpt2_medium.yaml
    │   └── gpt2_large.yaml
    ├── prompts/
    │   └── default.yaml        # fixed evaluation prompts
    └── pyproject.toml
```
