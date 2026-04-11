# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Kronos is a foundation model for financial candlestick (K-line) forecasting. It uses a two-stage approach: a tokenizer quantizes OHLCV data into hierarchical discrete tokens via Binary Spherical Quantization (BSQ), then a decoder-only Transformer generates forecasts autoregressively. Pre-trained models are hosted on Hugging Face under the `NeoQuasar` namespace.

## Commands

### Install dependencies
```
pip install -r requirements.txt
```

### Run tests
```
pytest tests/
```
Tests download models from Hugging Face on first run (Kronos-Tokenizer-base and Kronos-small). Runs regression and MSE tests against known outputs in `tests/data/`.

### Run a single test
```
pytest tests/test_kronos_regression.py::test_kronos_predictor_regression -k "512"
```

### Finetuning (Qlib-based, multi-GPU)
```
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```
Configuration is in `finetune/config.py`. Requires `pyqlib` and local Qlib data.

### Finetuning (CSV-based)
See `finetune_csv/` — supports YAML config files in `finetune_csv/configs/`.

### Web UI
```
cd webui && python run.py
```
Flask app with Plotly visualization. Extra deps in `webui/requirements.txt`.

## Architecture

### Core model (`model/`)

Three main classes in `model/kronos.py`, exported via `model/__init__.py`:

- **`KronosTokenizer`** — Encoder-decoder Transformer with BSQ quantization. Takes raw OHLCV input, encodes it, quantizes via `BSQuantizer` into hierarchical tokens (s1_bits + s2_bits), then decodes. Used for both training the tokenizer and for encoding input during inference.

- **`Kronos`** — Decoder-only autoregressive Transformer. Takes token indices from the tokenizer, embeds them via `HierarchicalEmbedding` + `TemporalEmbedding`, and predicts next tokens through a `DualHead` that outputs both pre-token (s1) and full token logits. Supports `generate()` with temperature, top-k, and top-p sampling.

- **`KronosPredictor`** — High-level inference wrapper. Handles the full pipeline: normalization → tokenization → autoregressive generation → detokenization → denormalization. Provides `predict()` for single series and `predict_batch()` for batched inference. All model loading uses `PyTorchModelHubMixin.from_pretrained()`.

### Building blocks (`model/module.py`)

- `BSQuantizer` / `BinarySphericalQuantizer` — Binary spherical quantization with entropy regularization
- `TransformerBlock` — Standard block with `MultiHeadAttentionWithRoPE`, `FeedForward`, `RMSNorm`
- `HierarchicalEmbedding` — Embeds the two-level (pre/post) token structure
- `DualHead` — Produces logits for both hierarchical token levels
- `TemporalEmbedding` — Time feature encoding (minute, hour, weekday, day, month)

### Model variants

| Model | Tokenizer | Context | Params |
|-------|-----------|---------|--------|
| Kronos-mini | Kronos-Tokenizer-2k | 2048 | 4.1M |
| Kronos-small | Kronos-Tokenizer-base | 512 | 24.7M |
| Kronos-base | Kronos-Tokenizer-base | 512 | 102.3M |

### Data flow (inference)

1. Input: DataFrame with OHLCV columns + timestamps
2. `KronosPredictor` normalizes each feature to [0,1] range using min/max from context
3. `KronosTokenizer.encode()` produces token indices
4. `Kronos.generate()` autoregressively produces future token indices
5. `KronosTokenizer.decode_from_indices()` reconstructs normalized values
6. `KronosPredictor` denormalizes back to original scale
