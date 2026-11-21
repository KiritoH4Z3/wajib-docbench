# ADR-001: Self-hosted open models plus one API baseline

Status: accepted, 20 November 2026

## Context

The question is whether a self-hosted open model can extract fields from UAE business
documents well enough to trust, and what it costs next to a commercial API. That needs at
least one UAE-built open model, a commercial baseline, and the same client code for all of
them so only the model changes.

The machine is an RTX 4060 Laptop GPU with 8 GB of VRAM, of which the desktop already uses
about 1.85 GB, so about 6.3 GB is free. The budget is USD 0: no paid GPU, no paid API.

Every fact below was checked on 20 November 2026 against Hugging Face's API, the vendors' pages
and the serving projects' release notes, then re-checked by a second reader who tried to
refute it.

## Decision

### Models

| Slot | Model | File | Licence |
|---|---|---|---|
| TII (Abu Dhabi) | `tiiuae/Falcon-H1-7B-Instruct` | `Falcon-H1-7B-Instruct-Q4_K_M.gguf`, 4,598,344,960 bytes, repo revision `058c8c8f` | TII Falcon License (December 2024) and its acceptable use policy |
| Inception, MBZUAI, Cerebras | `inception42/Jais-2-8B-Chat` | `Q4_K_M.gguf` from `inception42/Jais-2-8B-Chat-GGUF`, 5,104,022,560 bytes, revision `61365d41` | Apache-2.0 |
| Commercial baseline | `gemini-3.5-flash-lite` | Google's OpenAI-compatible endpoint, free tier | Gemini API terms |

**Falcon-H1-Arabic is not in v1 because it cannot be downloaded.** TII announced it in January
2026 (3B, 7B and 34B, a vendor-reported 71.7% on the Open Arabic LLM Leaderboard for the 7B),
but its product page links only to the chat playground. There is no repository under `tiiuae`
on Hugging Face, in its collections or on ModelScope. Falcon-Emirati-7B, announced in October
2026, is in the same state. Both are listed as pending in `config/models.yaml`; if weights
appear, the Falcon-H1-Arabic 7B at Q4 replaces the general model.

**Falcon-H1 7B Instruct takes the TII slot.** It is public, not gated, tagged for Arabic, and
TII's README describes native training in 18 languages including Arabic. No Arabic benchmark
is published for it, so this benchmark is partly a measurement of that gap. Falcon-H1R-7B was
rejected: it is tagged English only and emits long reasoning blocks before answering.

**Jais 2 8B stays in the Jais slot, and Jais 3 waits for v2.** Jais 3 (October 2026, ungated,
Apache-2.0) is newer, but it is a reasoning model that always thinks before answering. v1
compares non-reasoning extractors of similar size; mixing in a reasoning model would change the
cost and latency story for a reason that has nothing to do with Arabic. Its 7B Q4_K_M (5.59 GB)
also leaves no room for a KV cache in 6 GB. Jais 2 is gated with instant approval, so a Hugging
Face account must accept the gate before `make models` can fetch it.

**Gemini 3.5 Flash-Lite is the baseline.** It is free on the free tier, Google describes it as
built for document parsing and extraction, and it speaks the OpenAI chat API. Two consequences
are recorded rather than hidden:

- It ignores `temperature`, so its sampling cannot be pinned the way the open models' can.
  `make samples MODEL=gemini-3.5-flash-lite` repeats 12 documents (one per document type and
  layout) three times into separate run directories, and the report publishes the
  disagreement between draws. Until that has run, the
  API row is a single draw and the README says so.
- It thinks before answering at its default (minimal) setting and gets a 4,096-token output
  budget so thinking cannot truncate the JSON; the open models get 1,024. Its latency and cost
  include the thinking, which is recorded separately where the API reports it. This is the same
  kind of confound that kept Jais 3 out, accepted here because the baseline is the product a
  buyer would actually call, as configured by default.
- Free-tier inputs and outputs may be used to improve Google's products and may be read by
  human reviewers. Every document here is synthetic, which is the only reason this is
  acceptable, and it is exactly the data-residency point the self-hosted models answer.

The stretch slot (a general multilingual 7B) is cut first if time runs short, as the scope says.

### Serving: llama.cpp locally, vLLM kept for a free cloud GPU

The scope planned vLLM. On this GPU it cannot hold either model:

- Falcon-H1 7B in GPTQ Int4 is 7,797,756,040 bytes, more than the free VRAM before any cache.
- Jais 2 has no first-party 4-bit safetensors checkpoint. The only third-party AWQ repository
  is empty, and BF16 is about 16 GB.
- vLLM v0.34.0 documents its GGUF support as highly experimental. It now lives in an
  out-of-tree plugin that is not in the stock image, and neither Falcon-H1 nor Jais 2 is in the
  plugin's tested families.

llama.cpp supports both architectures natively (Falcon-H1 since build b5858, Jais 2 since
b8106) and both vendors publish official GGUF files. The server exposes the same
OpenAI-compatible endpoint, so the client code does not change. The scope's risk table already
names llama.cpp as the fallback, and the engine comparison itself is out of scope until January.

Pinned image: `ghcr.io/ggml-org/llama.cpp:server-cuda12-b12464`. The tag's index digest is
`sha256:ae96e52f8b2106ea82bad786ee2d08bb9b0a8c93403df9c27ae1bf065a118089` and its linux/amd64
manifest is `sha256:6b1c681b535a15d406c7270fd2730f5bd36d3598fbb0ca894bfa47857a700fe3`, which
is the digest Docker reports and `results/results.json` records for every run. The CUDA 13 images are avoided because they require a newer
driver than the 580 series on this machine.

Launch settings per model live in `config/models.yaml`:

- Falcon-H1 7B: all layers on the GPU, 4,096 tokens of context per slot. Only 2 of its KV heads
  are cached per layer, so the cache is small (about 352 MiB at 8k context).

  Measured on 21 November 2026, with about 1.5 GB of VRAM held by desktop apps: 4 slots load
  at 5.6 GB. Falcon-H1 is a hybrid with Mamba layers, and its recurrent state takes about
  134 MiB per slot whatever the context, so 16 slots need 2.1 GB for state alone and do not
  load. Extraction runs on 4 slots. The throughput test runs on 8 slots that share one
  12,288-token cache, each request still capped at 4,096 tokens (`make serve PARALLEL=8
  SHARED_CTX=12288`), which loads at 7.5 GB and is the most that fits. At concurrency 16,
  eight requests wait in the server's queue, and that wait is part of the measured time to
  first token.
- Jais 2 8B: all layers on the GPU, 4,096 tokens per slot, q8_0 KV cache. Jais 2 uses full
  multi-head attention (416 KiB of KV cache per token at fp16), so 8,192 tokens would not fit
  next to the weights. Its model card also warns that K-quants may need a small batch size on
  GPU; the first run compares the default against `-b 8` on 20 documents and records the result
  here.

A vLLM launch is kept in the Makefile for a free Kaggle T4 run, which needs driver 580 or later
for the v0.34.0 image. That run is optional and would be reported as a separate row.

### One client

Every model is called through the `openai` Python SDK with streaming on. Only `base_url`, the
model name and the key change, all read from `config/models.yaml`. The SDK's own retries are
switched off (`max_retries=0`) so a rate-limit retry is always visible in the logs.

## Consequences

- The CV line from the scope changes from "self-hosted with vLLM" to "self-hosted with
  llama.cpp" unless the Kaggle vLLM run happens.
- Self-hosted cost is a cloud equivalent: measured throughput on the laptop GPU, priced at an
  AWS g4dn.xlarge (T4 16 GB) at USD 0.526 per hour on demand in us-east-1 (20 November 2026),
  with an L4 and an RTX 4090 as sensitivity rows, assuming the instance is busy 100% of the
  time. What is known about the match: a T4 has more memory bandwidth than this laptop GPU
  (320 against 256 GB/s), which favours the T4 when generating tokens; prefill compute and
  sustained clocks under load were not measured on a T4, so the self-hosted cost is an
  estimate, not a measurement. A measured Kaggle T4 run would replace it.
- Quantisation is part of the result. The open models run at 4 bits; their vendors evaluated
  them at 16. The README says so next to every score.

## Licence obligations

- Falcon-H1: link the licence and the acceptable use policy, redistribute no weights (the repo
  links to Hugging Face instead), and use TII or Falcon names only to identify the model. The
  README carries the sentence "Falcon-H1-7B-Instruct is built using artificial intelligence
  technology from the Technology Innovation Institute." No clause restricts publishing scores or
  outputs. This is a reading of the text, not legal advice.
- Jais 2: Apache-2.0 with no extra gate terms. The gate shares the requesting account's username
  and email with the model's owner.
- Gemini: the free-tier data-use line above appears in the README.
