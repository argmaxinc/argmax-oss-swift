# Fused MultiCodeDecoder conversion

Converts the Qwen3-TTS `code_predictor` into a **fused** CoreML model that
produces all 15 RVQ codes of a talker frame in **one prediction call**,
replacing 16 stepped predictions + 15 host samplings + 15 embedding lookups
per frame. TTSKit auto-detects the fused variant at load time by its `gumbel`
input; stepped models are unaffected.

Sampling runs in-graph via the Gumbel-max trick — the runtime supplies
Gumbel(0,1) noise per frame and `argmax(logits/T + G)` draws from
`softmax(logits/T)`, the same distribution as the stepped host sampler, and
deterministically so given the noise (which is what the parity gate exploits).

## Usage

```bash
QWEN3_TTS_CHECKPOINT=/path/to/model.safetensors \
  python convert_fused_mcd.py --out ./fused-out --top-k 50
```

- `--top-k` is **baked into the graph** (`torch.topk` needs a static k to
  trace); match it to your runtime's sampling configuration.
- `temperature` remains a runtime input.
- Base and CustomVoice checkpoints share the `talker.code_predictor` tensor
  layout, so the script converts either; the resulting asset is
  weight-specific to the checkpoint it was converted from.
- The script gates itself: the fused graph must reproduce a stepped PyTorch
  reference exactly (same noise) before conversion proceeds, and reports
  fp16-CoreML vs fp32-torch code agreement afterwards. For a like-for-like
  agreement number, compare against a stepped fp16 CoreML model with
  identical noise (autoregressive tie-flips cascade, so fp16-vs-fp32
  disagreement overstates the difference).

Measured on an Apple silicon Mac with the 0.6B model: end-to-end generation
went from 19.8 to 24.3 steps/s (+23%); code agreement vs the stepped fp16
model under identical noise was 99.3% (19/20 frames bit-identical).

Deploy the compiled asset as
`multi_code_decoder/<versionDir>/<variant>/MultiCodeDecoder.mlmodelc` and
select it via `TTSKitConfig.multiCodeDecoderVariant`.
