# Qwen3 voice-clone Phase 2: PyTorch reference for the MultiCodeDecoder — the
# 5-layer code_predictor that expands each talker frame into 15 residual codec
# codes — matching Argmax's MultiCodeDecoder.mlmodelc (argmax-specs/.../model.mil):
#
#   inputs : input_embeds [1,1024,1,1], cache_length int32 [1],
#            kv_cache_update_mask [1,16], key_padding_mask [1,16] (additive),
#            key_cache, value_cache [1,5120,1,16]  (5 layers x 1024 ch, ctx 16;
#            NON-stateful: host keeps the tiny cache between the 15 sub-steps)
#   outputs: all_logits [1,15,2048] (every lm_head applied to the same final
#            hidden — the host picks the head for the current sub-step),
#            key_cache_updates / value_cache_updates [1,5120,1,1]
#
# Same building blocks as the CodeDecoder (plain Qwen3 rope baked as a [16,128]
# table, mask-blend KV, GQA 16/8, per-head q/k RMSNorm).
import torch
import torch.nn as nn

from code_decoder_ref import (
    HIDDEN, VOCAB as _TALKER_VOCAB,  # noqa: F401 (talker vocab unused here)
    CodeDecoderLayer, _rms_norm, find_base_safetensors, load_tensors, rope_tables,
)

MCD_CTX = 16
MCD_LAYERS = 5
MCD_HEADS = 15
MCD_VOCAB = 2048
PREFIX = "talker.code_predictor"


class MultiCodeDecoderRef(nn.Module):
    """Single-step code_predictor; KV caches passed as plain tensors."""

    def __init__(self, tensors=None):
        super().__init__()
        if tensors is None:
            tensors = load_tensors(
                find_base_safetensors(),
                [f"{PREFIX}.model.layers.", f"{PREFIX}.model.norm.", f"{PREFIX}.lm_head."],
            )
        self.layers = nn.ModuleList(
            CodeDecoderLayer(tensors, f"{PREFIX}.model.layers.{i}", ctx=MCD_CTX)
            for i in range(MCD_LAYERS)
        )
        self.register_buffer("final_ln_w", tensors[f"{PREFIX}.model.norm.weight"].view(1, HIDDEN, 1, 1))
        heads = []
        for i in range(MCD_HEADS):
            h = nn.Conv2d(HIDDEN, MCD_VOCAB, 1, bias=False)
            with torch.no_grad():
                h.weight.copy_(tensors[f"{PREFIX}.lm_head.{i}.weight"].view(MCD_VOCAB, HIDDEN, 1, 1))
            heads.append(h)
        self.lm_heads = nn.ModuleList(heads)
        cos, sin = rope_tables(ctx=MCD_CTX)
        self.register_buffer("cos_table", cos)
        self.register_buffer("sin_table", sin)

    def forward(self, input_embeds, cache_length, kv_cache_update_mask, key_padding_mask,
                key_cache, value_cache):
        cos = self.cos_table[cache_length].view(1, 1, -1, 1)
        sin = self.sin_table[cache_length].view(1, 1, -1, 1)
        upd = kv_cache_update_mask.view(1, 1, 1, MCD_CTX)
        pad = key_padding_mask.view(1, 1, 1, MCD_CTX)

        x = input_embeds
        k_upd, v_upd = [], []
        for i, layer in enumerate(self.layers):
            kc = key_cache[:, i * HIDDEN:(i + 1) * HIDDEN]
            vc = value_cache[:, i * HIDDEN:(i + 1) * HIDDEN]
            x, ck, cv = layer(x, cos, sin, upd, pad, kc, vc)
            k_upd.append(ck)
            v_upd.append(cv)

        h = _rms_norm(x, self.final_ln_w, dim=1)                       # [1,1024,1,1]
        logits = [head(h).squeeze(3) for head in self.lm_heads]        # 15 x [1,2048,1]
        all_logits = torch.cat(logits, dim=2).permute(0, 2, 1)         # [1,15,2048]
        key_cache_updates = torch.cat(k_upd, dim=1)                    # [1,5120,1,1]
        value_cache_updates = torch.cat(v_upd, dim=1)
        return all_logits, key_cache_updates, value_cache_updates
