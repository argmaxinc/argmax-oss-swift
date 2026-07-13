# Qwen3 voice-clone Phase 2: PyTorch reference for the CodeDecoder — the
# 28-layer stateful talker — matching Argmax's CodeDecoder.mlmodelc graph
# op-for-op (argmax-specs/.../CodeDecoder.mlmodelc/model.mil):
#
#   inputs : input_embeds [1,1024,1,1] fp, cache_length int32 [1],
#            kv_cache_update_mask [1,256] (one-hot at cache_length),
#            key_padding_mask [1,256] (additive: 0 valid / -inf masked),
#            key_cache, value_cache [1,28672,1,256]  (28 layers x 8 kv-heads x
#            128 head_dim channels, 256 ctx positions; read-only in-graph)
#   outputs: logits [1,1,3072], hidden_states [1,1024,1,1] (post final norm),
#            key_cache_updates / value_cache_updates [1,28672,1,1] (this
#            step's per-layer K (normed+rope'd) and V; host scatters them
#            into the cache at column cache_length)
#
# Rope is a BAKED [256,128] cos/sin table gathered by cache_length: the
# talker's 3 mrope position ids are always equal, so interleaved mrope
# degenerates to standard Qwen3 rope (theta 1e6, head_dim 128, half-split
# rotate_half). Linears are 1x1 convs on [1,C,1,1] (ANE layout).
#
# Importable from both venvs (needs only torch + numpy).
import json
import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CTX = 256
N_LAYERS = 28
HIDDEN = 1024
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 3072
VOCAB = 3072
ROPE_THETA = 1e6
RMS_EPS = 1e-6


def find_base_safetensors():
    """Resolve the Qwen3-TTS checkpoint (any variant whose code_predictor you
    are converting - Base and CustomVoice share the tensor layout).

    Set QWEN3_TTS_CHECKPOINT to the model.safetensors path, or rely on the
    fallback scan of the HF cache for a Qwen3-TTS snapshot."""
    explicit = os.environ.get("QWEN3_TTS_CHECKPOINT")
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError(f"QWEN3_TTS_CHECKPOINT={explicit} does not exist")
    root = os.path.expanduser("~/.cache/huggingface/hub")
    for dirpath, _, files in os.walk(root):
        if "Qwen3-TTS" in dirpath and "model.safetensors" in files and "speech_tokenizer" not in dirpath:
            return os.path.join(dirpath, "model.safetensors")
    raise FileNotFoundError(
        "No Qwen3-TTS model.safetensors found - set QWEN3_TTS_CHECKPOINT")


def load_tensors(path, wanted_prefixes):
    """Load tensors whose name starts with one of wanted_prefixes, as fp32."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in hdr.items():
            if name == "__metadata__":
                continue
            if not any(name.startswith(p) for p in wanted_prefixes):
                continue
            dtype, shape = meta["dtype"], meta["shape"]
            s, e = meta["data_offsets"]
            f.seek(base + s)
            raw = f.read(e - s)
            if dtype == "BF16":
                arr = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                arr = (arr << 16).view(np.float32).reshape(shape)
            elif dtype == "F32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            elif dtype == "F16":
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
            else:
                raise ValueError(f"unhandled dtype {dtype} for {name}")
            out[name] = torch.from_numpy(arr.copy())
    return out


def rope_tables(ctx=CTX, head_dim=HEAD_DIM, theta=ROPE_THETA):
    """cos/sin [ctx, head_dim] in the cat(freqs, freqs) layout upstream uses."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim))
    pos = torch.arange(ctx, dtype=torch.float64)
    freqs = torch.outer(pos, inv_freq)          # [ctx, 64]
    emb = torch.cat([freqs, freqs], dim=-1)     # [ctx, 128]
    return emb.cos().float(), emb.sin().float()


def _rms_norm(x, weight, dim):
    # x * rsqrt(mean(x^2, dim)+eps) * weight ; weight already shaped to broadcast
    var = (x * x).mean(dim=dim, keepdim=True)
    return x * torch.rsqrt(var + RMS_EPS) * weight


class CodeDecoderLayer(nn.Module):
    def __init__(self, t, prefix, ctx=CTX):
        super().__init__()
        self.ctx = ctx
        def conv(name, out_ch, in_ch):
            c = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            with torch.no_grad():
                c.weight.copy_(t[f"{prefix}.{name}.weight"].view(out_ch, in_ch, 1, 1))
            return c

        self.q_proj = conv("self_attn.q_proj", N_HEADS * HEAD_DIM, HIDDEN)
        self.k_proj = conv("self_attn.k_proj", N_KV_HEADS * HEAD_DIM, HIDDEN)
        self.v_proj = conv("self_attn.v_proj", N_KV_HEADS * HEAD_DIM, HIDDEN)
        self.o_proj = conv("self_attn.o_proj", HIDDEN, N_HEADS * HEAD_DIM)
        self.gate_proj = conv("mlp.gate_proj", INTERMEDIATE, HIDDEN)
        self.up_proj = conv("mlp.up_proj", INTERMEDIATE, HIDDEN)
        self.down_proj = conv("mlp.down_proj", HIDDEN, INTERMEDIATE)
        self.register_buffer("input_ln_w", t[f"{prefix}.input_layernorm.weight"].view(1, HIDDEN, 1, 1))
        self.register_buffer("post_ln_w", t[f"{prefix}.post_attention_layernorm.weight"].view(1, HIDDEN, 1, 1))
        self.register_buffer("q_norm_w", t[f"{prefix}.self_attn.q_norm.weight"].view(1, HEAD_DIM, 1, 1))
        self.register_buffer("k_norm_w", t[f"{prefix}.self_attn.k_norm.weight"].view(1, HEAD_DIM, 1, 1))
        self.scale = HEAD_DIM ** -0.5

    def forward(self, x, cos, sin, upd_mask, pad_mask, k_cache, v_cache):
        # x [1,1024,1,1]; cos/sin [1,1,128,1]; upd_mask [1,1,1,256];
        # pad_mask [1,1,1,256] additive; k_cache/v_cache [1,1024,1,256]
        h = _rms_norm(x, self.input_ln_w, dim=1)

        q = self.q_proj(h)   # [1,2048,1,1]
        k = self.k_proj(h)   # [1,1024,1,1]
        v = self.v_proj(h)   # [1,1024,1,1]

        # per-head RMSNorm on head_dim (mil: reshape [heads,128,1,1], norm dim=1)
        q = _rms_norm(q.view(N_HEADS, HEAD_DIM, 1, 1), self.q_norm_w, dim=1)
        k = _rms_norm(k.view(N_KV_HEADS, HEAD_DIM, 1, 1), self.k_norm_w, dim=1)

        q = q.view(1, N_HEADS, HEAD_DIM, 1)
        k = k.view(1, N_KV_HEADS, HEAD_DIM, 1)

        def rope(z):  # half-split rotate_half on dim 2
            z1, z2 = z[:, :, :HEAD_DIM // 2], z[:, :, HEAD_DIM // 2:]
            return z * cos + torch.cat([-z2, z1], dim=2) * sin

        q = rope(q)
        k = rope(k)

        cur_key = k.reshape(1, N_KV_HEADS * HEAD_DIM, 1, 1)    # update output
        cur_value = v                                          # update output

        # blend current K/V into the cached window at cache_length
        key = k_cache * (1.0 - upd_mask) + cur_key * upd_mask      # [1,1024,1,ctx]
        value = v_cache * (1.0 - upd_mask) + cur_value * upd_mask

        kh = key.view(1, N_KV_HEADS, HEAD_DIM, self.ctx).repeat_interleave(2, dim=1)
        vh = value.view(1, N_KV_HEADS, HEAD_DIM, self.ctx).repeat_interleave(2, dim=1)

        w = torch.matmul((q * self.scale).transpose(2, 3), kh)     # [1,16,1,256]
        w = torch.softmax(w + pad_mask, dim=3)
        attn = torch.matmul(vh, w.transpose(2, 3))                 # [1,16,128,1]
        attn = attn.reshape(1, N_HEADS * HEAD_DIM, 1, 1)
        x = x + self.o_proj(attn)

        h = _rms_norm(x, self.post_ln_w, dim=1)
        h = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x + h, cur_key, cur_value


class CodeDecoderRef(nn.Module):
    """Single-token decode step; KV caches passed as plain tensors."""

    def __init__(self, tensors=None):
        super().__init__()
        if tensors is None:
            tensors = load_tensors(
                find_base_safetensors(), ["talker.model.layers.", "talker.model.norm.", "talker.codec_head."]
            )
        self.layers = nn.ModuleList(
            CodeDecoderLayer(tensors, f"talker.model.layers.{i}") for i in range(N_LAYERS)
        )
        self.register_buffer("final_ln_w", tensors["talker.model.norm.weight"].view(1, HIDDEN, 1, 1))
        head = nn.Conv2d(HIDDEN, VOCAB, 1, bias=False)
        with torch.no_grad():
            head.weight.copy_(tensors["talker.codec_head.weight"].view(VOCAB, HIDDEN, 1, 1))
        self.codec_head = head
        cos, sin = rope_tables()
        self.register_buffer("cos_table", cos)
        self.register_buffer("sin_table", sin)

    def forward(self, input_embeds, cache_length, kv_cache_update_mask, key_padding_mask,
                key_cache, value_cache):
        cos = self.cos_table[cache_length].view(1, 1, HEAD_DIM, 1)
        sin = self.sin_table[cache_length].view(1, 1, HEAD_DIM, 1)
        upd = kv_cache_update_mask.view(1, 1, 1, CTX)
        pad = key_padding_mask.view(1, 1, 1, CTX)

        x = input_embeds
        k_upd, v_upd = [], []
        for i, layer in enumerate(self.layers):
            kc = key_cache[:, i * HIDDEN:(i + 1) * HIDDEN]
            vc = value_cache[:, i * HIDDEN:(i + 1) * HIDDEN]
            x, ck, cv = layer(x, cos, sin, upd, pad, kc, vc)
            k_upd.append(ck)
            v_upd.append(cv)

        hidden_states = _rms_norm(x, self.final_ln_w, dim=1)          # [1,1024,1,1]
        logits = self.codec_head(hidden_states)                       # [1,3072,1,1]
        logits = logits.squeeze(3).permute(0, 2, 1)                   # [1,1,3072]
        key_cache_updates = torch.cat(k_upd, dim=1)                   # [1,28672,1,1]
        value_cache_updates = torch.cat(v_upd, dim=1)
        return logits, hidden_states, key_cache_updates, value_cache_updates


def make_step_inputs(pos, dtype=torch.float32, neg=float("-inf"), ctx=CTX):
    """Host-side mask/cache_length construction for decode position `pos`."""
    cache_length = torch.tensor([pos], dtype=torch.int32)
    upd = torch.zeros(1, ctx, dtype=dtype)
    upd[0, pos] = 1.0
    pad = torch.full((1, ctx), neg, dtype=dtype)
    pad[0, : pos + 1] = 0.0
    return cache_length, upd, pad
