# Convert the Qwen3-TTS code_predictor (MultiCodeDecoder) into a FUSED CoreML
# model: the entire 15-code RVQ frame in ONE prediction call.
#
# The stepped MultiCodeDecoder needs 16 CoreML predictions per talker frame
# (prefill pos0=talker_hidden, pos1=code0_embed, then one per remaining code),
# with host-side sampling and an embedding lookup between calls. The fused
# model unrolls all 16 positions in-graph (ctx 16 and 15 heads are static),
# samples in-graph via the Gumbel-max trick, and gathers each next-step
# embedding from the codec embedding table in-graph.
#
#   inputs : talker_hidden [1,1024,1,1] fp16 - the talker step's hidden_states
#            code0_embed   [1,1024,1,1] fp16 - CodeEmbedder(code0)
#            gumbel        [15,2048]    fp16 - pre-drawn Gumbel(0,1) noise,
#                                              -log(-log(U)); one row per head.
#                                              argmax(logits/T + G) samples
#                                              from softmax(logits/T) - the
#                                              same distribution as the
#                                              stepped host sampler, and
#                                              deterministic given the noise.
#            temperature   [1]          fp16
#   outputs: codes     [1,15] int32 - codebooks 1..15 (code0 is the talker's)
#            embed_sum [1,1024,1,1] fp16 - sum of the sampled codes' embedding
#                       rows; the runtime adds CodeEmbedder(code0) and the
#                       text embed to form the next talker input.
#
# top-k is BAKED into the graph (torch.topk needs a static k to trace);
# pass --top-k to match your runtime's sampling configuration.
#
# Checkpoint: set QWEN3_TTS_CHECKPOINT to the model.safetensors of the variant
# being converted. Base and CustomVoice share the code_predictor tensor
# layout, so this script converts either; the weights (and therefore the
# asset) are variant-specific.
#
# Usage:
#   QWEN3_TTS_CHECKPOINT=/path/to/model.safetensors \
#     python convert_fused_mcd.py --out ./out --top-k 50 \
#       [--stepped-package /path/to/MultiCodeDecoder.mlpackage]
#
# The optional --stepped-package enables a timing comparison and requires the
# matching stepped MultiCodeDecoder converted from the SAME checkpoint.
import argparse
import os
import sys
import time

import numpy as np
import torch
import coremltools as ct

from code_decoder_ref import HIDDEN, find_base_safetensors, load_tensors, _rms_norm
from multi_code_decoder_ref import (
    MCD_CTX, MCD_HEADS, MCD_LAYERS, MCD_VOCAB, PREFIX, MultiCodeDecoderRef,
)

NEG = -3.0e4  # additive mask fill; representable in fp16, softmax underflows to 0


def load_embed_table():
    """codec_embedding.{0..14} concatenated -> [30720,1024]. Row
    group*2048+code embeds a group-(g+1) code."""
    t = load_tensors(find_base_safetensors(), [f"{PREFIX}.model.codec_embedding."])
    parts = [t[f"{PREFIX}.model.codec_embedding.{i}.weight"] for i in range(MCD_HEADS)]
    return torch.cat(parts, dim=0)


class FusedMCD(torch.nn.Module):
    """All 16 positions + 15 samples of one code_predictor frame, in-graph."""

    def __init__(self, ref: MultiCodeDecoderRef, embed_table: torch.Tensor, top_k: int):
        super().__init__()
        self.ref = ref
        self.top_k = top_k
        self.register_buffer("embed_table", embed_table)

    def _pass(self, x, pos, kc, vc):
        """One position through the 5 layers; returns (final_x, kc, vc). All
        masks/rope rows are constants per unrolled position."""
        cos = self.ref.cos_table[pos].view(1, 1, -1, 1)
        sin = self.ref.sin_table[pos].view(1, 1, -1, 1)
        upd = torch.zeros(1, 1, 1, MCD_CTX)
        upd[0, 0, 0, pos] = 1.0
        pad = torch.full((1, 1, 1, MCD_CTX), NEG)
        pad[0, 0, 0, : pos + 1] = 0.0

        new_kc, new_vc = [], []
        for i, layer in enumerate(self.ref.layers):
            kci = kc[:, i * HIDDEN:(i + 1) * HIDDEN]
            vci = vc[:, i * HIDDEN:(i + 1) * HIDDEN]
            x, ck, cv = layer(x, cos, sin, upd, pad, kci, vci)
            new_kc.append(kci * (1.0 - upd) + ck * upd)
            new_vc.append(vci * (1.0 - upd) + cv * upd)
        return x, torch.cat(new_kc, dim=1), torch.cat(new_vc, dim=1)

    def _sample(self, x, head_idx, gumbel_row, temperature):
        """head -> temperature scale -> top-k mask -> Gumbel-max argmax."""
        h = _rms_norm(x, self.ref.final_ln_w, dim=1)
        logits = self.ref.lm_heads[head_idx](h).view(1, MCD_VOCAB)
        scaled = logits / temperature
        kth = torch.topk(scaled, self.top_k, dim=1).values[:, self.top_k - 1:self.top_k]
        masked = torch.where(scaled >= kth, scaled, torch.full_like(scaled, NEG))
        return torch.argmax(masked + gumbel_row.view(1, MCD_VOCAB), dim=1)

    def forward(self, talker_hidden, code0_embed, gumbel, temperature):
        kc = torch.zeros(1, MCD_LAYERS * HIDDEN, 1, MCD_CTX)
        vc = torch.zeros(1, MCD_LAYERS * HIDDEN, 1, MCD_CTX)

        x, kc, vc = self._pass(talker_hidden, 0, kc, vc)   # prefill pos 0
        x, kc, vc = self._pass(code0_embed, 1, kc, vc)     # prefill pos 1

        codes = []
        embed_sum = torch.zeros(1, HIDDEN, 1, 1)
        code = self._sample(x, 0, gumbel[0], temperature)  # head 0 -> group 1
        for t in range(1, MCD_HEADS):                      # heads 1..14
            codes.append(code)
            row = torch.index_select(self.embed_table, 0, (t - 1) * MCD_VOCAB + code)
            e = row.view(1, HIDDEN, 1, 1)
            embed_sum = embed_sum + e
            x, kc, vc = self._pass(e, t + 1, kc, vc)       # positions 2..15
            code = self._sample(x, t, gumbel[t], temperature)
        codes.append(code)                                  # group 15
        row = torch.index_select(self.embed_table, 0, (MCD_HEADS - 1) * MCD_VOCAB + code)
        embed_sum = embed_sum + row.view(1, HIDDEN, 1, 1)

        return torch.stack(codes, dim=1).to(torch.int32), embed_sum


def stepped_reference_codes(ref, table, talker_hidden, code0_embed, gumbel, temperature, top_k):
    """Drive MultiCodeDecoderRef step-by-step with host sampling identical to
    the in-graph math - the ground truth the fused model must reproduce."""
    from code_decoder_ref import make_step_inputs
    kc = torch.zeros(1, MCD_LAYERS * HIDDEN, 1, MCD_CTX)
    vc = torch.zeros(1, MCD_LAYERS * HIDDEN, 1, MCD_CTX)

    def step(x, pos):
        cl, upd, pad = make_step_inputs(pos, neg=NEG, ctx=MCD_CTX)
        with torch.no_grad():
            all_logits, ku, vu = ref(x, cl, upd, pad, kc, vc)
        kc[:, :, :, pos] = ku[:, :, :, 0]
        vc[:, :, :, pos] = vu[:, :, :, 0]
        return all_logits

    def sample(all_logits, head, g):
        scaled = all_logits[0, head] / temperature
        kth = torch.topk(scaled, top_k).values[-1]
        masked = torch.where(scaled >= kth, scaled, torch.full_like(scaled, NEG))
        return int(torch.argmax(masked + g))

    step(talker_hidden, 0)
    logits = step(code0_embed, 1)
    codes = []
    code = sample(logits, 0, gumbel[0])
    for t in range(1, MCD_HEADS):
        codes.append(code)
        e = table[(t - 1) * MCD_VOCAB + code]
        logits = step(e.view(1, HIDDEN, 1, 1), t + 1)
        code = sample(logits, t, gumbel[t])
    codes.append(code)
    return codes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="./fused-out", help="output directory")
    ap.add_argument("--top-k", type=int, default=50,
                    help="top-k baked into the sampling graph (match your runtime)")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="temperature used for the parity checks (it is a runtime input)")
    ap.add_argument("--stepped-package", default=None,
                    help="stepped MultiCodeDecoder.mlpackage from the SAME checkpoint, for timing")
    ap.add_argument("--skip-w8", action="store_true", help="skip 8-bit palettization")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("loading reference + embed table ...")
    ref = MultiCodeDecoderRef().eval()
    table = load_embed_table()
    fused = FusedMCD(ref, table, args.top_k).eval()

    torch.manual_seed(7)
    ex_hidden = torch.randn(1, HIDDEN, 1, 1) * 0.02
    ex_code0 = table[123].view(1, HIDDEN, 1, 1).clone()
    ex_gumbel = -torch.log(-torch.log(torch.rand(MCD_HEADS, MCD_VOCAB)))
    ex_temp = torch.tensor([args.temperature])

    print("torch-level gate: fused vs stepped reference (same noise) ...")
    with torch.no_grad():
        f_codes, _ = fused(ex_hidden, ex_code0, ex_gumbel, ex_temp)
    s_codes = stepped_reference_codes(
        ref, table, ex_hidden, ex_code0, ex_gumbel, ex_temp, args.top_k)
    if f_codes[0].tolist() != s_codes:
        sys.exit("FUSED GRAPH DOES NOT MATCH STEPPED REFERENCE - fix before converting")
    print("  exact match")

    print("tracing + converting ...")
    ts = torch.jit.trace(fused, (ex_hidden, ex_code0, ex_gumbel, ex_temp))
    m = ct.convert(
        ts,
        inputs=[
            ct.TensorType(name="talker_hidden", shape=(1, HIDDEN, 1, 1), dtype=np.float16),
            ct.TensorType(name="code0_embed", shape=(1, HIDDEN, 1, 1), dtype=np.float16),
            ct.TensorType(name="gumbel", shape=(MCD_HEADS, MCD_VOCAB), dtype=np.float16),
            ct.TensorType(name="temperature", shape=(1,), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="codes", dtype=np.int32),
            ct.TensorType(name="embed_sum", dtype=np.float16),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
    )
    path = os.path.join(args.out, "MultiCodeDecoderFused.mlpackage")
    m.save(path)
    print(f"saved {path}")

    print("CoreML vs torch agreement over 20 random frames (fp16 vs fp32 -")
    print("expect occasional tie-flips; compare against a stepped fp16 model")
    print("for the like-for-like number) ...")
    agree = total = 0
    for i in range(20):
        torch.manual_seed(100 + i)
        th = torch.randn(1, HIDDEN, 1, 1) * 0.02
        c0 = table[torch.randint(0, MCD_VOCAB, (1,))].view(1, HIDDEN, 1, 1).clone()
        g = -torch.log(-torch.log(torch.rand(MCD_HEADS, MCD_VOCAB)))
        with torch.no_grad():
            t_codes, _ = fused(th, c0, g, ex_temp)
        out = m.predict({
            "talker_hidden": th.numpy().astype(np.float16),
            "code0_embed": c0.numpy().astype(np.float16),
            "gumbel": g.numpy().astype(np.float16),
            "temperature": np.array([args.temperature], dtype=np.float16),
        })
        agree += sum(a == b for a, b in zip(t_codes[0].tolist(), out["codes"].reshape(-1).tolist()))
        total += MCD_HEADS
    print(f"  agreement: {agree}/{total} ({100.0 * agree / total:.1f}%)")

    if not args.skip_w8:
        import coremltools.optimize as cto
        print("palettizing to 8-bit (matches the shipped stepped assets) ...")
        cfg = cto.coreml.OptimizationConfig(
            global_config=cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=8))
        m8 = cto.coreml.palettize_weights(
            ct.models.MLModel(path, skip_model_load=True), cfg)
        w8_path = os.path.join(args.out, "MultiCodeDecoderFused_w8.mlpackage")
        m8.save(w8_path)
        print(f"saved {w8_path}")

    if args.stepped_package:
        print("timing: fused vs 16 stepped predictions ...")
        from code_decoder_ref import make_step_inputs
        stepped = ct.models.MLModel(args.stepped_package)
        inp = {
            "talker_hidden": ex_hidden.numpy().astype(np.float16),
            "code0_embed": ex_code0.numpy().astype(np.float16),
            "gumbel": ex_gumbel.numpy().astype(np.float16),
            "temperature": np.array([args.temperature], dtype=np.float16),
        }
        m.predict(inp)
        lat = []
        for _ in range(20):
            t0 = time.time(); m.predict(inp); lat.append(time.time() - t0)
        fused_ms = sorted(lat)[len(lat) // 2] * 1000

        kc = np.zeros((1, MCD_LAYERS * HIDDEN, 1, MCD_CTX), dtype=np.float16)
        vc = np.zeros_like(kc)
        x = ex_hidden.numpy().astype(np.float16)

        def one_step(pos):
            cl, upd, pad = make_step_inputs(pos, neg=NEG, ctx=MCD_CTX)
            return stepped.predict({
                "input_embeds": x, "cache_length": cl.numpy(),
                "kv_cache_update_mask": upd.numpy().astype(np.float16),
                "key_padding_mask": pad.numpy().astype(np.float16),
                "key_cache": kc, "value_cache": vc,
            })
        one_step(0)
        lat = []
        for _ in range(5):
            t0 = time.time()
            for pos in range(16):
                one_step(pos)
            lat.append(time.time() - t0)
        stepped_ms = sorted(lat)[len(lat) // 2] * 1000
        print(f"  fused: {fused_ms:.1f} ms/frame   stepped x16: {stepped_ms:.1f} ms/frame   "
              f"speedup {stepped_ms / fused_ms:.2f}x (prediction only; the stepped runtime "
              f"also pays host sampling + embedding lookups per code)")


if __name__ == "__main__":
    main()
