#!/usr/bin/env python
"""Export a Kompress PyTorch checkpoint to ONNX INT8 for Headroom's light path.

Why this exists
---------------
Headroom's ``[proxy]`` extra ships ``onnxruntime`` but **not** torch — the
proxy runs Kompress text compression on ONNX Runtime alone. The loader
(``headroom/transforms/kompress_compressor.py``) downloads
``onnx/kompress-int8.onnx`` from the model repo and runs it through
``_OnnxModel``, which expects a single graph output named ``final_scores``
(per-token importance in ``[0, 1]``, kept when ``> 0.5``).

``chopratejas/kompress-v2-base`` ships only PyTorch weights
(``model.safetensors`` / ``merged.pt``) — no ONNX. So pointing Headroom at v2
without an ONNX export would silently force the heavier ``[ml]`` (torch) path
on every proxy install. This script reproduces v1's exact ONNX contract from
the v2 PyTorch checkpoint, so a default swap stays zero-cost for light installs.

The model is a *custom* dual-head ModernBERT (token classifier + span CNN), not
a standard HF architecture, so ``optimum-cli export onnx`` does not apply — we
trace the real module from ``kompress_compressor._get_model_class()``.

Requires
--------
    pip install headroom-ai[ml] onnxruntime    # torch + transformers + onnxruntime

Usage
-----
    # Convert + verify locally (writes onnx/kompress-int8.onnx):
    python scripts/export_kompress_v2_onnx.py --model-id chopratejas/kompress-v2-base

    # Convert, verify, and upload back to the HF repo (needs `huggingface-cli login`):
    python scripts/export_kompress_v2_onnx.py --model-id chopratejas/kompress-v2-base --upload
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("export_kompress_v2_onnx")

# ModernBERT encoder + tokenizer base (must match training and the loader).
BASE_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_MODEL_ID = "chopratejas/kompress-v2-base"


def _build_core(model_id: str):
    """Instantiate HeadroomCompressorModel and load the merged v2 weights.

    The v2 repo's ``model.safetensors`` is the *unmerged* PEFT structure
    (``encoder.base_model.model...`` with separate ``base_layer`` + LoRA
    adapters), which does not map onto ``HeadroomCompressorModel``. The
    canonical artifact is ``merged.pt`` — a structured checkpoint with already
    LoRA-merged sub-state-dicts:

        {"encoder_state_dict", "token_head_state_dict",
         "span_conv_state_dict", "config", "checkpoint_kind"}

    Each loads cleanly (0 missing / 0 unexpected) into the encoder + heads.
    """
    import torch
    from huggingface_hub import hf_hub_download

    from headroom.transforms.kompress_compressor import _get_model_class

    ckpt_path = hf_hub_download(model_id, "merged.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    for key in ("encoder_state_dict", "token_head_state_dict", "span_conv_state_dict"):
        if key not in ckpt:
            raise RuntimeError(
                f"merged.pt missing '{key}'. Found: {sorted(ckpt)}. "
                "This script targets the v2 'merged' checkpoint format."
            )

    core = _get_model_class()(model_name=BASE_MODEL)

    def _strict_load(module, sd, label: str) -> None:
        missing, unexpected = module.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"{label}: state_dict mismatch (missing={list(missing)[:5]}, "
                f"unexpected={list(unexpected)[:5]}). Architecture drifted from the checkpoint."
            )
        logger.info("  %s loaded (%d tensors, exact match)", label, len(sd))

    logger.info("Loading merged.pt (checkpoint_kind=%s)", ckpt.get("checkpoint_kind"))
    _strict_load(core.encoder, ckpt["encoder_state_dict"], "encoder")
    _strict_load(core.token_head, ckpt["token_head_state_dict"], "token_head")
    _strict_load(core.span_conv, ckpt["span_conv_state_dict"], "span_conv")

    core.eval()
    return core


def _export_wrapper(core):
    """Wrap the dual head so forward() returns `final_scores` (== get_scores)."""
    import torch
    import torch.nn as nn

    class ExportWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):  # noqa: ANN001
            hidden = self.inner.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
            token_probs = torch.softmax(self.inner.token_head(hidden), dim=-1)[:, :, 1]
            span_scores = self.inner.span_conv(hidden.transpose(1, 2)).squeeze(1)
            return token_probs * (0.5 + 0.5 * span_scores)

    return ExportWrapper(core).eval()


def export(model_id: str, out_path: Path, opset: int, precision: str) -> None:
    import numpy as np
    import torch

    core = _build_core(model_id)
    wrapper = _export_wrapper(core)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # fp32 path: trace straight to the final artifact (lossless — verified 100%
    # keep-decision agreement with PyTorch). int8 path: trace to a temp fp32
    # graph, then dynamically quantize into the final artifact.
    trace_target = out_path if precision == "fp32" else out_path.with_name("kompress-fp32-tmp.onnx")

    dummy_ids = torch.randint(0, 1000, (1, 64), dtype=torch.long)
    dummy_mask = torch.ones((1, 64), dtype=torch.long)

    logger.info("Tracing → ONNX (opset %d, precision=%s) ...", opset, precision)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_ids, dummy_mask),
            str(trace_target),
            input_names=["input_ids", "attention_mask"],
            output_names=["final_scores"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "final_scores": {0: "batch", 1: "seq"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    if precision == "int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic

        logger.info("INT8 dynamic quantization (MatMul only) → %s", out_path)
        # Restrict to MatMul: the encoder's linear layers carry ~all the weight
        # mass and ORT's CPU provider implements MatMulInteger. Quantizing the
        # tiny span_conv Conv1d layers would emit ConvInteger, which ORT CPU
        # cannot run. per_channel recovers transformer accuracy at the 0.5 boundary.
        quantize_dynamic(
            str(trace_target),
            str(out_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
        )
        trace_target.unlink(missing_ok=True)

    _verify(model_id, core, out_path, np, torch)


def _verify(model_id: str, core, out_path: Path, np, torch) -> None:
    """Compare ONNX scores against PyTorch get_scores on a real tokenized sample."""
    import onnxruntime as ort
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    sample = (
        "The proxy compresses tool outputs before they reach the model. "
        "Errors and stack traces should survive; boilerplate should not. "
    ) * 6
    words = sample.split()
    enc = tok(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        torch_scores = core.get_scores(enc["input_ids"], enc["attention_mask"])[0].cpu().numpy()

    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    onnx_scores = sess.run(
        ["final_scores"],
        {
            "input_ids": enc["input_ids"].numpy().astype(np.int64),
            "attention_mask": enc["attention_mask"].numpy().astype(np.int64),
        },
    )[0][0]

    max_abs = float(np.max(np.abs(torch_scores - onnx_scores)))
    keep_torch = torch_scores > 0.5
    keep_onnx = onnx_scores > 0.5
    agree = float((keep_torch == keep_onnx).mean())
    logger.info(
        "Verify: max|Δscore|=%.4f  keep-decision agreement=%.1f%% (fp32 ~100%%, int8 ~98-100%%)",
        max_abs,
        agree * 100,
    )
    if agree < 0.98:
        logger.warning(
            "Keep-decision agreement below 98%% — for fp32 this means a tracing "
            "problem; for int8 consider per_channel/fp32. Inspect before publishing."
        )


def upload(model_id: str, out_path: Path) -> None:
    from huggingface_hub import upload_file

    # Publish under onnx/<artifact filename> so int8 and fp32 can coexist.
    repo_path = f"onnx/{out_path.name}"
    logger.info("Uploading %s → %s:%s", out_path, model_id, repo_path)
    upload_file(
        path_or_fileobj=str(out_path),
        path_in_repo=repo_path,
        repo_id=model_id,
        commit_message="Add ONNX export for Headroom lightweight (no-torch) path",
    )
    logger.info("Uploaded. Headroom's ONNX loader will now find it on next cold start.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument(
        "--precision",
        choices=["fp32", "int8"],
        default="fp32",
        help="fp32 = lossless, larger artifact. int8 = ~2x smaller, tiny accuracy cost.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Local output path. Defaults to onnx/kompress-<precision>.onnx.",
    )
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--upload",
        action="store_true",
        help="Upload to the HF repo under onnx/<filename> (needs HF write auth).",
    )
    args = ap.parse_args()

    out_path = args.out or Path(f"onnx/kompress-{args.precision}.onnx")
    export(args.model_id, out_path, args.opset, args.precision)
    if args.upload:
        upload(args.model_id, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
