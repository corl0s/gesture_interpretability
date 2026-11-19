"""Thin instrumentation wrapper around Qwen3-VL (transformers >= 4.57).

Residual-stream convention: `resid_pre[L]` = the input to decoder layer L
(L = 0..n_layers-1) and `resid_pre[n_layers]` = the input to the final norm.
We hook *inputs* (forward pre-hooks) rather than layer outputs because Qwen3-VL adds
DeepStack visual features to image-token positions *after* each of the first few
layers (`Qwen3VLTextModel._deepstack_process`). A pre-hook on layer L sees the stream
after any DeepStack injection, so caching and patching act on what layer L reads.
Caveat: when patching image positions at L <= len(deepstack_visual_indexes), the
receiving run still adds its own DeepStack features at later injection points.

Kang et al. hook layer outputs of LLaVA (`layers[L-1]` output == hidden_states[L]);
for a model without DeepStack the two conventions coincide.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

MODEL_ALIASES = {
    "qwen3vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3vl-32b": "Qwen/Qwen3-VL-32B-Instruct",
}
MERGED_PX = 32  # Qwen3-VL: patch 16 x spatial merge 2


_ST_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def _load_streaming(cls, model_id: str, dtype, device):
    """Build the model directly on `device`, then copy weights in tensor by tensor using
    plain file reads (no mmap), so peak host memory is one tensor."""
    import glob
    import json
    import os
    import struct

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig

    root = model_id if os.path.isdir(model_id) else snapshot_download(model_id, allow_patterns=["*.json", "*.safetensors"])
    config = AutoConfig.from_pretrained(root)
    with torch.device(device):
        model = cls._from_config(config, dtype=dtype)
    params = dict(model.state_dict())
    loaded = set()
    for path in sorted(glob.glob(os.path.join(root, "*.safetensors"))):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
            base = 8 + n
            for key, info in header.items():
                if key == "__metadata__":
                    continue
                name = key if key in params else None
                if name is None:  # checkpoint 'model.X' vs module 'model.language_model.X' style differences
                    for cand in (key.replace("model.", "model.language_model.", 1), "model." + key):
                        if cand in params:
                            name = cand
                            break
                if name is None:
                    print("unused checkpoint tensor:", key)
                    continue
                s, e = info["data_offsets"]
                f.seek(base + s)
                src_dt = _ST_DTYPES[info["dtype"]]
                dst = params[name].view(-1)
                el = torch.tensor([], dtype=src_dt).element_size()
                chunk = (64 << 20) // el  # 64 MB host buffer at a time
                off, total = 0, dst.numel()
                while off < total:
                    k = min(chunk, total - off)
                    buf = bytearray(f.read(k * el))
                    dst[off:off + k].copy_(torch.frombuffer(buf, dtype=src_dt).to(dst.dtype))
                    off += k
                loaded.add(name)
    if getattr(config, "tie_word_embeddings", False) or getattr(config.text_config, "tie_word_embeddings", False):
        model.tie_weights()
        loaded |= {k for k in params if k.startswith("lm_head")}
    missing = [k for k in params if k not in loaded]
    if missing:
        raise RuntimeError(f"{len(missing)} parameters not loaded, e.g. {missing[:5]}")
    return model


@dataclass
class Encoded:
    inputs: dict
    tokens: List[str]
    image_pos: torch.Tensor  # sequence indices of image tokens, row-major over merged grid
    grid_hw: tuple  # (rows, cols) of merged image tokens
    scale: tuple  # (sx, sy): processed px per original px
    image_size: tuple  # original (W, H)

    @property
    def seq_len(self) -> int:
        return len(self.tokens)

    @property
    def last(self) -> int:
        return self.seq_len - 1


class VLM:
    def __init__(self, model: str = "qwen3vl-2b", device: Optional[str] = None, dtype: str = "auto"):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.model_id = MODEL_ALIASES.get(model, model)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            dtype = "bfloat16"
        self.dtype = getattr(torch, dtype)
        # Streaming loader instead of from_pretrained: on this Windows machine safetensors'
        # mmap path fails (paging-file error / access violation) when commit memory is low.
        self.model = _load_streaming(Qwen3VLForConditionalGeneration, self.model_id, self.dtype, self.device).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tok = self.processor.tokenizer
        lm = self.model.model.language_model
        self.layers = lm.layers
        self.final_norm = lm.norm
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.text_config.hidden_size
        self.image_token_id = self.model.config.image_token_id
        self.deepstack_layers = len(getattr(self.model.config.vision_config, "deepstack_visual_indexes", []) or [])
        self._answer_cache: Dict[str, List[int]] = {}
        self.answer_aliases: Dict[str, List[str]] = {}
        torch.set_grad_enabled(False)

    # ------------------------------------------------------------------ inputs
    def encode(self, image: Image.Image, question: str) -> Encoded:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        ids = inputs["input_ids"][0]
        tokens = self.tok.convert_ids_to_tokens(ids.tolist())
        image_pos = (ids == self.image_token_id).nonzero().flatten().cpu()
        t, h, w = inputs["image_grid_thw"][0].tolist()
        rows, cols = h // 2, w // 2
        assert rows * cols == len(image_pos), (rows, cols, len(image_pos))
        W, H = image.size
        scale = (cols * MERGED_PX / W, rows * MERGED_PX / H)
        return Encoded(inputs, tokens, image_pos, (rows, cols), scale, (W, H))

    def image_tokens_in_box(self, enc: Encoded, box) -> torch.Tensor:
        """Sequence positions of merged image tokens overlapping pixel box (x0,y0,x1,y1)."""
        x0, y0, x1, y1 = box
        sx, sy = enc.scale
        rows, cols = enc.grid_hw
        c0, c1 = max(0, int(x0 * sx // MERGED_PX)), min(cols - 1, int(x1 * sx // MERGED_PX))
        r0, r1 = max(0, int(y0 * sy // MERGED_PX)), min(rows - 1, int(y1 * sy // MERGED_PX))
        idx = [r * cols + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        return enc.image_pos[idx]

    def text_positions(self, enc: Encoded) -> torch.Tensor:
        """All positions after the image span (question + chat template suffix)."""
        return torch.arange(int(enc.image_pos.max()) + 1, enc.seq_len)

    def find_token(self, enc: Encoded, word: str, occurrence: int = -1) -> int:
        """Position of `word` (last sub-token, Kang et al. convention) after the image."""
        sub = self.tok.tokenize(" " + word)[-1]
        start = int(enc.image_pos.max()) + 1
        hits = [i for i in range(start, enc.seq_len) if enc.tokens[i] == sub]
        if not hits:
            raise ValueError(f"token {sub!r} for {word!r} not in prompt")
        return hits[occurrence]

    # ----------------------------------------------------------------- answers
    def answer_ids(self, word: str) -> List[int]:
        """First-token ids of plausible spellings of a one-word answer."""
        if word not in self._answer_cache:
            ids = set()
            for w in [word] + self.answer_aliases.get(word, []):
                for v in (w, w.capitalize(), " " + w, " " + w.capitalize()):
                    ids.add(self.tok.encode(v, add_special_tokens=False)[0])
            self._answer_cache[word] = sorted(ids)
        return self._answer_cache[word]

    def answer_logprob(self, logits_last: torch.Tensor, word: str) -> float:
        lp = torch.log_softmax(logits_last.float(), -1)
        return torch.logsumexp(lp[self.answer_ids(word)], 0).item()

    def check_answer_vocab(self, words: Sequence[str]):
        seen = {}
        for w in words:
            for i in self.answer_ids(w):
                if i in seen and seen[i] != w:
                    raise ValueError(f"answer token collision: {w} / {seen[i]} ({self.tok.decode([i])!r})")
                seen[i] = w

    # ------------------------------------------------------------------- runs
    def _module(self, L: int):
        return self.final_norm if L == self.n_layers else self.layers[L]

    @contextmanager
    def hooks(self, cache: Optional[dict] = None, cache_layers: Sequence[int] = (), cache_pos=None,
              patch: Optional[dict] = None, add: Optional[dict] = None):
        """cache: dict to fill {L: tensor[len(cache_pos), d]} (float32, cpu).
        patch: {L: (positions, values[len(pos), d])} -> overwrite resid_pre[L] at positions.
        add:   {L: (positions, vectors[len(pos), d] or [d])} -> add to resid_pre[L].
        Interventions apply before caching at the same layer."""
        patch, add = patch or {}, add or {}
        layers = sorted(set(cache_layers) | set(patch) | set(add))
        handles = []

        def make(L):
            def pre_hook(module, args, kwargs):
                hs = args[0] if args else kwargs["hidden_states"]
                if L in patch or L in add:
                    hs = hs.clone()
                    if L in patch:
                        pos, val = patch[L]
                        hs[0, pos] = val.to(hs.device, hs.dtype)
                    if L in add:
                        pos, vec = add[L]
                        hs[0, pos] = hs[0, pos] + vec.to(hs.device, hs.dtype)
                if cache is not None and L in cache_layers:
                    sel = hs[0] if cache_pos is None else hs[0, cache_pos]
                    cache[L] = sel.detach().float().cpu()
                if args:
                    return (hs,) + tuple(args[1:]), kwargs
                kwargs["hidden_states"] = hs
                return args, kwargs
            return pre_hook

        for L in layers:
            handles.append(self._module(L).register_forward_pre_hook(make(L), with_kwargs=True))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def forward(self, enc: Encoded, **hook_kw) -> torch.Tensor:
        """Returns last-position logits (float32, cpu)."""
        with self.hooks(**hook_kw):
            out = self.model(**enc.inputs, use_cache=False, logits_to_keep=1)
        return out.logits[0, -1].float().cpu()

    def run_cache(self, enc: Encoded, positions, layers: Optional[Sequence[int]] = None, **hook_kw):
        """Forward pass caching resid_pre at `positions` for `layers` (default: all incl. final)."""
        layers = list(range(self.n_layers + 1)) if layers is None else list(layers)
        cache: dict = {}
        logits = self.forward(enc, cache=cache, cache_layers=layers, cache_pos=positions, **hook_kw)
        acts = torch.stack([cache[L] for L in layers], 0)  # [n_layers, n_pos, d]
        return logits, acts

    def generate(self, enc: Encoded, max_new_tokens: int = 8) -> str:
        out = self.model.generate(**enc.inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0, enc.seq_len:], skip_special_tokens=True).strip()
