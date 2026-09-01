"""UTF-8 guard logits processor for vLLM v1 (byte-level BPE tokenizers).

GLM-5.3-Flash's tokenizer spells many Japanese kanji as a 2-byte fragment token followed by a 1-byte
continuation token (測 = e6b8 + ac, 範 = e7af + 84, …). When the model skips the continuation byte the
character comes back as U+FFFD (measured ≈ 2 per 10K chars on 2× DGX Spark, independent of speculation,
KV dtype and sampler pruning). This processor masks, at every step, the tokens that would make the
byte stream invalid: after a token that ends mid-character only tokens beginning with exactly the
missing continuation bytes are allowed; at a character boundary tokens beginning with a continuation
byte are forbidden. The model then completes the character it started.

Usage (vLLM serve): --logits-processors utf8_guard_lp:Utf8GuardLogitsProcessor  (PYTHONPATH must contain this file)
Note: vLLM rejects custom logits processors while speculative decoding is enabled.
"""
import os
import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor, MoveDirectionality

_BS = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
_CS = _BS[:]
_n = 0
for _b in range(256):
    if _b not in _BS:
        _BS.append(_b); _CS.append(256 + _n); _n += 1
_CHAR2BYTE = {chr(c): b for b, c in zip(_BS, _CS)}


def _token_bytes(s):
    out = bytearray()
    for ch in s:
        b = _CHAR2BYTE.get(ch)
        if b is None:
            return None            # not a byte-level token (special / added) -> treat as boundary-only
        out.append(b)
    return bytes(out)


def _is_cont(b):
    return 0x80 <= b <= 0xBF


def _lead_len(b):
    if b < 0x80: return 1
    if 0xC2 <= b <= 0xDF: return 2
    if 0xE0 <= b <= 0xEF: return 3
    if 0xF0 <= b <= 0xF4: return 4
    return 0                       # invalid lead byte


def _analyse(bts):
    """Return (leading_continuation_count, valid_body, pending_after) for a token's bytes.

    leading_continuation_count: how many continuation bytes the token starts with.
    valid_body: the bytes after those continuations parse as complete-or-trailing-incomplete UTF-8.
    pending_after: continuation bytes still missing at the end of the token (0..3).
    """
    i = 0
    while i < len(bts) and _is_cont(bts[i]):
        i += 1
    lead = i
    pending = 0
    while i < len(bts):
        n = _lead_len(bts[i])
        if n == 0:
            return lead, False, 0
        j = 1
        while j < n and i + j < len(bts):
            if not _is_cont(bts[i + j]):
                return lead, False, 0
            j += 1
        if i + j >= len(bts) and j < n:
            pending = n - j        # token ends mid-character
            i = len(bts)
            break
        i += n
    return lead, True, pending


class Utf8GuardLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        from tokenizers import Tokenizer
        model_path = vllm_config.model_config.tokenizer or vllm_config.model_config.model
        tok = Tokenizer.from_file(os.path.join(model_path, "tokenizer.json"))
        vocab = tok.get_vocab()
        size = max(max(vocab.values()) + 1, int(vllm_config.model_config.get_vocab_size()))
        allowed = torch.zeros((4, size), dtype=torch.bool)
        pending_after = torch.zeros(size, dtype=torch.int64)
        allowed[0, :] = True       # default for ids without a byte mapping: boundary-only
        for s, tid in vocab.items():
            bts = _token_bytes(s)
            if bts is None or not bts:
                continue
            lead, ok, pend = _analyse(bts)
            if not ok:
                allowed[:, tid] = False   # never emit a token that is internally malformed
                continue
            allowed[0, tid] = lead == 0
            for k in (1, 2, 3):
                allowed[k, tid] = (lead == k) or (lead == len(bts) and len(bts) < k)
            pending_after[tid] = pend if lead < len(bts) else 0
        # a token that is all continuation bytes and shorter than the pending count leaves the rest pending;
        # handled at runtime (see apply)
        self.allowed = allowed.to(device)
        self.pending_after = pending_after.to(device)
        self.tok_len = torch.tensor([len(_token_bytes(s) or b"") for s, _ in sorted(vocab.items(), key=lambda kv: kv[1])] + [0] * (size - len(vocab)), dtype=torch.int64, device=device)
        self.lead_cnt = torch.zeros(size, dtype=torch.int64)
        for s, tid in vocab.items():
            bts = _token_bytes(s) or b""
            self.lead_cnt[tid] = _analyse(bts)[0] if bts else 0
        self.lead_cnt = self.lead_cnt.to(device)
        self.device = device
        self.req_out = {}          # batch index -> output token id list (live reference)
        self.req_pending = {}      # batch index -> pending continuation bytes after the last token

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for idx in batch_update.removed:
            self.req_out.pop(idx, None); self.req_pending.pop(idx, None)
        for idx, params, prompt_ids, out_ids in batch_update.added:
            self.req_out[idx] = out_ids
            self.req_pending[idx] = 0
        for a, b, direction in batch_update.moved:
            if direction == MoveDirectionality.SWAP:
                self.req_out[a], self.req_out[b] = self.req_out.get(b), self.req_out.get(a)
                self.req_pending[a], self.req_pending[b] = self.req_pending.get(b, 0), self.req_pending.get(a, 0)
                for d in (self.req_out, self.req_pending):
                    for k in (a, b):
                        if d.get(k) is None: d.pop(k, None)
            else:
                self.req_out[b] = self.req_out.pop(a, None); self.req_pending[b] = self.req_pending.pop(a, 0)
                if self.req_out[b] is None: self.req_out.pop(b, None)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.req_out:
            return logits
        B, V = logits.shape
        ks = [0] * B
        for idx, out in self.req_out.items():
            if idx >= B or not out:
                continue
            last = out[-1]
            prev = self.req_pending.get(idx, 0)
            # advance the pending state with the last emitted token
            lead = int(self.lead_cnt[last]); ln = int(self.tok_len[last])
            if prev > 0 and lead == ln and ln < prev:
                k = prev - ln                    # token was all continuation bytes, still incomplete
            else:
                k = int(self.pending_after[last])
            self.req_pending[idx] = k
            ks[idx] = k
        k_vec = torch.tensor(ks, dtype=torch.int64, device=logits.device)
        allowed = self.allowed
        if allowed.shape[1] != V:
            pad = torch.ones((4, V), dtype=torch.bool, device=logits.device)
            pad[:, :min(V, allowed.shape[1])] = allowed[:, :min(V, allowed.shape[1])]
            allowed = pad
        mask = allowed[k_vec]
        logits.masked_fill_(~mask, float("-inf"))
        return logits
