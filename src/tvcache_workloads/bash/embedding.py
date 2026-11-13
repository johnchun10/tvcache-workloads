"""The embedders a run can choose between, by short name.

tvclient embeds each executed stateful command (BashToolCall.embed) and stores
the vector with it; the server pairs siblings whose vectors are similar, and
contraction compares those pairs. Each rollout worker process loads its own
copy of the model on first use.

    TVCACHE_EMBEDDER       an alias below, or any sentence-transformers model name
    TVCACHE_EMBED_DEVICE   where it runs. Default cpu: the GPU belongs to
                           vLLM and the trainer, and commands are short
"""

import os
import threading
from typing import Any, List, Optional

# alias -> (model, SentenceTransformer options, max tokens per command or None for the model's own)
EMBEDDERS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", {}, None),
    "jina-code": ("jinaai/jina-embeddings-v2-base-code", {"trust_remote_code": True}, 512),
}


def resolve(name: str) -> tuple:
    """(model name, options, max tokens) for an alias or a full model name."""
    if name in EMBEDDERS:
        return EMBEDDERS[name]
    for model, options, limit in EMBEDDERS.values():
        if name == model:
            return model, options, limit
    return name, {}, None


class CommandEmbedder:
    """A sentence-transformers model, loaded on first use. It has the one
    method tvclient's Embedder interface asks for, encode(text)."""

    def __init__(self, name: str, device: Optional[str] = None):
        self.model_name, self.options, self.max_tokens = resolve(name)
        self.device = device or os.environ.get("TVCACHE_EMBED_DEVICE") or "cpu"
        self._model: Any = None
        # Guards the lazy load and serializes encode: fast tokenizers fail
        # when several threads use them at once.
        self._lock = threading.Lock()

    def encode(self, text: str) -> List[float]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device=self.device, **self.options)
                if self.max_tokens is not None:
                    self._model.max_seq_length = self.max_tokens
            return self._model.encode(text).tolist()


_current: Optional[str] = None


def use_embedder(name: str) -> None:
    """Makes `name` this process's embedder (once; the model loads on first use)."""
    global _current
    if _current != name:
        from tvclient import set_embedder  # only in a tvclient with embeddings

        set_embedder(CommandEmbedder(name))
        _current = name
