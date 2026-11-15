"""The embedders a run can choose between, by short name.

tvclient embeds each executed stateful command (BashToolCall.embed) and stores
the vector with it; the server pairs siblings whose vectors are similar, and
contraction compares those pairs.

In a training run the model lives in one service beside veRL, in its own
Python environment (`python embedding.py serve`, started by train_pilot.sh).
veRL pins an older transformers than the newer models need (ISSUES.md #28),
and one copy of the model then serves every rollout worker. Without a service
each process loads the model itself.

    TVCACHE_EMBEDDER       an alias below, or any sentence-transformers model name
    TVCACHE_EMBED_PORT     the service's port on this node; unset = load in this process
    TVCACHE_EMBED_DEVICE   where the model runs. Default cpu: the GPU belongs
                           to vLLM and the trainer, and commands are short

The service takes one JSON line per connection and answers with one:

    {"text": "<command>"} -> {"embedding": [...]}  or  {"error": "..."}
    {"op": "stats"}       -> {"model": ..., "count": ..., "seconds": ...}

This file imports nothing else from this package, so the service's
environment needs only sentence-transformers.
"""

import argparse
import json
import os
import signal
import socket
import socketserver
import sys
import threading
import time
from typing import Any, Callable, List, Optional

# alias -> model, and where the model needs them:
#   options     arguments for SentenceTransformer(...)
#   prompt      text put in front of every command before it is embedded
#   max_tokens  longer commands are cut here (default: the model's own limit)
EMBEDDERS = {
    "minilm": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "embeddinggemma2": {
        "model": "google/embeddinggemma-2",
        # Text only (no vision or audio encoder), and float32: its card says
        # float16 gives wrong vectors and bfloat16 is for GPUs.
        "options": {
            "config_kwargs": {"vision_config": None, "audio_config": None},
            "model_kwargs": {"torch_dtype": "float32"},
        },
        # The model is trained with a task prefix. Two commands are compared
        # with each other, so both get the sentence similarity one.
        "prompt": "task: sentence similarity | query: ",
        "max_tokens": 512,
    },
}


def resolve(name: str) -> dict:
    """The settings for an alias or a full model name."""
    if name in EMBEDDERS:
        return EMBEDDERS[name]
    for settings in EMBEDDERS.values():
        if name == settings["model"]:
            return settings
    return {"model": name}


class CommandEmbedder:
    """A sentence-transformers model, loaded on first use. It has the one
    method tvclient's Embedder interface asks for, encode(text)."""

    def __init__(self, name: str, device: Optional[str] = None):
        settings = resolve(name)
        self.model_name: str = settings["model"]
        self.options: dict = settings.get("options", {})
        self.prompt: Optional[str] = settings.get("prompt")
        self.max_tokens: Optional[int] = settings.get("max_tokens")
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
            extra = {"prompt": self.prompt} if self.prompt else {}
            return self._model.encode(text, **extra).tolist()


class RemoteEmbedder:
    """The embedding service on this node (serve below)."""

    def __init__(self, port: int, timeout: float = 120.0):
        self.port = port
        self.timeout = timeout

    def _call(self, request: dict) -> dict:
        with socket.create_connection(("127.0.0.1", self.port), timeout=self.timeout) as sock:
            sock.sendall((json.dumps(request) + "\n").encode())
            reply = sock.makefile("r").readline()
        return json.loads(reply) if reply else {}

    def encode(self, text: str) -> List[float]:
        reply = self._call({"text": text})
        if "embedding" not in reply:
            raise RuntimeError(f"embedding service on port {self.port}: {reply.get('error', 'no reply')}")
        return reply["embedding"]

    def stats(self) -> dict:
        return self._call({"op": "stats"})


_current: Optional[tuple] = None


def use_embedder(name: str, port: int = 0) -> None:
    """Makes `name` this process's embedder (once): the service on `port`, or
    with port 0 a model loaded here on first use."""
    global _current
    if _current != (name, port):
        from tvclient import set_embedder  # only in a tvclient with embeddings

        set_embedder(RemoteEmbedder(port) if port else CommandEmbedder(name))
        _current = (name, port)


def make_server(encode: Callable[[str], List[float]], model: str, port: int,
                host: str = "127.0.0.1") -> socketserver.ThreadingTCPServer:
    """The service around any encode(text); `server.stats` counts its work."""
    stats = {"model": model, "count": 0, "errors": 0, "seconds": 0.0}
    lock = threading.Lock()  # one command at a time, so `seconds` is model time, not time in line

    class Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            line = self.rfile.readline()
            if not line.strip():
                return  # a port probe: connected and left
            try:
                request = json.loads(line)
                if request.get("op") == "stats":
                    reply = dict(stats)
                else:
                    with lock:
                        start = time.perf_counter()
                        reply = {"embedding": encode(request["text"])}
                        stats["count"] += 1
                        stats["seconds"] += time.perf_counter() - start
            except Exception as error:  # the client reports it; the service keeps going
                stats["errors"] += 1
                reply = {"error": f"{type(error).__name__}: {error}"}
            self.wfile.write((json.dumps(reply) + "\n").encode())

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True
        request_queue_size = 256  # every rollout of a batch can ask at once

    server = Server((host, port), Handler)
    server.stats = stats
    return server


def serve(name: str, port: int, threads: int = 8) -> None:
    """Loads the model, then opens the port, so an open port means ready."""
    import torch

    # Commands are short: more threads than this only wait on each other.
    torch.set_num_threads(max(1, min(threads, os.cpu_count() or 1)))
    embedder = CommandEmbedder(name)
    dimensions = len(embedder.encode("ls -la /app"))
    server = make_server(embedder.encode, embedder.model_name, port)
    print(f"embedding service: {embedder.model_name}, {dimensions} dimensions, port {port}", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    count, seconds = server.stats["count"], server.stats["seconds"]
    mean = f", {1000 * seconds / count:.0f} ms each" if count else ""
    print(f"embedding service: embedded {count} commands{mean}, {server.stats['errors']} errors", flush=True)


def check() -> int:
    """Loads every model above and embeds one command with it (setup.sh)."""
    failed = 0
    for alias in EMBEDDERS:
        embedder = CommandEmbedder(alias)
        try:
            dimensions = len(embedder.encode("ls -la /app"))
            print(f"embedder {alias} ok: {embedder.model_name}, {dimensions} dimensions")
        except Exception as error:
            failed += 1
            print(f"embedder {alias} FAILED ({embedder.model_name}): {type(error).__name__}: {error}")
    return failed


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    serving = commands.add_parser("serve", help="serve one model to this node's rollouts")
    serving.add_argument("name", help=f"{', '.join(EMBEDDERS)} or a sentence-transformers model name")
    serving.add_argument("--port", type=int, required=True)
    commands.add_parser("check", help="load every model once")
    args = parser.parse_args(argv)
    if args.command == "check":
        return 1 if check() else 0
    serve(args.name, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
