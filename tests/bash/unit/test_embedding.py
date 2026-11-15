"""Unit tests (no services, no model downloads): the embedder table, the
prompt a model asks for, and the embedding service with its client."""

import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

from tvcache_workloads.bash import embedding
from tvcache_workloads.bash.embedding import CommandEmbedder, RemoteEmbedder, make_server, resolve
from tvcache_workloads.bash.verl.session import CacheSettings


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeModel:
    """Stands in for sentence_transformers.SentenceTransformer."""

    loaded = []

    def __init__(self, name, **options):
        self.name, self.options, self.calls = name, options, []
        self.max_seq_length = None
        FakeModel.loaded.append(self)

    def encode(self, text, **extra):
        self.calls.append((text, extra))
        return FakeVector([float(len(text)), 1.0])


@pytest.fixture
def fake_models(monkeypatch):
    FakeModel.loaded = []
    fake = types.SimpleNamespace(SentenceTransformer=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    return FakeModel.loaded


def test_an_alias_a_full_name_and_an_unknown_name_resolve():
    assert resolve("minilm") == {"model": "sentence-transformers/all-MiniLM-L6-v2"}
    gemma = resolve("embeddinggemma2")
    assert gemma["model"] == "google/embeddinggemma-2"
    assert resolve("google/embeddinggemma-2") is gemma
    assert resolve("someone/some-model") == {"model": "someone/some-model"}


def test_embeddinggemma_gets_its_task_prefix_and_loads_text_only(fake_models):
    embedder = CommandEmbedder("embeddinggemma2")
    assert embedder.encode("ls -la /app") == [11.0, 1.0]
    (model,) = fake_models
    assert model.name == "google/embeddinggemma-2" and model.max_seq_length == 512
    assert model.options["device"] == "cpu"
    assert model.options["config_kwargs"] == {"vision_config": None, "audio_config": None}
    assert model.options["model_kwargs"] == {"torch_dtype": "float32"}
    # the command itself is unchanged; the prefix goes in as the model's prompt
    assert model.calls == [("ls -la /app", {"prompt": "task: sentence similarity | query: "})]


def test_a_model_without_a_prefix_gets_none(fake_models):
    embedder = CommandEmbedder("minilm", device="cuda")
    embedder.encode("ls")
    embedder.encode("pwd")
    (model,) = fake_models  # loaded once
    assert model.options == {"device": "cuda"} and model.max_seq_length is None
    assert model.calls == [("ls", {}), ("pwd", {})]


@pytest.fixture
def service():
    def encode(text):
        if text == "boom":
            raise ValueError("cannot embed this")
        return [float(len(text)), 1.0]

    server = make_server(encode, "stub-model", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_the_service_embeds_for_other_processes(service):
    remote = RemoteEmbedder(service.server_address[1])
    assert remote.encode("ls -la /app") == [11.0, 1.0]
    assert remote.encode("echo 'two\nlines' > f") == [20.0, 1.0]
    stats = remote.stats()
    assert stats["model"] == "stub-model" and stats["count"] == 2 and stats["errors"] == 0


def test_a_failed_embedding_is_reported_and_the_service_keeps_going(service):
    remote = RemoteEmbedder(service.server_address[1])
    with pytest.raises(RuntimeError, match="ValueError: cannot embed this"):
        remote.encode("boom")
    assert remote.encode("ls") == [2.0, 1.0]
    assert remote.stats()["errors"] == 1


def test_a_port_probe_is_not_counted(service):
    import socket

    socket.create_connection(service.server_address, timeout=2).close()  # as wait_for_port does
    remote = RemoteEmbedder(service.server_address[1])
    assert remote.encode("ls") == [2.0, 1.0]
    assert remote.stats()["errors"] == 0 and remote.stats()["count"] == 1


def test_no_service_on_the_port_is_an_error():
    server = make_server(lambda text: [1.0], "stub-model", port=0)
    port = server.server_address[1]
    server.server_close()
    with pytest.raises(OSError):
        RemoteEmbedder(port, timeout=2).encode("ls")


def test_rollouts_use_the_service_when_its_port_is_set(monkeypatch):
    import tvclient

    chosen = []
    monkeypatch.setattr(tvclient, "set_embedder", chosen.append)
    monkeypatch.setattr(embedding, "_current", None)
    embedding.use_embedder("embeddinggemma2", 7881)
    embedding.use_embedder("embeddinggemma2", 7881)  # same choice: nothing new
    embedding.use_embedder("minilm")
    assert isinstance(chosen[0], RemoteEmbedder) and chosen[0].port == 7881
    assert isinstance(chosen[1], CommandEmbedder) and chosen[1].model_name.endswith("all-MiniLM-L6-v2")
    assert len(chosen) == 2


def test_settings_read_the_service_port(monkeypatch):
    assert CacheSettings.from_config({}).embed_port == 0
    monkeypatch.setenv("TVCACHE_EMBED_PORT", "7881")
    assert CacheSettings.from_config({}).embed_port == 7881


def test_the_file_runs_on_its_own_as_the_service_does():
    # The service's environment has no tvclient and no tvcache_workloads:
    # -S -I leaves only the standard library.
    script = Path(embedding.__file__)
    result = subprocess.run([sys.executable, "-S", "-I", str(script), "serve", "--help"],
                            capture_output=True, text=True, cwd=script.parent.parent)
    assert result.returncode == 0, result.stderr
    assert "minilm, embeddinggemma2" in result.stdout
