"""
Guards against silent decode corruption.

The structural tests assert that a response exists and that metrics are
populated, which stays true when every generated token is token 0. That gap let
an all-zero logits source ship: `eval_logits` and `_scores` read back as zeros
on real Llama instances, so greedy sampling emitted "!!!!!!" while every test
still passed. These tests assert on content, not shape.
"""

import numpy as np
import pytest

from liteagent.utils.inference import InferenceEngine, reset_sampling_rng


class _StubCtx:
    """Mimics llama.cpp's low-level context exposing only the final logits row."""

    def __init__(self, logits):
        self._logits = np.asarray(logits, dtype=np.float32)

    def get_logits(self):
        import ctypes
        arr = (ctypes.c_float * len(self._logits))(*self._logits)
        return ctypes.cast(arr, ctypes.POINTER(ctypes.c_float))


class _StubLlama:
    """Real instances expose _ctx; the zeroed high-level mirrors must be ignored."""

    def __init__(self, logits, n_vocab):
        self._ctx = _StubCtx(logits)
        self._n_vocab = n_vocab
        self.n_tokens = 4
        # Deliberately zeroed, mirroring observed llama-cpp-python behaviour.
        self.eval_logits = [[0.0] * n_vocab]
        self._scores = np.zeros((8, n_vocab), dtype=np.float32)


def test_context_pointer_wins_over_zeroed_high_level_accessors():
    logits = [0.0] * 32
    logits[7] = 12.5
    llama = _StubLlama(logits, n_vocab=32)

    out = InferenceEngine.get_last_logits(llama)
    assert int(np.argmax(out)) == 7, "must not read the all-zero eval_logits mirror"
    assert float(np.max(out)) == pytest.approx(12.5)


def test_greedy_sampling_does_not_collapse_to_token_zero():
    logits = [0.0] * 64
    logits[42] = 9.0
    llama = _StubLlama(logits, n_vocab=64)

    token = InferenceEngine.sample_next_token(llama, temperature=0.0)
    assert token == 42
    assert token != 0, "token 0 is the signature of reading an all-zero logits buffer"


def test_mock_without_ctx_still_falls_back():
    class MockLlama:
        _ctx = None
        eval_logits = [[0.1, 5.0, 0.2]]

    assert int(np.argmax(InferenceEngine.get_last_logits(MockLlama()))) == 1


def test_sampling_is_reproducible_across_rng_resets():
    logits = [1.0, 2.0, 3.0, 4.0] * 8
    llama = _StubLlama(logits, n_vocab=32)

    reset_sampling_rng(42)
    first = [InferenceEngine.sample_next_token(llama, temperature=0.8) for _ in range(12)]
    reset_sampling_rng(42)
    second = [InferenceEngine.sample_next_token(llama, temperature=0.8) for _ in range(12)]

    assert first == second, "temperature sampling must be reproducible under a fixed seed"


@pytest.mark.slow
def test_real_model_produces_meaningful_text():
    """End-to-end guard: a real model must not decode to repeated punctuation."""
    llama_cpp = pytest.importorskip("llama_cpp")
    import liteagent  # noqa: F401  - CUDA DLL path
    from liteagent.utils.model_resolver import resolve_model_path

    try:
        path = resolve_model_path("llama3.2:1b")
    except Exception:
        pytest.skip("llama3.2:1b weights unavailable")

    model = llama_cpp.Llama(model_path=path, n_ctx=512, n_gpu_layers=0, verbose=False, seed=42)
    model.reset()
    model.eval(model.tokenize(b"Question: What is 17 times 23?\nAnswer:"))
    tokens, text = InferenceEngine.generate_tokens(model, 24, temperature=0.0)

    assert tokens, "no tokens generated"
    assert len(set(tokens)) > 1, f"degenerate output: {text!r}"
    assert not all(t == 0 for t in tokens), "all-token-zero output indicates zeroed logits"
    assert "391" in text, f"expected the correct product in {text!r}"
