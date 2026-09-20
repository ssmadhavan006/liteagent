import numpy as np

# Evaluation plan requires a fixed seed for reproducibility. Sampling must not
# draw from numpy's global RNG, whose state depends on unrelated callers.
_DEFAULT_SAMPLING_SEED = 42
_rng = np.random.default_rng(_DEFAULT_SAMPLING_SEED)


def reset_sampling_rng(seed: int = _DEFAULT_SAMPLING_SEED) -> None:
    """Restores the sampler RNG to a known state so a run can be reproduced."""
    global _rng
    _rng = np.random.default_rng(seed)


class InferenceEngine:
    """
    Centralized inference helper providing correct logits indexing for token generation.
    Fixes issue where naive numpy slicing evaluates initial prompt logits instead of the latest token.
    """
    @staticmethod
    def get_last_logits(llama_instance) -> np.ndarray:
        """
        Reads the logits for the most recently evaluated token.

        Order matters. With `logits_all` left at its default, `llama.eval_logits`
        and `llama._scores` are all-zero on real Llama instances (verified on
        llama-cpp-python 0.3.4, CPU and CUDA): only the low-level context pointer
        holds the decoded values. Preferring either high-level accessor makes
        argmax return token 0 and every generation decode as "!!!!!!".

        The context pointer is therefore the primary source, matching the design
        described in architecture.md 10.1; the high-level accessors remain as a
        fallback for mock instances in tests, which have no `_ctx`.
        """
        if getattr(llama_instance, "_ctx", None) is not None:
            n_vocab = getattr(llama_instance, "_n_vocab", None)
            if n_vocab is None and callable(getattr(llama_instance, "n_vocab", None)):
                n_vocab = llama_instance.n_vocab()
            if n_vocab is None and hasattr(llama_instance, "vocab_size"):
                n_vocab = llama_instance.vocab_size() if callable(llama_instance.vocab_size) else llama_instance.vocab_size
            if n_vocab is None:
                raise RuntimeError("Could not determine vocab size for Llama instance logits retrieval.")

            logits_ptr = llama_instance._ctx.get_logits()
            # With logits_all disabled llama.cpp exposes only the final row, so
            # reading a full n_tokens x n_vocab block would run off the buffer.
            raw_array = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,))
            return np.array(raw_array, dtype=np.float32)

        if hasattr(llama_instance, "eval_logits"):
            eval_logits = llama_instance.eval_logits
            if len(eval_logits) > 0:
                return np.array(eval_logits[-1], dtype=np.float32)

        if hasattr(llama_instance, "_scores") and hasattr(llama_instance, "n_tokens"):
            if llama_instance.n_tokens > 0:
                return llama_instance._scores[llama_instance.n_tokens - 1, :]

        raise RuntimeError("Unable to extract logits from Llama instance.")

    @staticmethod
    def sample_next_token(llama_instance, temperature: float = 0.0, top_p: float = 0.9) -> int:
        logits = InferenceEngine.get_last_logits(llama_instance)
        if temperature <= 0.0:
            return int(np.argmax(logits))

        # Temperature scaling
        scaled_logits = logits / max(temperature, 1e-5)
        # Softmax
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        probs = exp_logits / np.sum(exp_logits)

        # Top-p (nucleus) filtering
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Keep tokens within top_p threshold
        cutoff_index = np.searchsorted(cumulative_probs, top_p) + 1
        valid_indices = sorted_indices[:cutoff_index]
        valid_probs = probs[valid_indices]
        valid_probs /= np.sum(valid_probs)

        return int(_rng.choice(valid_indices, p=valid_probs))

    @staticmethod
    def generate_tokens(
        llama_instance,
        max_tokens: int,
        temperature: float = 0.0,
        timings: dict = None,
    ) -> tuple[list[int], str]:
        """
        Greedy/sampled decode loop.

        `timings`, when supplied, receives `ttft_ms` - the interval from entering
        this call to the first token being produced. Time-to-first-token is the
        metric the cache hypothesis (H2) is stated in, since restoring a context
        replaces prefill rather than generation.
        """
        import time as _time
        start = _time.perf_counter()
        response_tokens = []
        for _ in range(max_tokens):
            next_token = InferenceEngine.sample_next_token(llama_instance, temperature=temperature)
            if timings is not None and not response_tokens:
                timings["ttft_ms"] = (_time.perf_counter() - start) * 1000.0
            if next_token == llama_instance.token_eos():
                break
            response_tokens.append(next_token)
            llama_instance.eval([next_token])

        response_text = llama_instance.detokenize(response_tokens).decode("utf-8", errors="ignore")
        return response_tokens, response_text
