import time
import json
import struct
import numpy as np
from llama_cpp import LlamaState

class SerializationError(Exception):
    pass

def serialize_llama_state(llama_instance) -> tuple[bytes, float]:
    """
    Saves state from Llama instance and serializes it using safe binary layout.
    Returns (serialized_bytes, save_state_ms).
    """
    start_time = time.perf_counter()
    try:
        state = llama_instance.save_state()
        if isinstance(state, LlamaState):
            scores_bytes = state.scores.tobytes()
            metadata = {
                "n_tokens": int(state.n_tokens),
                "seed": int(state.seed),
                "input_ids": [int(x) for x in state.input_ids],
                "scores_shape": [int(x) for x in state.scores.shape],
                "scores_dtype": str(state.scores.dtype),
                "llama_state_size": int(state.llama_state_size),
                "scores_bytes_len": len(scores_bytes),
            }
            meta_json = json.dumps(metadata).encode("utf-8")
            # Layout: [Magic 4B: LMA1][Meta len 4B][Meta JSON][scores bytes][llama_state bytes]
            header = b"LMA1" + struct.pack(">I", len(meta_json)) + meta_json
            serialized_bytes = header + scores_bytes + bytes(state.llama_state)
        else:
            # Fallback for mock objects in testing
            raw = getattr(state, "llama_state", b"")
            meta_json = json.dumps({"raw_mock": True}).encode("utf-8")
            serialized_bytes = b"LMA1" + struct.pack(">I", len(meta_json)) + meta_json + raw
    except Exception as e:
        raise SerializationError(f"Failed to serialize llama state: {e}") from e

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return serialized_bytes, latency_ms

def deserialize_llama_state(llama_instance, state_bytes: bytes) -> float:
    """
    Deserializes safe state bytes and loads it into the Llama instance.
    Returns load_state_ms.
    """
    start_time = time.perf_counter()
    try:
        if not state_bytes.startswith(b"LMA1"):
            raise ValueError("Invalid magic bytes in cache state payload.")

        meta_len = struct.unpack(">I", state_bytes[4:8])[0]
        meta_json_bytes = state_bytes[8 : 8 + meta_len]
        metadata = json.loads(meta_json_bytes.decode("utf-8"))

        offset = 8 + meta_len
        if metadata.get("raw_mock"):
            mock_data = state_bytes[offset:]
            if hasattr(llama_instance, "load_state"):
                class MockState:
                    def __init__(self, data):
                        self.llama_state = data
                        self.llama_state_size = len(data)
                llama_instance.load_state(MockState(mock_data))
        else:
            scores_len = metadata["scores_bytes_len"]
            scores_data = state_bytes[offset : offset + scores_len]
            llama_state_data = state_bytes[offset + scores_len :]

            if len(llama_state_data) != metadata["llama_state_size"]:
                raise SerializationError(
                    f"llama_state_data length ({len(llama_state_data)}) does not match metadata llama_state_size ({metadata['llama_state_size']})"
                )

            scores_array = np.frombuffer(scores_data, dtype=metadata["scores_dtype"]).reshape(metadata["scores_shape"])

            state = LlamaState(
                scores=scores_array,
                input_ids=metadata["input_ids"],
                n_tokens=metadata["n_tokens"],
                llama_state=llama_state_data,
                llama_state_size=metadata["llama_state_size"],
                seed=metadata["seed"],
            )
            llama_instance.load_state(state)
    except Exception as e:
        raise SerializationError(f"Failed to deserialize/load llama state: {e}") from e

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return latency_ms

