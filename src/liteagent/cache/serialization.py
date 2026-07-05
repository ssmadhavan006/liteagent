import time
import pickle

class SerializationError(Exception):
    pass

def serialize_llama_state(llama_instance) -> tuple[bytes, float]:
    """
    Saves state from Llama instance and serializes it using pickle.
    Returns (serialized_bytes, save_state_ms).
    """
    start_time = time.perf_counter()
    try:
        state = llama_instance.save_state()
        serialized_bytes = pickle.dumps(state)
    except Exception as e:
        raise SerializationError(f"Failed to serialize llama state: {e}") from e
        
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return serialized_bytes, latency_ms

def deserialize_llama_state(llama_instance, state_bytes: bytes) -> float:
    """
    Deserializes state bytes and loads it into the Llama instance.
    Returns load_state_ms.
    """
    start_time = time.perf_counter()
    try:
        state = pickle.loads(state_bytes)
        llama_instance.load_state(state)
    except Exception as e:
        raise SerializationError(f"Failed to deserialize/load llama state: {e}") from e
        
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    return latency_ms
