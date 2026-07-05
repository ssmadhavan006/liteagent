def compute_latency_breakdown(
    client_serialize_ms: float,
    client_deserialize_ms: float,
    client_e2e_ms: float,
    server_queue_ms: float,
    server_compute_ms: float,
    server_serialize_ms: float
) -> dict:
    """
    Computes a clock-drift-independent breakdown of remote task execution latency.
    All incoming metrics are measured relative to local client or server clocks,
    preventing clock synchronization errors from yielding negative delays.
    """
    # Derived communication & framework transport overhead
    communication_overhead_ms = client_e2e_ms - (
        client_serialize_ms +
        server_queue_ms +
        server_compute_ms +
        server_serialize_ms +
        client_deserialize_ms
    )
    
    return {
        "client_serialize_ms": round(client_serialize_ms, 2),
        "client_deserialize_ms": round(client_deserialize_ms, 2),
        "client_e2e_ms": round(client_e2e_ms, 2),
        "server_queue_ms": round(server_queue_ms, 2),
        "server_compute_ms": round(server_compute_ms, 2),
        "server_serialize_ms": round(server_serialize_ms, 2),
        "communication_overhead_ms": round(communication_overhead_ms, 2)
    }
