import time
import grpc
from liteagent.network.protos import coordinator_pb2, coordinator_pb2_grpc

PROTOCOL_VERSION = 1

class WorkstationClient:
    def __init__(self, host: str, port: int = 50051):
        self.target = f"{host}:{port}"

    def ping(self, timeout: float = 3.0) -> dict:
        """
        Sends Ping request and checks version compatibility.
        """
        try:
            with grpc.insecure_channel(self.target) as channel:
                stub = coordinator_pb2_grpc.WorkstationCoordinatorStub(channel)
                req = coordinator_pb2.PingRequest(protocol_version=PROTOCOL_VERSION)
                res = stub.Ping(req, timeout=timeout)
                return {
                    "protocol_version": res.protocol_version,
                    "model_version": res.model_version,
                    "compatible": res.compatible,
                    "llama_cpp_version": res.llama_cpp_version,
                    "liteagent_version": res.liteagent_version
                }
        except grpc.RpcError as e:
            raise ConnectionError(f"Failed to ping workstation coordinator: {e}") from e

    def warmup(self):
        """
        Sends a single dummy task to warm up model loading and VRAM page faults on the workstation.
        """
        try:
            self.dispatch_task(
                request_id="warmup-id",
                task_id="warmup-task",
                session_id="warmup-session",
                agent_role="Planner",
                prompt="ping",
                system_prompt="ping",
                max_tokens=1
            )
        except Exception:
            pass

    def dispatch_task(
        self,
        request_id: str,
        task_id: str,
        session_id: str,
        agent_role: str,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 100,
        timeout: float = 30.0,
        cache_disabled: bool = False
    ) -> dict:
        """
        Dispatches high-complexity task and returns metrics + output dictionary.
        """
        # Client serialization
        ser_start = time.perf_counter()
        client_send_ts = time.time()
        
        req = coordinator_pb2.TaskRequest(
            task_id=task_id,
            session_id=session_id,
            agent_role=agent_role,
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            protocol_version=PROTOCOL_VERSION,
            request_id=request_id,
            client_send_ts=client_send_ts,
            cache_disabled=cache_disabled
        )
        client_serialize_ms = (time.perf_counter() - ser_start) * 1000.0

        # Remote execution over network
        try:
            with grpc.insecure_channel(self.target) as channel:
                stub = coordinator_pb2_grpc.WorkstationCoordinatorStub(channel)
                res = stub.DispatchTask(req, timeout=timeout)
                client_receive_ts = time.time()
        except grpc.RpcError as e:
            raise ConnectionError(f"RPC dispatch failed: {e}") from e

        # Client deserialization
        deser_start = time.perf_counter()
        response_text = res.response_text
        tokens_generated = res.tokens_generated
        prefill_tokens = res.prefill_tokens
        prefill_latency_ms = res.prefill_latency_ms
        generation_latency_ms = res.generation_latency_ms
        cache_hit_tier = res.cache_hit_tier
        client_deserialize_ms = (time.perf_counter() - deser_start) * 1000.0

        # Compute durations
        client_e2e_ms = (client_receive_ts - client_send_ts) * 1000.0
        server_queue_ms = (res.server_start_compute_ts - res.server_receive_ts) * 1000.0
        server_compute_ms = (res.server_end_compute_ts - res.server_start_compute_ts) * 1000.0
        server_serialize_ms = res.server_serialize_duration_ms

        # Derive communication overhead (network RTT + OS/gRPC buffering/queuing)
        communication_overhead_ms = client_e2e_ms - (
            client_serialize_ms +
            server_queue_ms +
            server_compute_ms +
            server_serialize_ms +
            client_deserialize_ms
        )

        return {
            "response_text": response_text,
            "tokens_generated": tokens_generated,
            "prefill_tokens": prefill_tokens,
            "prefill_latency_ms": prefill_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "cache_hit_tier": cache_hit_tier,
            
            # Detailed Latency Metrics
            "client_serialize_ms": round(client_serialize_ms, 2),
            "client_deserialize_ms": round(client_deserialize_ms, 2),
            "client_e2e_ms": round(client_e2e_ms, 2),
            "server_queue_ms": round(server_queue_ms, 2),
            "server_compute_ms": round(server_compute_ms, 2),
            "server_serialize_ms": round(server_serialize_ms, 2),
            "communication_overhead_ms": round(communication_overhead_ms, 2),
            
            # Sync validation info
            "request_id": res.request_id,
            "protocol_version": res.protocol_version
        }
