import logging
import time
import grpc
from liteagent.network.protos import coordinator_pb2, coordinator_pb2_grpc
from liteagent.network.metrics import compute_latency_breakdown

logger = logging.getLogger("liteagent.network.client")
PROTOCOL_VERSION = 1

class WorkstationClient:
    def __init__(self, host: str, port: int = 50051, root_certificates: bytes = None, auth_token: str = None):
        self.target = f"{host}:{port}"
        self.root_certificates = root_certificates
        self.auth_token = auth_token

    def _get_channel(self):
        if self.root_certificates:
            creds = grpc.ssl_channel_credentials(root_certificates=self.root_certificates)
            return grpc.secure_channel(self.target, creds)
        return grpc.insecure_channel(self.target)

    def _get_metadata(self):
        if self.auth_token:
            return (("authorization", f"Bearer {self.auth_token}"),)
        return None

    def ping(self, timeout: float = 3.0) -> dict:
        """
        Sends Ping request and checks version compatibility.
        """
        try:
            with self._get_channel() as channel:
                stub = coordinator_pb2_grpc.WorkstationCoordinatorStub(channel)
                req = coordinator_pb2.PingRequest(protocol_version=PROTOCOL_VERSION)
                res = stub.Ping(req, timeout=timeout, metadata=self._get_metadata())
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
        except Exception as e:
            logger.warning("Warmup task failed (non-fatal): %s", e)

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
        timeout: float = 180.0,
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
            with self._get_channel() as channel:
                stub = coordinator_pb2_grpc.WorkstationCoordinatorStub(channel)
                res = stub.DispatchTask(req, timeout=timeout, metadata=self._get_metadata())
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

        # Compute latency breakdown
        client_e2e_ms = (client_receive_ts - client_send_ts) * 1000.0
        server_queue_ms = (res.server_start_compute_ts - res.server_receive_ts) * 1000.0
        server_compute_ms = (res.server_end_compute_ts - res.server_start_compute_ts) * 1000.0
        server_serialize_ms = res.server_serialize_duration_ms

        breakdown = compute_latency_breakdown(
            client_serialize_ms,
            client_deserialize_ms,
            client_e2e_ms,
            server_queue_ms,
            server_compute_ms,
            server_serialize_ms
        )

        out = {
            "response_text": response_text,
            "tokens_generated": tokens_generated,
            "prefill_tokens": prefill_tokens,
            "prefill_latency_ms": prefill_latency_ms,
            "generation_latency_ms": generation_latency_ms,
            "cache_hit_tier": cache_hit_tier,

            # Sync validation info
            "request_id": res.request_id,
            "protocol_version": res.protocol_version
        }
        out.update(breakdown)
        return out
