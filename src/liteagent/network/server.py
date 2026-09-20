import time
import hashlib
import grpc
import logging
import threading
from concurrent import futures
import llama_cpp
from llama_cpp import Llama

from liteagent.network.protos import coordinator_pb2, coordinator_pb2_grpc
from liteagent.cache import KVCacheManager, CacheRestoreError
from liteagent.utils.model_resolver import resolve_model_path
from liteagent.utils.inference import InferenceEngine
from liteagent.config import (
    PROTOCOL_VERSION,
    LITEAGENT_VERSION,
    MAX_TOKENS_CAP,
    MAX_PROMPT_LEN,
    DEFAULT_WORKSTATION_HOST,
    DEFAULT_WORKSTATION_PORT,
    WORKSTATION_N_GPU_LAYERS
)

logger = logging.getLogger("liteagent.network.server")

class WorkstationCoordinatorServicer(coordinator_pb2_grpc.WorkstationCoordinatorServicer):
    def __init__(self, cache_manager: KVCacheManager, auth_token: str = None):
        self.cache_manager = cache_manager
        self.auth_token = auth_token
        self.models = {}  # Dict[model_tag, Llama]
        self.model_locks = {}
        self.locks_lock = threading.Lock()

        # Simple token-bucket rate limiter: max 60 requests per minute
        self.rate_limit_lock = threading.Lock()
        self.last_tokens = 60.0
        self.last_update = time.time()

    def _verify_auth(self, context) -> bool:
        if not self.auth_token:
            return True
        metadata = dict(context.invocation_metadata())
        auth_header = metadata.get("authorization", "")
        expected = f"Bearer {self.auth_token}"
        if auth_header != expected:
            context.set_code(grpc.StatusCode.UNAUTHENTICATED)
            context.set_details("Invalid or missing authorization token.")
            return False
        return True

    def _check_rate_limit(self) -> bool:
        with self.rate_limit_lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            # Replenish 1 token per second up to max capacity of 60
            self.last_tokens = min(60.0, self.last_tokens + elapsed * 1.0)
            if self.last_tokens >= 1.0:
                self.last_tokens -= 1.0
                return True
            return False

    def Ping(self, request, context):
        if not self._verify_auth(context):
            return coordinator_pb2.PingResponse()

        client_ver = request.protocol_version
        compatible = (client_ver == PROTOCOL_VERSION)

        # Get llama-cpp version programmatically
        try:
            llama_ver = llama_cpp.__version__
        except AttributeError:
            llama_ver = "0.3.1"

        return coordinator_pb2.PingResponse(
            protocol_version=PROTOCOL_VERSION,
            model_version="llama3.1:8b",
            compatible=compatible,
            llama_cpp_version=llama_ver,
            liteagent_version=LITEAGENT_VERSION
        )

    def DispatchTask(self, request, context):
        if not self._verify_auth(context):
            return coordinator_pb2.TaskResponse()

        # Input Validation: Protocol version
        if request.protocol_version != PROTOCOL_VERSION:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details(f"Protocol version mismatch: server={PROTOCOL_VERSION}, client={request.protocol_version}")
            return coordinator_pb2.TaskResponse()

        # Input Validation: Prompt length & max_tokens bounds
        if len(request.prompt) > MAX_PROMPT_LEN or len(request.system_prompt) > MAX_PROMPT_LEN:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Prompt length exceeds maximum allowed limit ({MAX_PROMPT_LEN} characters).")
            return coordinator_pb2.TaskResponse()

        if request.max_tokens <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("max_tokens must be positive.")
            return coordinator_pb2.TaskResponse()

        # Clamp max_tokens to safety cap
        effective_max_tokens = min(request.max_tokens, MAX_TOKENS_CAP)

        # Rate Limiting
        if not self._check_rate_limit():
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Rate limit exceeded. Please retry later.")
            return coordinator_pb2.TaskResponse()

        try:
            server_receive_ts = time.time()

            # Log SERVER_RECEIVED event
            self.cache_manager._write_log({
                "request_id": request.request_id,
                "component": "workstation_server",
                "event": "SERVER_RECEIVED",
                "timestamp": "" # Filled by _write_log
            })

            model_tag = "llama3.1:8b"
            ctx_size = 4096
            prompt_hash = hashlib.sha256(request.prompt.encode("utf-8")).hexdigest()

            # 1. Resolve model and ensure loaded
            model_path = resolve_model_path(model_tag)
            with self.locks_lock:
                if model_tag not in self.models:
                    # Workstation offloads to the RTX 5070 so the Hot tier is
                    # VRAM-resident and nvidia-smi energy sampling measures
                    # real inference draw rather than idle draw.
                    self.models[model_tag] = Llama(
                        model_path=model_path,
                        n_ctx=ctx_size,
                        n_gpu_layers=WORKSTATION_N_GPU_LAYERS,
                        verbose=False,
                        seed=42,
                    )
                if model_tag not in self.model_locks:
                    self.model_locks[model_tag] = threading.Lock()
                llama = self.models[model_tag]
                lock = self.model_locks[model_tag]

            lock.acquire()
            try:
                # 2. Restore cache state
                if request.cache_disabled:
                    cache_hit_tier = "MISS"
                else:
                    try:
                        cache_hit_tier = self.cache_manager.load_cache(
                            session_key=request.session_id,
                            llama_instance=llama,
                            model_tag=model_tag,
                            ctx_size=ctx_size,
                            prompt_hash=prompt_hash
                        )
                    except CacheRestoreError as e:
                        logger.warning("Cache restoration failed for request %s: %s", request.request_id, e)
                        self.cache_manager._write_log({
                            "request_id": request.request_id,
                            "component": "workstation_server",
                            "event": "RESTORE_FAILED",
                            "error": str(e)
                        })
                        context.set_code(grpc.StatusCode.INTERNAL)
                        context.set_details("Cache restoration failed.")
                        return coordinator_pb2.TaskResponse()

                server_start_compute_ts = time.time()

                # Log SERVER_COMPUTE_STARTED
                self.cache_manager._write_log({
                    "request_id": request.request_id,
                    "component": "workstation_server",
                    "event": "SERVER_COMPUTE_STARTED",
                    "timestamp": ""
                })

                # 3. Evaluate prompt (prefill)
                prefill_start = time.time()
                if cache_hit_tier == "MISS":
                    llama.reset()
                    # Prefill both system instructions and prompt
                    full_prompt = f"{request.system_prompt}\n{request.prompt}".encode("utf-8")
                    tokens = llama.tokenize(full_prompt)
                    llama.eval(tokens)
                    prefill_tokens = len(tokens)
                else:
                    # Evaluate new prompt on top of restored state
                    tokens = llama.tokenize(request.prompt.encode("utf-8"))
                    llama.eval(tokens)
                    prefill_tokens = len(tokens)

                prefill_latency_ms = (time.time() - prefill_start) * 1000.0

                # 4. Generate tokens
                gen_start = time.time()
                response_tokens, response_text = InferenceEngine.generate_tokens(
                    llama,
                    effective_max_tokens,
                    temperature=request.temperature if hasattr(request, "temperature") else 0.0
                )
                generation_latency_ms = (time.time() - gen_start) * 1000.0
                server_end_compute_ts = time.time()

                # Log SERVER_COMPUTE_FINISHED
                self.cache_manager._write_log({
                    "request_id": request.request_id,
                    "component": "workstation_server",
                    "event": "SERVER_COMPUTE_FINISHED",
                    "timestamp": ""
                })

                # 5. Save updated cache state
                if request.cache_disabled:
                    pass
                else:
                    try:
                        self.cache_manager.save_cache(
                            session_key=request.session_id,
                            agent_role=request.agent_role,
                            llama_instance=llama,
                            model_tag=model_tag,
                            ctx_size=ctx_size,
                            prompt_hash=prompt_hash
                        )
                    except Exception as e:
                        logger.warning("Cache save failed for request %s: %s", request.request_id, e)
                        self.cache_manager._write_log({
                            "request_id": request.request_id,
                            "component": "workstation_server",
                            "event": "SAVE_FAILED",
                            "error": str(e)
                        })
                        context.set_code(grpc.StatusCode.INTERNAL)
                        context.set_details("Cache serialization failed.")
                        return coordinator_pb2.TaskResponse()

                # Calculate serialization time for response
                server_serialize_duration_ms = (time.time() - server_end_compute_ts) * 1000.0

                return coordinator_pb2.TaskResponse(
                    task_id=request.task_id,
                    response_text=response_text,
                    tokens_generated=len(response_tokens),
                    prefill_tokens=prefill_tokens,
                    prefill_latency_ms=prefill_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    cache_hit_tier=cache_hit_tier,
                    protocol_version=PROTOCOL_VERSION,
                    request_id=request.request_id,
                    server_receive_ts=server_receive_ts,
                    server_start_compute_ts=server_start_compute_ts,
                    server_end_compute_ts=server_end_compute_ts,
                    server_serialize_duration_ms=server_serialize_duration_ms
                )
            finally:
                lock.release()
        except Exception as e:
            logger.error("Exception in WorkstationCoordinatorServicer.DispatchTask: %s", e, exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error during task execution.")
            return coordinator_pb2.TaskResponse()

def serve(
    cache_manager: KVCacheManager,
    host: str = DEFAULT_WORKSTATION_HOST,
    port: int = DEFAULT_WORKSTATION_PORT,
    auth_token: str = None,
    ssl_credentials: grpc.ServerCredentials = None
) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    coordinator_pb2_grpc.add_WorkstationCoordinatorServicer_to_server(
        WorkstationCoordinatorServicer(cache_manager, auth_token=auth_token), server
    )
    bind_address = f"{host}:{port}"
    if ssl_credentials:
        server.add_secure_port(bind_address, ssl_credentials)
    else:
        server.add_insecure_port(bind_address)
    server.start()
    logger.info("Workstation Coordinator Server running on %s...", bind_address)
    return server

