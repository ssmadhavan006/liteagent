import os
import time
import hashlib
import grpc
import threading
from concurrent import futures
import llama_cpp
from llama_cpp import Llama

from liteagent.network.protos import coordinator_pb2, coordinator_pb2_grpc
from liteagent.cache import KVCacheManager, CacheRestoreError
from liteagent.utils.model_resolver import resolve_model_path

PROTOCOL_VERSION = 1
LITEAGENT_VERSION = "1.0.0"

class WorkstationCoordinatorServicer(coordinator_pb2_grpc.WorkstationCoordinatorServicer):
    def __init__(self, cache_manager: KVCacheManager):
        self.cache_manager = cache_manager
        self.models = {}  # Dict[model_tag, Llama]
        self.model_locks = {}
        self.locks_lock = threading.Lock()

    def Ping(self, request, context):
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
                    self.models[model_tag] = Llama(model_path=model_path, n_ctx=ctx_size, verbose=False, seed=42)
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
                        self.cache_manager._write_log({
                            "request_id": request.request_id,
                            "component": "workstation_server",
                            "event": "RESTORE_FAILED",
                            "error": str(e)
                        })
                        context.set_code(grpc.StatusCode.INTERNAL)
                        context.set_details(f"Cache restoration failed: {e}")
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

                # 4. Generate tokens (greedy decoding)
                gen_start = time.time()
                response_tokens = []
                for _ in range(request.max_tokens):
                    logits = llama.eval_logits[-1]
                    next_token = logits.index(max(logits))
                    
                    if next_token == llama.token_eos():
                        break
                        
                    response_tokens.append(next_token)
                    llama.eval([next_token])
                    
                response_text = llama.detokenize(response_tokens).decode("utf-8", errors="ignore")
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
                        self.cache_manager._write_log({
                            "request_id": request.request_id,
                            "component": "workstation_server",
                            "event": "SAVE_FAILED",
                            "error": str(e)
                        })
                        context.set_code(grpc.StatusCode.INTERNAL)
                        context.set_details(f"Cache serialization failed: {e}")
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
            import traceback
            print("Exception in WorkstationCoordinatorServicer.DispatchTask:")
            traceback.print_exc()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return coordinator_pb2.TaskResponse()

def serve(cache_manager: KVCacheManager, port: int = 50051) -> grpc.Server:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    coordinator_pb2_grpc.add_WorkstationCoordinatorServicer_to_server(
        WorkstationCoordinatorServicer(cache_manager), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"Workstation Coordinator Server running on port {port}...")
    return server
