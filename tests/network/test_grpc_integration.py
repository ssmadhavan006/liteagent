import os
import time
import shutil
import pytest
import json
import threading
import grpc
from concurrent import futures

from liteagent.network.server import WorkstationCoordinatorServicer
from liteagent.network.protos import coordinator_pb2_grpc, coordinator_pb2
from liteagent.network.client import WorkstationClient
from liteagent.network.dispatch import TaskDispatcher
from liteagent.cache import KVCacheManager

SSD_TEST_DIR = "tests/network/temp_ssd"
LOG_TEST_DIR = "tests/network/temp_logs"

COMPLEX_LARGE_PROMPT = (
    "def calculate_multistep_logic(x, y, z):\n"
    "    # Logic and arithmetic check\n"
    "    if x > 0 and y < z or z == 0:\n"
    "        return (x + y) * z - (y / x) + math.sqrt(x)\n"
    "    else:\n"
    "        return x * y * z\n"
) * 20

class MockLlamaState:
    def __init__(self, data: bytes):
        self.llama_state = data
        self.llama_state_size = len(data)
        self.input_ids = []
        self.scores = []
        self.n_tokens = 0
        self.seed = 42

class MockLlama:
    def __init__(self, state_bytes: bytes = b"mock_weight_context"):
        self.state_bytes = state_bytes
        self.loaded_state = None

    def save_state(self) -> MockLlamaState:
        return MockLlamaState(self.state_bytes)

    def load_state(self, state):
        self.loaded_state = state
        
    def reset(self):
        pass
        
    def tokenize(self, prompt: bytes) -> list[int]:
        return [1, 2, 3]
        
    def eval(self, tokens: list[int]):
        pass
        
    def eval_logits(self):
        pass
        
    @property
    def eval_logits(self):
        return [[0.1, 0.9, 0.0]]
        
    def token_eos(self) -> int:
        return 2
        
    def detokenize(self, tokens: list[int]) -> bytes:
        return b"mock_workstation_response"

@pytest.fixture(autouse=True)
def setup_and_teardown():
    for d in [SSD_TEST_DIR, LOG_TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
    yield
    for d in [SSD_TEST_DIR, LOG_TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

class TestServer:
    def __init__(self, servicer):
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        coordinator_pb2_grpc.add_WorkstationCoordinatorServicer_to_server(servicer, self.server)
        self.server.add_insecure_port("[::]:50055")
        
    def start(self):
        self.server.start()
        
    def stop(self):
        self.server.stop(0)

def test_grpc_ping_and_dispatch():
    # Setup server and client
    workstation_cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    servicer = WorkstationCoordinatorServicer(workstation_cm)
    
    # Inject MockLlama to bypass GGUF resolution during loopback unit testing
    servicer.models["llama3.1:8b"] = MockLlama()
    
    srv = TestServer(servicer)
    srv.start()
    
    try:
        client = WorkstationClient("localhost", 50055)
        
        # Test Ping
        ping_res = client.ping()
        assert ping_res["compatible"] is True
        assert ping_res["protocol_version"] == 1
        assert "liteagent_version" in ping_res
        
        # Test dispatch
        res = client.dispatch_task(
            request_id="req-123",
            task_id="task-456",
            session_id="session-789",
            agent_role="Planner",
            prompt="Hello workstation",
            system_prompt="Be helpful"
        )
        
        assert res["response_text"] == "mock_workstation_response"
        assert res["request_id"] == "req-123"
        assert res["communication_overhead_ms"] is not None
        assert res["client_serialize_ms"] >= 0.0
        assert res["server_compute_ms"] >= 0.0
        
    finally:
        srv.stop()

def test_dispatcher_fallback_medium_local():
    # Force a fallback by pointing client to invalid port 50059 (unreachable)
    edge_cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    client = WorkstationClient("localhost", 50059)
    
    dispatcher = TaskDispatcher(
        router_config_path="config/router_config.yaml",
        edge_cache_manager=edge_cm,
        workstation_client=client,
        log_dir=LOG_TEST_DIR
    )
    # Configure fallback policy override
    dispatcher.fallback_policy = "medium_local"
    dispatcher.max_retries = 1
    
    # Inject MockLlama on local dispatcher model caches
    dispatcher.local_models["llama3.2:3b"] = MockLlama(b"mock_local_medium_state")
    
    task = {"prompt": COMPLEX_LARGE_PROMPT, "benchmark": "GSM8K"}
    
    # Run task: remote fails -> fall back to local Medium
    res = dispatcher.execute_task(
        task=task,
        session_id="session-fallback",
        system_prompt="Be local fallback"
    )
    
    assert res["executed_tier"] == "Medium"
    assert res["routed_tier"] == "Large"
    assert res["fallback_occurred"] is True
    
    # Check log for FALLBACK event
    log_file = os.path.join(LOG_TEST_DIR, "operations.jsonl")
    assert os.path.exists(log_file)
    
    events = []
    with open(log_file, "r") as f:
        for line in f:
            events.append(json.loads(line))
            
    fallback_events = [e for e in events if e["event"] == "FALLBACK"]
    assert len(fallback_events) == 1
    assert fallback_events[0]["policy"] == "medium_local"
    assert fallback_events[0]["executed_on"] == "edge_medium"
    
    completed_events = [e for e in events if e["event"] == "COMPLETED"]
    assert len(completed_events) == 1
    assert completed_events[0]["outcome"] == "FALLBACK_SUCCESS"

def test_dispatcher_fallback_fail():
    edge_cm = KVCacheManager(max_ram_states=2, ssd_dir=SSD_TEST_DIR, log_dir=LOG_TEST_DIR)
    client = WorkstationClient("localhost", 50059)
    
    dispatcher = TaskDispatcher(
        router_config_path="config/router_config.yaml",
        edge_cache_manager=edge_cm,
        workstation_client=client,
        log_dir=LOG_TEST_DIR
    )
    dispatcher.fallback_policy = "fail"
    dispatcher.max_retries = 0
    
    task = {"prompt": COMPLEX_LARGE_PROMPT, "benchmark": "GSM8K"}
    
    with pytest.raises(ConnectionError):
        dispatcher.execute_task(
            task=task,
            session_id="session-fail",
            system_prompt="Be local fail"
        )
