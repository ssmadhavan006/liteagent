import pytest
import grpc
from concurrent import futures
from liteagent.cache import KVCacheManager
from liteagent.network.server import WorkstationCoordinatorServicer
from liteagent.network.client import WorkstationClient
from liteagent.network.protos import coordinator_pb2_grpc

class MockLlama:
    def tokenize(self, text):
        return [1, 2, 3]
    def eval(self, tokens):
        pass
    def reset(self):
        pass
    def token_eos(self):
        return 999
    def detokenize(self, tokens):
        return b"mock_response"

@pytest.fixture
def auth_server(tmp_path):
    cm = KVCacheManager(max_ram_states=2, ssd_dir=str(tmp_path / "ssd"), log_dir=str(tmp_path / "logs"))
    servicer = WorkstationCoordinatorServicer(cm, auth_token="secret-token-123")
    servicer.models["llama3.1:8b"] = MockLlama()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    coordinator_pb2_grpc.add_WorkstationCoordinatorServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    yield port, "secret-token-123"
    server.stop(0)

def test_grpc_auth_enforcement(auth_server):
    port, token = auth_server

    # 1. Client with missing/invalid token -> fails
    bad_client = WorkstationClient("127.0.0.1", port, auth_token="wrong-token")
    with pytest.raises(ConnectionError):
        bad_client.ping()

    # 2. Client with valid token -> succeeds
    good_client = WorkstationClient("127.0.0.1", port, auth_token=token)
    res = good_client.ping()
    assert res["compatible"] is True

def test_grpc_input_validation(auth_server):
    port, token = auth_server
    client = WorkstationClient("127.0.0.1", port, auth_token=token)

    # Oversized prompt validation
    huge_prompt = "A" * 40000
    with pytest.raises(ConnectionError):
        client.dispatch_task(
            request_id="r1",
            task_id="t1",
            session_id="s1",
            agent_role="Planner",
            prompt=huge_prompt,
            system_prompt="system"
        )
