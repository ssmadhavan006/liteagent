"""
Guards the HOT tier against serving a stale context.

`load_cache` returns "HOT" without deserialising anything: it assumes the model
still holds the context that was saved. Generation breaks that assumption, since
the context has advanced by the generated tokens. Before this was fixed, prefix
caching evaluated every task on top of the previous task's leftovers while
reporting a cache hit, and Exp 2's losslessness check was what exposed it.
"""

import os
import tempfile

import pytest

from liteagent.cache import KVCacheManager


class FakeState:
    def __init__(self, data: bytes):
        self.llama_state = data
        self.llama_state_size = len(data)


class FakeLlama:
    """Records restores so a real deserialisation can be distinguished from none."""

    def __init__(self):
        self.loads = 0
        self.context = "saved"

    def save_state(self):
        return FakeState(b"state-bytes")

    def load_state(self, state):
        self.loads += 1
        self.context = "saved"


@pytest.fixture
def manager():
    with tempfile.TemporaryDirectory() as tmp:
        yield KVCacheManager(max_ram_states=4,
                             ssd_dir=os.path.join(tmp, "ssd"),
                             log_dir=None)


def test_clean_context_is_served_hot_without_restoring(manager):
    llama = FakeLlama()
    manager.save_cache("k", "Executor", llama, "m", 512, "h")

    tier = manager.load_cache("k", llama, "m", 512, "h")
    assert tier == "HOT"
    assert llama.loads == 0, "HOT must not pay for a deserialisation"


def test_dirty_context_is_not_served_hot(manager):
    """The defect: a mutated context reported as HOT and silently reused."""
    llama = FakeLlama()
    manager.save_cache("k", "Executor", llama, "m", 512, "h")

    manager.mark_context_dirty()
    llama.context = "advanced by generation"

    tier = manager.load_cache("k", llama, "m", 512, "h")
    assert tier != "HOT", "a mutated context must not be served as HOT"
    assert tier == "STANDBY"
    assert llama.loads == 1, "falling through must perform a real restore"
    assert llama.context == "saved", "restore must undo the mutation"


def test_restore_clears_the_dirty_flag(manager):
    llama = FakeLlama()
    manager.save_cache("k", "Executor", llama, "m", 512, "h")
    manager.mark_context_dirty()

    assert manager.load_cache("k", llama, "m", 512, "h") == "STANDBY"
    # Now resident and untouched again, so the cheap path is valid.
    assert manager.load_cache("k", llama, "m", 512, "h") == "HOT"


def test_saving_marks_the_context_clean(manager):
    llama = FakeLlama()
    manager.mark_context_dirty()
    manager.save_cache("k", "Executor", llama, "m", 512, "h")

    assert manager.load_cache("k", llama, "m", 512, "h") == "HOT"


def test_dirty_flag_does_not_leak_across_keys(manager):
    llama = FakeLlama()
    manager.save_cache("a", "Executor", llama, "m", 512, "ha")
    manager.save_cache("b", "Executor", llama, "m", 512, "hb")

    # "b" was saved last, so it is the resident context.
    assert manager.load_cache("b", llama, "m", 512, "hb") == "HOT"
    # "a" is resident in RAM but not active; it must be restored, not assumed.
    assert manager.load_cache("a", llama, "m", 512, "ha") == "STANDBY"
