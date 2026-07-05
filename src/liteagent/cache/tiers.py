import time
import datetime
import llama_cpp

class CacheStateMetadata:
    def __init__(
        self,
        session_key: str,
        agent_role: str,
        model_tag: str,
        ctx_size: int,
        prompt_hash: str,
        state_size_bytes: int,
        cache_format_version: int = 1
    ):
        self.session_key = session_key
        self.agent_role = agent_role
        self.model_tag = model_tag
        self.ctx_size = ctx_size
        self.prompt_hash = prompt_hash
        self.state_size_bytes = state_size_bytes
        self.cache_format_version = cache_format_version
        
        # Get llama-cpp version programmatically
        try:
            self.llama_cpp_version = llama_cpp.__version__
        except AttributeError:
            self.llama_cpp_version = "0.3.1"
            
        self.last_accessed = time.time()
        self.created_at = datetime.datetime.now(datetime.UTC).isoformat() + "Z"

    def to_dict(self) -> dict:
        return {
            "cache_format_version": self.cache_format_version,
            "llama_cpp_version": self.llama_cpp_version,
            "model_tag": self.model_tag,
            "ctx_size": self.ctx_size,
            "agent_role": self.agent_role,
            "created_at": self.created_at,
            "prompt_hash": self.prompt_hash,
            "state_size_bytes": self.state_size_bytes,
            "last_accessed": self.last_accessed
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CacheStateMetadata':
        meta = cls(
            session_key=data.get("session_key", ""),
            agent_role=data.get("agent_role", ""),
            model_tag=data.get("model_tag", ""),
            ctx_size=data.get("ctx_size", 0),
            prompt_hash=data.get("prompt_hash", ""),
            state_size_bytes=data.get("state_size_bytes", 0),
            cache_format_version=data.get("cache_format_version", 1)
        )
        meta.llama_cpp_version = data.get("llama_cpp_version", meta.llama_cpp_version)
        meta.created_at = data.get("created_at", meta.created_at)
        meta.last_accessed = data.get("last_accessed", meta.last_accessed)
        return meta
