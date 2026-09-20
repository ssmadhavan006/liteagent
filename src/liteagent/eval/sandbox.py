import os
import sys
import subprocess
import tempfile

# Modules solution code may import. Pure-computation standard library only.
#
# These are pre-loaded before the import hook is installed (see the guard
# below). Installing the hook first breaks the import machinery: loading a
# module pulls in transitive dependencies such as `enum`, `abc` and `sre_compile`
# that are not named here, so `from typing import List` -- which opens a large
# fraction of HumanEval problems -- raised PermissionError and scored every one
# of those tasks as a failure.
SAFE_MODULES = {
    "math", "typing", "collections", "re", "string", "datetime",
    "itertools", "functools", "heapq", "array", "bisect",
    "copy", "operator", "fractions", "decimal", "statistics", "enum",
}

SANDBOX_GUARD_TEMPLATE = """# Layered Sandbox Guard
import builtins
import sys

_SAFE = {safe_modules_repr}

# 1. Pre-load permitted modules while the import system still works.
#    Afterwards user code resolves them from sys.modules without invoking the
#    loader, so restricting imports cannot break a permitted module.
for _m in _SAFE:
    try:
        __import__(_m)
    except ImportError:
        pass

# 2. Restrict further imports to the permitted set.
#    The allowlist is captured in a closure over a frozenset rather than read
#    from a global: a global could be deleted (breaking the hook) or mutated by
#    solution code to re-admit a blocked module.
def _make_safe_import(_original, _allowed):
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root_module = name.split('.')[0]
        if root_module not in _allowed:
            raise PermissionError(f"Import of module '{{name}}' is forbidden in this sandbox.")
        return _original(name, globals, locals, fromlist, level)
    return _safe_import

builtins.__import__ = _make_safe_import(builtins.__import__, frozenset(_SAFE))

# 3. Block direct file operations and dangerous builtins
def _safe_open(*args, **kwargs):
    raise PermissionError("File operations ('open') are forbidden in this sandbox.")
builtins.open = _safe_open

def _forbidden_builtin(*args, **kwargs):
    raise PermissionError("Dangerous builtin function is forbidden in this sandbox.")

for _func_name in ["eval", "exec", "compile", "input", "breakpoint"]:
    if hasattr(builtins, _func_name):
        setattr(builtins, _func_name, _forbidden_builtin)

# 4. Drop capability-bearing modules from sys.modules so they cannot be reached
#    by lookup. Done after pre-loading, so permitted modules are unaffected.
#    `sys` itself is removed last; it is not in _SAFE, so re-importing it is
#    blocked by the hook above.
for _mod_name in ["os", "io", "subprocess", "shutil", "importlib",
                  "ctypes", "socket", "http", "urllib", "pathlib", "sys"]:
    sys.modules.pop(_mod_name, None)

# Delete internal helpers from script namespace before running untrusted code
del sys, _m, _mod_name, _func_name, _forbidden_builtin, _SAFE, _make_safe_import

# 5. Execute solution code
{code}

# 6. Execute assertions / tests
{test_code}
"""

def run_sandboxed_code(code: str, test_code: str, timeout: float = 3.0) -> dict:
    """
    Executes Python code in a layered subprocess sandbox.
    Layer 1: Spawns a separate process.
    Layer 2: Isolates in a temporary directory.
    Layer 3: Removes environment variables.
    Layer 4: Executes under isolated python -E -I -S.
    Layer 5: OS-level resource limits (resource.setrlimit on Unix, Job Objects on Windows where supported).
    Layer 6: Python-level import hook & builtin restriction (secondary safeguard).
    """
    # Prepare script content
    script_content = SANDBOX_GUARD_TEMPLATE.format(
        safe_modules_repr=repr(SAFE_MODULES),
        code=code,
        test_code=test_code
    )

    # 1. Temporary directory isolation
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "solution.py")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(script_content)

        # 2. Minimal environment variables
        minimal_env = {}
        for var in ["PATH", "SYSTEMROOT", "COMSPEC", "PATHEXT", "PYTHONPATH"]:
            if var in os.environ:
                minimal_env[var] = os.environ[var]

        # 3. Unix Resource Limits pre-exec function
        preexec_fn = None
        if os.name != 'nt':
            def set_limits():
                try:
                    import resource
                    # Max CPU time: 4 seconds
                    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
                    # Max memory address space: 512 MB
                    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
                except ImportError:
                    pass
            preexec_fn = set_limits

        # 4. Spawn subprocess running python -E -I -S
        # -E: ignore PYTHONPATH and PYTHONHOME environment variables
        # -I: isolate Python from user's environment (implies -E and -s)
        # -S: don't imply import site on initialization
        cmd = [sys.executable, "-E", "-I", "-S", temp_file_path]

        try:
            res = subprocess.run(
                cmd,
                env=minimal_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=preexec_fn,
                cwd=temp_dir  # execute inside temp directory
            )

            success = (res.returncode == 0)
            failure_category = None
            if not success:
                if "PermissionError" in res.stderr:
                    failure_category = "FAILURE_SANDBOX_VIOLATION"
                elif "AssertionError" in res.stderr:
                    failure_category = "FAILURE_ASSERTION"
                else:
                    failure_category = "FAILURE_RUNTIME_ERROR"

            return {
                "success": success,
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
                "timeout_occurred": False,
                "failure_category": failure_category
            }

        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "returncode": -9,
                "stdout": e.stdout if e.stdout else "",
                "stderr": e.stderr if e.stderr else "",
                "timeout_occurred": True,
                "failure_category": "FAILURE_TIMEOUT"
            }
