import subprocess
import sys
import os

def compile_proto():
    print("Compiling protocol buffers...")
    proto_file = "src/liteagent/network/protos/coordinator.proto"
    if not os.path.exists(proto_file):
        print(f"Error: {proto_file} not found.")
        sys.exit(1)
        
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-Isrc",
        "--python_out=src",
        "--grpc_python_out=src",
        proto_file
    ]
    
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("Protobuf compilation failed:")
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        sys.exit(res.returncode)
    else:
        print("Protobuf compilation succeeded!")

if __name__ == "__main__":
    compile_proto()
