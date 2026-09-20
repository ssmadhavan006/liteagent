import argparse
import logging
from liteagent.cache import KVCacheManager
from liteagent.network.server import serve
from liteagent.router.router import route_task

def main():
    parser = argparse.ArgumentParser(description="LiteAgent Workstation & Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to execute")

    # Serve sub-command
    serve_parser = subparsers.add_parser("serve", help="Start the Workstation Coordinator gRPC Server")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind")
    serve_parser.add_argument("--port", type=int, default=50051, help="Port to bind")
    serve_parser.add_argument("--ssd-dir", type=str, default="temp_vllm_ssd", help="SSD cache directory")
    serve_parser.add_argument("--log-dir", type=str, default="experiments", help="Log directory")
    serve_parser.add_argument("--auth-token", type=str, default=None, help="Bearer auth token")

    # Route sub-command
    route_parser = subparsers.add_parser("route", help="Route a sample prompt")
    route_parser.add_argument("--prompt", type=str, required=True, help="Prompt text to route")
    route_parser.add_argument("--config", type=str, default="config/router_config.yaml", help="Router config YAML")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.command == "serve":
        cm = KVCacheManager(max_ram_states=5, ssd_dir=args.ssd_dir, log_dir=args.log_dir)
        server = serve(cm, host=args.host, port=args.port, auth_token=args.auth_token)
        print(f"Server started on {args.host}:{args.port}. Press Ctrl+C to stop.")
        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            print("Stopping server...")
            server.stop(0)
    elif args.command == "route":
        res = route_task({"prompt": args.prompt}, config_path=args.config)
        import json
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
