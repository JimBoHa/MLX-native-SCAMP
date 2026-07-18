"""Run a standalone Apple/MLX distributed worker."""

from __future__ import annotations

import argparse

from .runtime import WorkerServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an MLX-native SCAMP gRPC worker")
    parser.add_argument(
        "--host", default="127.0.0.1", help="bind address (default: loopback)"
    )
    parser.add_argument("--port", type=int, default=30078, help="bind port")
    parser.add_argument("--backend", choices=("auto", "metal", "cpu"), default="auto")
    parser.add_argument("--rpc-threads", type=int, default=4)
    parser.add_argument("--max-concurrent-tiles", type=int, default=1)
    parser.add_argument(
        "--max-tile-working-set-bytes",
        type=int,
        default=None,
        help="reject dense tiles estimated to exceed this working set",
    )
    parser.add_argument(
        "--allow-insecure-remote",
        action="store_true",
        help="allow an unauthenticated non-loopback bind (unsafe on untrusted networks)",
    )
    args = parser.parse_args()

    server = WorkerServer(
        host=args.host,
        port=args.port,
        backend=args.backend,
        rpc_threads=args.rpc_threads,
        max_concurrent_tiles=args.max_concurrent_tiles,
        max_tile_working_set_bytes=args.max_tile_working_set_bytes,
        allow_insecure_remote=args.allow_insecure_remote,
    ).start()
    print(f"MLX SCAMP worker listening on {server.target}", flush=True)
    try:
        server.wait()
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
