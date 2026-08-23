"""Entry point for the 0G realtime backend server."""

import asyncio
import logging
import argparse

from ante.zerog.realtime_server import ZeroGRealtimeServer


logger = logging.getLogger(__name__)


def main() -> None:
    """Run the 0G realtime server until interrupted."""
    parser = argparse.ArgumentParser(description="0G Compute Network realtime backend for Ante.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument(
        "--push-to-talk",
        action="store_true",
        help="Only listen while the antenna gate is open, instead of always-on voice detection.",
    )
    parser.add_argument("--language", default="en", help="Transcription language code (default: en)")
    parser.add_argument(
        "--daemon-url",
        default="http://localhost:8000",
        help="Reachy Mini daemon URL, used to read the antenna gate (Wireless: http://reachy-mini.local:8000)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    server = ZeroGRealtimeServer(
        host=args.host,
        port=args.port,
        gated=args.push_to_talk,
        language=args.language,
        daemon_url=args.daemon_url,
    )
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        logger.info("0G realtime server stopped")


if __name__ == "__main__":
    main()
