"""Supervises the local `0g-compute-cli inference serve` proxies.

The CLI proxy owns the wallet key, signs every request and settles payment with the
provider, so nothing in this app ever handles the private key or the single-use
request headers that trip up hand-rolled 0G clients.
"""

import time
import shutil
import logging
import subprocess

import httpx

from ante.zerog.providers import ZeroGProvider


logger = logging.getLogger(__name__)

CLI_NAME = "0g-compute-cli"
_STARTUP_TIMEOUT_S = 90.0
_POLL_INTERVAL_S = 1.0


class ZeroGGatewayError(RuntimeError):
    """Raised when a 0G proxy cannot be started or never becomes reachable."""


class ZeroGGateway:
    """Runs one local OpenAI-compatible proxy per 0G provider and tears them down on exit."""

    def __init__(self, providers: list[ZeroGProvider], host: str = "127.0.0.1") -> None:
        """Prepare a proxy for each provider, bound to ``host``."""
        self.host = host
        self.providers = providers
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

    def base_url(self, provider: ZeroGProvider) -> str:
        """Return the local OpenAI-compatible base URL for a provider."""
        return f"http://{self.host}:{provider.port}/v1"

    def start(self) -> None:
        """Start every proxy and block until each answers, so the first turn is never a cold start."""
        if shutil.which(CLI_NAME) is None:
            raise ZeroGGatewayError(
                f"{CLI_NAME} not found on PATH. Install it with `npm i -g @0glabs/0g-serving-broker` "
                f"and log in with `{CLI_NAME} login`."
            )

        for provider in self.providers:
            if self._is_reachable(provider):
                logger.info("Reusing existing 0G proxy for %s on port %d", provider.model, provider.port)
                continue
            self._spawn(provider)

        for provider in self.providers:
            self._await_ready(provider)

    def _spawn(self, provider: ZeroGProvider) -> None:
        logger.info(
            "Starting 0G proxy for %s (provider %s) on port %d", provider.model, provider.address, provider.port
        )
        self._processes[provider.address] = subprocess.Popen(
            [
                CLI_NAME,
                "inference",
                "serve",
                "--provider",
                provider.address,
                "--port",
                str(provider.port),
                "--host",
                self.host,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _is_reachable(self, provider: ZeroGProvider) -> bool:
        """Return whether the proxy port answers at all; any HTTP status counts as up."""
        try:
            httpx.post(f"{self.base_url(provider)}/chat/completions", json={}, timeout=2.0)
        except httpx.HTTPError:
            return False
        return True

    def _await_ready(self, provider: ZeroGProvider) -> None:
        deadline = time.monotonic() + _STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            process = self._processes.get(provider.address)
            if process is not None and process.poll() is not None:
                raise ZeroGGatewayError(
                    f"0G proxy for {provider.model} exited with code {process.returncode}. "
                    f"Check the provider sub-account is funded: `{CLI_NAME} get-account`."
                )
            if self._is_reachable(provider):
                logger.info("0G proxy for %s ready on port %d", provider.model, provider.port)
                return
            time.sleep(_POLL_INTERVAL_S)

        raise ZeroGGatewayError(
            f"0G proxy for {provider.model} did not become reachable within {_STARTUP_TIMEOUT_S:.0f}s."
        )

    def stop(self) -> None:
        """Terminate every proxy this gateway started."""
        for address, process in self._processes.items():
            if process.poll() is not None:
                continue
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("0G proxy %s ignored terminate; killing", address)
                process.kill()
        self._processes.clear()
