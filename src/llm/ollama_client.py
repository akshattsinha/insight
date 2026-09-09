import json
import os
import urllib.error
import urllib.request
from typing import Any


class OllamaClient:
    """
    Lightweight client for communicating with a
    locally running Ollama server.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int = 120,
    ):
        self.host = (
            host
            or os.getenv(
                "OLLAMA_HOST",
                "http://host.docker.internal:11434",
            )
        ).rstrip("/")

        self.model = (
            model
            or os.getenv(
                "OLLAMA_MODEL",
                "qwen3.5:4b",
            )
        )

        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Generate a response from the configured
        Ollama model.
        """

        payload: dict[str, Any] = {
    "model": self.model,
    "prompt": prompt,
    "stream": False,
    "think": False,
}

        if system_prompt:
            payload["system"] = system_prompt

        request_data = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            url=f"{self.host}/api/generate",
            data=request_data,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                response_body = (
                    response.read()
                    .decode("utf-8")
                )

        except urllib.error.URLError as error:

            raise RuntimeError(
                "Could not connect to Ollama at "
                f"{self.host}. "
                "Make sure Ollama is running on the host machine."
            ) from error

        except TimeoutError as error:

            raise RuntimeError(
                "Ollama request timed out after "
                f"{self.timeout} seconds."
            ) from error

        try:
            result = json.loads(
                response_body
            )

        except json.JSONDecodeError as error:

            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from error

        response_text = result.get(
            "response"
        )

        if not response_text:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        return str(response_text).strip()

    def health_check(self) -> bool:
        """
        Check whether the Ollama server is reachable.
        """

        request = urllib.request.Request(
            url=f"{self.host}/api/tags",
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=10,
            ) as response:

                return (
                    response.status == 200
                )

        except (
            urllib.error.URLError,
            TimeoutError,
        ):

            return False

    def get_config(self) -> dict[str, str]:
        """
        Return the active Ollama configuration.
        """

        return {
            "host": self.host,
            "model": self.model,
        }


def main():
    """
    Test the Ollama connection and generate
    a simple response.
    """

    client = OllamaClient()

    config = client.get_config()

    print()
    print("=" * 80)
    print("OLLAMA CLIENT")
    print("=" * 80)

    print(
        f"Host:  {config['host']}"
    )

    print(
        f"Model: {config['model']}"
    )

    print()

    print(
        "Checking Ollama connection..."
    )

    if not client.health_check():

        raise RuntimeError(
            "Ollama is not reachable."
        )

    print(
        "Ollama connection: PASS"
    )

    print()
    print(
        "Testing model generation..."
    )

    response = client.generate(
        prompt=(
            "Explain in one short sentence "
            "what business analytics is."
        )
    )

    print()
    print("MODEL RESPONSE")
    print("-" * 80)
    print(response)

    print()
    print("=" * 80)
    print(
        "STATUS: PASS"
    )
    print(
        "Ollama integration is working."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()