"""DeepEval LLM wrapper for Claude Sonnet 4.6 on AWS Bedrock."""

from __future__ import annotations

import asyncio
import json
import time

import boto3
import botocore.config
import botocore.exceptions
from deepeval.models import DeepEvalBaseLLM


BEDROCK_MAX_RETRIES = 5
BEDROCK_RETRY_DELAY = 10
BEDROCK_EXPIRED_TOKEN_DELAY = 60


class BedrockJudge(DeepEvalBaseLLM):
    """DeepEval LLM backed by Claude Sonnet 4.6 on Bedrock Converse API.

    Mirrors Strands' BedrockModel credential strategy: each instance owns a
    fresh boto3.Session() so credentials are resolved independently of the
    module-level default session. On ExpiredToken the session is discarded and
    a new one is created so the next attempt picks up the renewed credentials.
    """

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-6",
        max_tokens: int = 4096,
    ):
        self.model_id = model_id
        self._max_tokens = max_tokens
        self._session = boto3.Session()  # per-instance session, like Strands BedrockModel
        self._client = None

    def _new_client(self):
        self._client = self._session.client(
            "bedrock-runtime",
            config=botocore.config.Config(read_timeout=120, connect_timeout=10),
        )
        return self._client

    @property
    def client(self):
        if self._client is None:
            self._new_client()
        return self._client

    def load_model(self):
        return self.model_id

    def generate(self, prompt: str, schema=None) -> str:
        if schema:
            schema_str = (
                schema.model_json_schema()
                if hasattr(schema, "model_json_schema")
                else str(schema)
            )
            prompt = (
                prompt
                + "\n\nRespond with valid JSON matching this schema:\n"
                + json.dumps(schema_str)
            )

        last_err: Exception | None = None
        for attempt in range(BEDROCK_MAX_RETRIES):
            try:
                response = self.client.converse(
                    modelId=self.model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    # temperature=0 for the judge: minimise scoring non-determinism
                    # so run-to-run variance reflects the agent, not the judge.
                    inferenceConfig={"maxTokens": self._max_tokens, "temperature": 0},
                )
                return response["output"]["message"]["content"][0]["text"]
            except (
                botocore.exceptions.ClientError,
                botocore.exceptions.NoCredentialsError,
                botocore.exceptions.ReadTimeoutError,
            ) as e:
                last_err = e
                if attempt < BEDROCK_MAX_RETRIES - 1:
                    is_expired = "ExpiredToken" in str(e) or "NoCredentials" in str(e)
                    delay = BEDROCK_EXPIRED_TOKEN_DELAY if is_expired else BEDROCK_RETRY_DELAY
                    time.sleep(delay)
                    if is_expired:
                        # Fresh session picks up renewed credentials from source,
                        # same reason Strands creates a new Session per BedrockModel.
                        self._session = boto3.Session()
                    self._new_client()
                    continue
                raise
        if last_err is not None:
            raise last_err
        return ""

    async def a_generate(self, prompt: str, schema=None) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, prompt, schema)

    def get_model_name(self) -> str:
        return f"Bedrock {self.model_id}"
