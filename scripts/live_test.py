"""Optional live test against a Google GenAI-compatible endpoint.

Environment variables:
    GEMINI_API_KEY          required
    GEMINI_API_BASE         defaults to https://generativelanguage.googleapis.com
    GEMINI_MODEL            defaults to gemini-3.5-flash
"""

from __future__ import annotations

import json
import os
import sys

from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI
from google import genai
from google.genai import types

from fix import install_provider_patch, restore_provider_patch


def main() -> int:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is required.", file=sys.stderr)
        return 2

    api_base = os.environ.get(
        "GEMINI_API_BASE", "https://generativelanguage.googleapis.com"
    ).rstrip("/")
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(base_url=api_base, timeout=60_000),
    )
    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="echo",
                description="Echo the provided text.",
                parameters={
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            )
        ]
    )
    forced = types.GenerateContentConfig(
        tools=[tool],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode="ANY", allowed_function_names=["echo"]
            )
        ),
    )

    first = client.models.generate_content(
        model=model,
        contents="Call echo with text 'hi'.",
        config=forced,
    )
    returned_part = next(
        part for part in (first.candidates[0].content.parts or []) if part.function_call
    )
    function_call = returned_part.function_call
    if not function_call or not function_call.name or not function_call.id:
        print("Endpoint did not return a function call with a non-empty ID.")
        return 1

    extra_content = None
    if returned_part.thought_signature:
        import base64

        extra_content = {
            "google": {
                "thought_signature": base64.b64encode(
                    returned_part.thought_signature
                ).decode("ascii")
            }
        }

    tool_call_record = {
        "id": function_call.id,
        "type": "function",
        "function": {
            "name": function_call.name,
            "arguments": json.dumps(function_call.args or {}),
        },
    }
    if extra_content:
        tool_call_record["extra_content"] = extra_content

    payloads = {
        "messages": [
            {"role": "user", "content": "Call echo with text 'hi'."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call_record],
            },
            {
                "role": "tool",
                "tool_call_id": function_call.id,
                "content": "hi",
            },
        ]
    }

    handle = install_provider_patch(ProviderGoogleGenAI)
    try:
        provider = object.__new__(ProviderGoogleGenAI)
        contents = provider._prepare_conversation(payloads)
        result = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(tools=[tool]),
        )
    finally:
        restore_provider_patch(handle)
        client.close()

    response_name = contents[-1].parts[-1].function_response.name
    follow_up = (result.text or "")[:80]
    print(
        f"PASS: endpoint returned ID={function_call.id!r}; "
        f"patched functionResponse.name={response_name!r}; "
        f"follow-up={follow_up!r}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
