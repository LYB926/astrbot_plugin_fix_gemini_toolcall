from __future__ import annotations

import asyncio

from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI

from fix import install_provider_patch, restore_provider_patch


def test_real_astrbot_converter_emits_matching_function_names() -> None:
    handle = install_provider_patch(ProviderGoogleGenAI)
    provider = object.__new__(ProviderGoogleGenAI)
    payloads = {
        "messages": [
            {"role": "user", "content": "Search for AstrBot"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "web_search_tavily",
                            "arguments": '{"query":"AstrBot"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "AstrBot result",
            },
        ]
    }

    try:
        contents = provider._prepare_conversation(payloads)
    finally:
        restore_provider_patch(handle)

    function_call = contents[1].parts[0].function_call
    function_response = contents[2].parts[0].function_response

    assert function_call is not None
    assert function_response is not None
    assert function_call.name == "web_search_tavily"
    assert function_response.name == function_call.name


def test_plugin_lifecycle_installs_and_restores_patch() -> None:
    from astrbot_plugin_fix_gemini_toolcall.main import FixGeminiToolCallPlugin

    original = ProviderGoogleGenAI._prepare_conversation
    plugin = FixGeminiToolCallPlugin(context=object())

    asyncio.run(plugin.initialize())
    try:
        assert ProviderGoogleGenAI._prepare_conversation is not original
    finally:
        asyncio.run(plugin.terminate())

    assert ProviderGoogleGenAI._prepare_conversation is original
