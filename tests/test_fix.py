from __future__ import annotations

from typing import Any

from fix import (
    install_provider_patch,
    repair_tool_response_names,
    restore_provider_patch,
)


def _assistant_call(call_id: str, name: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": "{}"},
            }
        ],
    }


def test_repairs_missing_tool_response_name_without_mutating_history() -> None:
    assistant = _assistant_call("call_123", "web_search_tavily")
    tool = {"role": "tool", "tool_call_id": "call_123", "content": "result"}
    messages = [assistant, tool]

    normalized, stats = repair_tool_response_names(messages)

    assert stats.repaired == 1
    assert stats.unresolved == 0
    assert normalized[1]["name"] == "web_search_tavily"
    assert "name" not in tool
    assert normalized[0] is assistant


def test_preserves_an_existing_tool_response_name() -> None:
    messages = [
        _assistant_call("call_123", "web_search_tavily"),
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "custom_name",
            "content": "result",
        },
    ]

    normalized, stats = repair_tool_response_names(messages)

    assert stats.repaired == 0
    assert stats.unresolved == 0
    assert normalized[1]["name"] == "custom_name"


def test_repairs_parallel_calls_by_id() -> None:
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_a",
                "function": {"name": "web_search_tavily", "arguments": "{}"},
            },
            {
                "id": "call_b",
                "function": {"name": "search_emoji", "arguments": "{}"},
            },
        ],
    }
    messages = [
        assistant,
        {"role": "tool", "tool_call_id": "call_b", "content": "emoji"},
        {"role": "tool", "tool_call_id": "call_a", "content": "search"},
    ]

    normalized, stats = repair_tool_response_names(messages)

    assert stats.repaired == 2
    assert normalized[1]["name"] == "search_emoji"
    assert normalized[2]["name"] == "web_search_tavily"


def test_reports_orphan_tool_response() -> None:
    tool = {"role": "tool", "tool_call_id": "missing", "content": "result"}

    normalized, stats = repair_tool_response_names([tool])

    assert normalized == [tool]
    assert stats.repaired == 0
    assert stats.unresolved == 1


def test_patch_repairs_a_copy_of_payload_before_calling_original() -> None:
    seen: list[dict[str, Any]] = []

    class FakeProvider:
        def _prepare_conversation(self, payloads: dict[str, Any]):
            seen.append(payloads)
            return payloads["messages"]

    original = FakeProvider._prepare_conversation
    reports = []
    handle = install_provider_patch(FakeProvider, on_repair=reports.append)
    payloads = {
        "messages": [
            _assistant_call("call_123", "web_search_tavily"),
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
        ]
    }

    try:
        result = FakeProvider()._prepare_conversation(payloads)
        second_handle = install_provider_patch(FakeProvider)

        assert result[1]["name"] == "web_search_tavily"
        assert "name" not in payloads["messages"][1]
        assert seen[0] is not payloads
        assert reports[0].repaired == 1
        assert second_handle.owns_patch is False
    finally:
        assert restore_provider_patch(handle) is True

    assert FakeProvider._prepare_conversation is original


def test_restore_does_not_overwrite_a_newer_patch() -> None:
    class FakeProvider:
        def _prepare_conversation(self, payloads: dict[str, Any]):
            return payloads

    handle = install_provider_patch(FakeProvider)

    def newer_patch(self, payloads):
        return payloads

    FakeProvider._prepare_conversation = newer_patch

    assert restore_provider_patch(handle) is False
    assert FakeProvider._prepare_conversation is newer_patch
