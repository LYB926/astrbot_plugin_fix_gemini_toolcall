"""Compatibility patch for Gemini function-call history conversion."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any

PATCH_MARKER = "_astrbot_fix_gemini_toolcall"
ORIGINAL_ATTR = "_astrbot_fix_gemini_toolcall_original"


@dataclass(frozen=True, slots=True)
class RepairStats:
    """Summary of a history normalization pass."""

    repaired: int = 0
    unresolved: int = 0


@dataclass(frozen=True, slots=True)
class PatchHandle:
    """State needed to safely restore an installed class-level patch."""

    provider_class: type
    original: Callable[..., Any]
    patched: Callable[..., Any]
    owns_patch: bool


def repair_tool_response_names(
    messages: Sequence[Any],
) -> tuple[list[Any], RepairStats]:
    """Fill missing tool-response names from preceding assistant tool calls.

    AstrBot stores OpenAI-style tool history. A tool response has a
    ``tool_call_id``, while its function name lives on the preceding assistant
    message. Gemini requires ``functionResponse.name`` to equal the matching
    ``functionCall.name``. This function joins those two pieces without
    mutating the stored conversation history.
    """

    tool_name_by_call_id: dict[str, str] = {}
    normalized: list[Any] = []
    repaired = 0
    unresolved = 0

    for raw_message in messages:
        if not isinstance(raw_message, Mapping):
            normalized.append(raw_message)
            continue

        role = raw_message.get("role")

        if role == "assistant":
            tool_calls = raw_message.get("tool_calls") or []
            if isinstance(tool_calls, Sequence) and not isinstance(
                tool_calls, (str, bytes)
            ):
                for tool_call in tool_calls:
                    if not isinstance(tool_call, Mapping):
                        continue
                    function = tool_call.get("function")
                    if not isinstance(function, Mapping):
                        continue
                    tool_call_id = tool_call.get("id")
                    function_name = function.get("name")
                    if tool_call_id is not None and function_name:
                        tool_name_by_call_id[str(tool_call_id)] = str(function_name)

            normalized.append(raw_message)
            continue

        if role != "tool" or raw_message.get("name"):
            normalized.append(raw_message)
            continue

        tool_call_id = raw_message.get("tool_call_id")
        function_name = (
            tool_name_by_call_id.get(str(tool_call_id))
            if tool_call_id is not None
            else None
        )
        if not function_name:
            unresolved += 1
            normalized.append(raw_message)
            continue

        repaired_message = dict(raw_message)
        repaired_message["name"] = function_name
        normalized.append(repaired_message)
        repaired += 1

    return normalized, RepairStats(repaired=repaired, unresolved=unresolved)


def install_provider_patch(
    provider_class: type,
    *,
    on_repair: Callable[[RepairStats], None] | None = None,
) -> PatchHandle:
    """Patch a Google GenAI provider class in an idempotent, reversible way."""

    current = getattr(provider_class, "_prepare_conversation", None)
    if not callable(current):
        raise RuntimeError(
            "The installed AstrBot Google GenAI provider does not expose "
            "_prepare_conversation()."
        )

    if getattr(current, PATCH_MARKER, False):
        original = getattr(current, ORIGINAL_ATTR, current)
        return PatchHandle(
            provider_class=provider_class,
            original=original,
            patched=current,
            owns_patch=False,
        )

    original = current

    @wraps(original)
    def patched_prepare_conversation(provider_self: Any, payloads: dict[str, Any]):
        messages = payloads.get("messages")
        if not isinstance(messages, list):
            return original(provider_self, payloads)

        normalized, stats = repair_tool_response_names(messages)
        if on_repair is not None and (stats.repaired or stats.unresolved):
            on_repair(stats)

        if not stats.repaired:
            return original(provider_self, payloads)

        patched_payloads = dict(payloads)
        patched_payloads["messages"] = normalized
        return original(provider_self, patched_payloads)

    setattr(patched_prepare_conversation, PATCH_MARKER, True)
    setattr(patched_prepare_conversation, ORIGINAL_ATTR, original)
    provider_class._prepare_conversation = patched_prepare_conversation

    return PatchHandle(
        provider_class=provider_class,
        original=original,
        patched=patched_prepare_conversation,
        owns_patch=True,
    )


def restore_provider_patch(handle: PatchHandle | None) -> bool:
    """Restore a patch owned by ``handle`` without removing another patch."""

    if handle is None or not handle.owns_patch:
        return False
    current = getattr(handle.provider_class, "_prepare_conversation", None)
    if current is not handle.patched:
        return False
    handle.provider_class._prepare_conversation = handle.original
    return True
