"""AstrBot plugin entry point."""

from __future__ import annotations

from astrbot.api import logger
from astrbot.api.star import Context, Star
from astrbot.core.provider.sources.gemini_source import ProviderGoogleGenAI

from .fix import (
    PatchHandle,
    RepairStats,
    install_provider_patch,
    restore_provider_patch,
)


class FixGeminiToolCallPlugin(Star):
    """Install and remove the Gemini history compatibility patch."""

    def __init__(self, context: Context) -> None:
        super().__init__(context)
        self._patch_handle: PatchHandle | None = None

    async def initialize(self) -> None:
        """Install the provider patch when AstrBot loads the plugin."""

        self._patch_handle = install_provider_patch(
            ProviderGoogleGenAI,
            on_repair=self._report_repair,
        )
        if self._patch_handle.owns_patch:
            logger.info("[fix_gemini_toolcall] Gemini 工具调用历史兼容补丁已安装。")
        else:
            logger.info("[fix_gemini_toolcall] 兼容补丁已经存在，跳过重复安装。")

    async def terminate(self) -> None:
        """Restore the original provider method when the plugin unloads."""

        if restore_provider_patch(self._patch_handle):
            logger.info("[fix_gemini_toolcall] Gemini 工具调用历史兼容补丁已卸载。")
        self._patch_handle = None

    @staticmethod
    def _report_repair(stats: RepairStats) -> None:
        if stats.repaired:
            logger.debug(
                "[fix_gemini_toolcall] 本次 Gemini 请求修复了 %d 条工具响应。",
                stats.repaired,
            )
        if stats.unresolved:
            logger.warning(
                "[fix_gemini_toolcall] 有 %d 条孤立工具响应无法恢复函数名；"
                "建议对受影响会话执行 /new。",
                stats.unresolved,
            )
