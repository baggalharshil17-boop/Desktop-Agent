"""Manual entry point for running the Section 17 eval set against the real
LLM configured via llm.provider in config.yaml (Groq or Hugging Face). NOT
automated / not gated by pytest -- this costs real API usage and needs the
matching provider key configured (GROQ_API_KEY or HF_TOKEN). Run with:

    python -m financial_voice_agent.eval

Set mode: "mock" in config.yaml before running this so Kite-backed tools
(and now get_news) use Phase 1's fixtures rather than requiring live
credentials -- the real LLM's behavior is what's under test here, not live
Kite/Tavily connectivity.
"""

from __future__ import annotations

import asyncio

from financial_voice_agent.config import load_config
from financial_voice_agent.eval.cases import load_eval_cases
from financial_voice_agent.eval.report import format_report
from financial_voice_agent.eval.runner import run_eval_set
from financial_voice_agent.http_clients import close_http_clients, create_http_clients
from financial_voice_agent.orchestrator.llm import make_llm_client
from financial_voice_agent.orchestrator.system_prompt import SYSTEM_PROMPT
from financial_voice_agent.tools.registry import TOOLS_SCHEMA, make_tool_executor


async def main() -> None:
    config = load_config()
    http_clients = await create_http_clients(config)
    try:
        llm_client = make_llm_client(config)
        tool_executor = make_tool_executor(config, http_clients)
        cases = load_eval_cases("eval/cases.json")
        results = await run_eval_set(
            cases,
            llm_client=llm_client,
            base_tool_executor=tool_executor,
            model=config.llm_model,
            tools_schema=TOOLS_SCHEMA,
            system_prompt=SYSTEM_PROMPT,
        )
        print(format_report(results))
    finally:
        await close_http_clients(http_clients)


if __name__ == "__main__":
    asyncio.run(main())
