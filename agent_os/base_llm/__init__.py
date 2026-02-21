#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base_api import async_talk_to_llm, LLMResult

__all__ = [
    "async_talk_to_llm",
    "LLMResult",
]

"""
The base_llm module provides a unified interface for interacting with different large language models (LLMs).

Available interfaces:
- async_talk_to_llm: asynchronously interact with a specified LLM, streaming text chunks.
- LLMResult: dataclass for collecting call statistics (token counts, elapsed time, price).

Usage example:

```python
import asyncio
from agent_os.base_llm import async_talk_to_llm, LLMResult

async def main():
    result = LLMResult()
    messages = [{"role": "user", "content": "Hello"}]
    async for chunk in async_talk_to_llm("gpt-4o", messages, result=result):
        print(chunk, end="")
    print(f"Price: ${result.price}")

asyncio.run(main())
```
"""