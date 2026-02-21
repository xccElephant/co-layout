#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import os
import sys
import time


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_FILE = os.path.join(REPO_ROOT, "key", "llm_key.json")
DEFAULT_PROMPT = "Hello, how are you?"


def parse_args():
    parser = argparse.ArgumentParser(description="Quick API connectivity test")
    parser.add_argument(
        "--model",
        default=os.environ.get("AI_MODEL", "deepseek-chat"),
        help="Model name, e.g. deepseek-chat, gpt-4o, gemini-2.5-pro-preview-06-05",
    )
    parser.add_argument(
        "--prompt",
        default=os.environ.get("API_TEST_PROMPT", DEFAULT_PROMPT),
        help="Prompt to send",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("API_TEST_TIMEOUT", "60")),
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming response",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug information (URL, raw response, etc.)",
    )
    return parser.parse_args()


def ensure_key_file_exists():
    if os.path.exists(KEY_FILE):
        return True
    print("Missing key file: key/llm_key.json")
    print("Run: cp key/llm_key_example.json key/llm_key.json")
    return False


async def run_test(model, prompt, timeout, stream, debug=False):
    sys.path.insert(0, REPO_ROOT)
    from agent_os.base_llm.base_api import async_talk_to_llm, LLMResult

    messages = [{"role": "user", "content": prompt}]
    result = LLMResult()

    async def _call():
        async for _chunk in async_talk_to_llm(
            model, messages, is_stream=stream, debug=debug, result=result
        ):
            pass  # consume the generator; stats are automatically written to result

    try:
        start_time = time.time()
        await asyncio.wait_for(_call(), timeout=timeout)
        elapsed_total = round(time.time() - start_time, 3)
    except asyncio.TimeoutError:
        return False, f"Timeout after {timeout}s"
    except Exception as exc:
        return False, f"Request failed: {exc}"

    if not result.full_text.strip():
        return False, "No response received"

    return (
        True,
        f"(elapsed={result.elapsed_time}s, total={elapsed_total}s, "
        f"input_tokens={result.input_tokens}, output_tokens={result.output_tokens}, "
        f"price=${result.price})\n"
        f"Response: {result.full_text.strip()!r}",
    )


def main():
    GREEN = "\033[32m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RESET = "\033[0m"

    if not ensure_key_file_exists():
        return 2

    args = parse_args()
    print(f"{YELLOW}Testing model: {args.model}{RESET}")
    ok, message = asyncio.run(
        run_test(args.model, args.prompt, args.timeout, not args.no_stream, args.debug)
    )
    if ok:
        print(f"{GREEN}OK{RESET} {message}")
        return 0
    print(f"{RED}FAILED: {message}{RESET}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
