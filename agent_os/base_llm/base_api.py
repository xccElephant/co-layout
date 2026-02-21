#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import base64
import json
import mimetypes
import os
import sys
import time
from typing import AsyncGenerator

import aiohttp
import tiktoken
from dataclasses import dataclass, field
# add parent directory to sys.path to ensure correct module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)

# import configuration from config module
from config import llm_model_configs, get_model_price, get_provider_for_model


@dataclass
class LLMResult:
    """Collect statistics from an LLM call."""

    full_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_time: float = 0.0
    price: float = 0.0


def format_json_to_print_str(info):
    info = json.dumps(info, indent=4, ensure_ascii=False)
    return info


def print_messages_with_images(messages):
    for message in messages:
        print("{")
        for key, value in message.items():
            if key == "content":
                print('  "content": [')
                for item in value:
                    if item["type"] == "text":
                        print(f'    {{"type": "text", "text": "{item["text"]}"}}')
                    elif item["type"] == "image_url":
                        image_data_url = item.get("image_url", {}).get("url", "")
                        if image_data_url.startswith("data:") and ";base64," in image_data_url:
                            prefix, base64_data = image_data_url.split(";base64,", 1)
                            mime_type = prefix.replace("data:", "")
                            info = {
                                "type": "image_url",
                                "image_url": {
                                    "mime_type": mime_type,
                                    "base64_len": len(base64_data),
                                },
                            }
                        else:
                            info = {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            }
                        print(f"    {json.dumps(info, ensure_ascii=False)}")
                print("  ]")
            else:
                print(f'  "{key}": "{value}"')
        print("}")


def num_tokens_from_messages(messages, model="gpt-4-0314"):
    """
    Count the total number of tokens in the messages, supporting list-type content.
    """
    num_tokens = 0
    tokens_per_message = 3
    tokens_per_name = 1
    encoding = tiktoken.encoding_for_model(model)
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if key == "content":
                # handle list-type content
                if isinstance(value, list):
                    for item in value:
                        if item.get("type") == "text":
                            num_tokens += len(encoding.encode(item.get("text", "")))
                        elif item.get("type") == "image_url":
                            url = item.get("image_url", {}).get("url", "")
                            num_tokens += len(encoding.encode(url))
                elif isinstance(value, str):
                    num_tokens += len(encoding.encode(value))
            elif key == "name":
                num_tokens += tokens_per_name + len(encoding.encode(value))
            else:
                num_tokens += len(encoding.encode(str(value)))
    # every reply starts with <|start|>assistant<|message|>
    num_tokens += 3

    return num_tokens


def process_messages_with_images(messages):
    """
    Process messages that contain an 'image' field.
    For messages with 'image', encode the image as base64 and include it in the appropriate format.
    """
    processed_messages = []
    for message in messages:
        if "image" in message:
            image_path = message["image"]
            try:
                # get MIME type
                mime_type, _ = mimetypes.guess_type(image_path)
                if mime_type is None:
                    print(f"Cannot determine MIME type for {image_path}")
                    continue
                # read and encode the image
                with open(image_path, "rb") as f:
                    image_data = f.read()
                encoded_image = base64.b64encode(image_data).decode("utf-8")
                # remove the 'image' key
                message.pop("image")
                # prepare content list
                content_list = []
                content_text = message.get("content", "")
                if content_text:
                    content_list.append({"type": "text", "text": content_text})
                # append image data
                content_list.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{encoded_image}"
                        },
                    }
                )
                # update message content
                message["content"] = content_list

            except Exception as e:
                print(f"Cannot read image file {image_path}: {e}")
                continue
        else:
            # if content is not a list, wrap it as a list with type 'text'
            content = message.get("content", "")
            if not isinstance(content, list):
                message["content"] = [{"type": "text", "text": content}]
        processed_messages.append(message)
    return processed_messages


async def chat_with_model(
    model_name: str, messages: list, temperature: float = 0.8, stream: bool = True, debug: bool = False
) -> AsyncGenerator[str, None]:
    """Unified model call interface supporting multiple providers.

    Args:
        model_name: model name
        messages: list of messages
        temperature: temperature parameter controlling randomness
        stream: whether to use streaming response
        debug: whether to print debug info (URL, raw response, etc.)
    """
    # determine provider from model name prefix (used to select api_key / api_base / path)
    provider = get_provider_for_model(model_name)

    if not provider or provider not in llm_model_configs:
        raise ValueError(f"Cannot identify provider for model '{model_name}'. Check the model name or add the config to llm_key.json")

    config = llm_model_configs[provider]

    # build request headers and payload
    headers = {"Authorization": f"Bearer {config['api_key']}"}

    # newer OpenAI models (gpt-5) require max_completion_tokens instead of max_tokens
    token_key = "max_completion_tokens" if model_name.startswith("gpt-5") else "max_tokens"

    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
        token_key: 8192,
    }

    max_retries = 3
    retry_delay = 5  # seconds

    url = f"{config['api_base']}{config['path']}"

    if debug:
        GRAY = "\033[90m"
        RESET = "\033[0m"
        print(f"{GRAY}[DEBUG] provider={provider}, url={url}{RESET}")
        print(f"{GRAY}[DEBUG] payload keys: {list(payload.keys())}, stream={stream}{RESET}")

    # send request
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if debug:
                        print(f"{GRAY}[DEBUG] HTTP {response.status}, content_type={response.content_type}{RESET}")
                    if response.status == 200:
                        async for line in response.content:
                            chunk = line.decode("utf-8").strip()
                            if debug and chunk:
                                # truncate long lines to avoid flooding the terminal
                                display = chunk[:200] + ("..." if len(chunk) > 200 else "")
                                print(f"{GRAY}[DEBUG] raw: {display}{RESET}")
                            # skip empty lines, SSE event-type lines, and comment lines
                            if not chunk or chunk.startswith("event:") or chunk.startswith(":"):
                                continue
                            if chunk.startswith("data: "):
                                chunk = chunk[len("data: "):]
                            if not chunk or chunk == "[DONE]":
                                continue
                            try:
                                chunk_data = json.loads(chunk)
                                content = None

                                # OpenAI format: choices[].delta.content / choices[].message.content
                                choices = chunk_data.get("choices")
                                if choices and len(choices) > 0:
                                    choice = choices[0]
                                    content = choice.get("delta", {}).get("content")
                                    if content is None:
                                        content = choice.get("message", {}).get("content")

                                # Gemini format: candidates[].content.parts[].text
                                if content is None:
                                    candidates = chunk_data.get("candidates")
                                    if candidates and len(candidates) > 0:
                                        parts = candidates[0].get("content", {}).get("parts", [])
                                        content = "".join(p.get("text", "") for p in parts)

                                if content:
                                    yield content
                                elif debug:
                                    print(f"{GRAY}[DEBUG] no content in chunk: {list(chunk_data.keys())}{RESET}")
                            except json.JSONDecodeError:
                                pass  # skip lines that cannot be parsed
                        return  # exit after successful processing
                    else:
                        response_text = await response.text()
                        print(f"Error: {response.status}, {response_text}")
                        if attempt < max_retries - 1:
                            print(
                                f"Attempt {attempt + 1}/{max_retries} failed. Retrying in {retry_delay} seconds..."
                            )
                            await asyncio.sleep(retry_delay)
                        else:
                            print("Max retries reached.")
                            return  # or raise an exception
        except aiohttp.ClientOSError as e:
            print(f"ClientOSError occurred: {e}")
            if attempt < max_retries - 1:
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed. Retrying in {retry_delay} seconds..."
                )
                await asyncio.sleep(retry_delay)
            else:
                print("Max retries reached.")
                # optionally re-raise the exception or return an error indicator
                return  # or raise e
        except Exception as e:  # other exception types may also need handling or logging
            print(f"Unexpected error: {e}")
            # for other error types, retrying may not be appropriate
            return  # or raise e

    # all retries exhausted (should have returned or raised inside the loop)
    print("Failed to connect after all attempts.")


async def async_talk_to_llm(
    model_name: str,
    messages: list,
    temperature=0.5,
    is_stream=True,
    debug=False,
    result: LLMResult = None,
) -> AsyncGenerator[str, None]:
    """
    Main function for interacting with an LLM. Processes messages, calls the model,
    and returns results along with usage statistics.

    Args:
        model_name: name of the LLM model to use
        messages: list of messages
        temperature: temperature parameter
        is_stream: whether to use streaming response
        debug: whether to print debug info
        result: optional LLMResult object to collect stats (token counts, elapsed time, price)

    Yields:
        str: streamed text chunks
    """
    YELLOW = "\033[33m"
    RED = "\033[31m"
    RESET = "\033[0m"

    # process messages that contain images
    processed_messages = process_messages_with_images(messages)

    if debug:
        print(f"{RESET}", end="")
        print_messages_with_images(processed_messages)
        print(f"{RESET}", end="")

    # determine provider from model name prefix
    provider = get_provider_for_model(model_name)

    if not provider or provider not in llm_model_configs:
        raise ValueError(f"Cannot identify provider for model '{model_name}'. Check the model name or add the config to llm_key.json")

    # get price configuration
    model_price = get_model_price(model_name)
    if model_price is None:
        print(f"[Warning] Price info not found for model {model_name}. Cost will be shown as 0")
        model_price = {"input_price": 0, "output_price": 0}
    input_token_price_per_1k = model_price["input_price"]
    output_token_price_per_1k = model_price["output_price"]

    input_num_tokens = num_tokens_from_messages(processed_messages)

    if debug:
        print(f"{YELLOW}", end="")
        print("\nInput token count:", input_num_tokens)
        print("Output process:")
        print(f"{RESET}", end="")
        print(f"{RED}\n", end="")

    encoding = tiktoken.encoding_for_model("gpt-4")
    all_data = ""
    start_time = time.time()

    # call the underlying API directly, only yield text
    async for text in chat_with_model(
        model_name, processed_messages, temperature, is_stream, debug=debug
    ):
        all_data += text
        yield text

    end_time = time.time()
    elapsed_time = round(end_time - start_time, 3)
    output_num_tokens = len(encoding.encode(all_data))

    price = round(
        input_num_tokens / 10**6 * input_token_price_per_1k
        + output_num_tokens / 10**6 * output_token_price_per_1k,
        3,
    )

    if debug:
        print(f"{RESET}", end="")
        print(f"{YELLOW}")
        print(
            "\nOutput token count:",
            output_num_tokens,
            "\nElapsed time (s):",
            elapsed_time,
            "\nPrice (USD):",
            price,
        )
        print(f"{RESET}", end="")

    # write statistics into the result object
    if result is not None:
        result.full_text = all_data
        result.input_tokens = input_num_tokens
        result.output_tokens = output_num_tokens
        result.elapsed_time = elapsed_time
        result.price = price
