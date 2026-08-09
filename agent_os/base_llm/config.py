#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

# get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# load API keys and base URLs from the config file
with open(os.path.join(current_dir, "../../key/llm_key.json"), "r") as f:
    api_key_json = json.load(f)


# helper function to safely retrieve provider configuration
def get_provider_config(provider_name):
    if provider_name in api_key_json:
        provider_data = api_key_json[provider_name]
        api_base = provider_data.get("api_base", "").rstrip("/")
        return {
            "api_key": provider_data.get("api_key", ""),
            "api_base": api_base,
            "path": provider_data.get("path", "/v1/chat/completions"),
        }
    return None


# define model configurations
llm_model_configs = {}

# add configurations for each provider
for provider in ["claude", "openai", "deepseek", "zhipu", "gemini"]:
    config = get_provider_config(provider)
    if config:
        llm_model_configs[provider] = config


# determine the provider based on the model name
def get_provider_for_model(model_name):
    prefix_map = {
        "gpt": "openai",
        "claude": "claude",
        "deepseek": "deepseek",
        "glm": "zhipu",
        "gemini": "gemini",
    }

    for prefix, provider in prefix_map.items():
        if model_name.startswith(prefix):
            return provider

    return None


# ---------- Price lookup ----------

from litellm import model_cost as _litellm_cost


def get_model_max_output_tokens(model_name, default=8192):
    """Get the maximum output tokens supported by a model.

    Looks up from litellm built-in data; falls back to `default` if the
    model is unknown or does not report a max output token count. This
    lets us request a generation budget that matches what the model can
    actually produce, instead of an arbitrary fixed cap that truncates
    long structured outputs (e.g. multi-room JSON constraint lists).

    Returns:
        int
    """
    cost_info = _litellm_cost.get(model_name)
    if cost_info:
        max_output_tokens = cost_info.get("max_output_tokens") or cost_info.get(
            "max_tokens"
        )
        if max_output_tokens:
            return max_output_tokens
    return default


def get_model_price(model_name):
    """Get the price of a model (unit: USD per million tokens).

    Looks up from litellm built-in data; returns None if not found.

    Returns:
        dict: {"input_price": float, "output_price": float}, or None
    """
    # query litellm (litellm unit is USD/token, convert to USD/million tokens)
    cost_info = _litellm_cost.get(model_name)
    if cost_info:
        input_price = cost_info.get("input_cost_per_token", 0) * 1_000_000
        output_price = cost_info.get("output_cost_per_token", 0) * 1_000_000
        print(
            f"[litellm] model={model_name} input_price={input_price} "
            f"output_price={output_price}"
        )
        return {
            "input_price": input_price,
            "output_price": output_price,
        }

    return None
