import re
import os
import asyncio
import json
import copy
from typing import Any, Dict, List, Union
from .base_llm import async_talk_to_llm, LLMResult


class AgentOutputError(RuntimeError):
    """Raised when a tool-type agent cannot produce a usable structured
    output after exhausting its JSON retry budget.

    This is intentionally NOT swallowed anywhere: writing a placeholder
    like {"text": ""} into memory.json instead of raising would let the
    pipeline silently continue with unusable data, only to fail several
    steps later (e.g. deep inside run_optimization.py) with a confusing,
    unrelated-looking error. Failing fast here, at the source, makes the
    real cause immediately visible.
    """


class BaseAgent:
    def __init__(self, name: str, memory_path: str, session_id="test"):
        self.name = name
        self.memory = {}
        self.memory_lock = asyncio.Lock()
        self.memory_path = memory_path

        self.agent_type = "tool"  # agent type: tool or plan
        self.role = ""
        self.user_info = ""
        self.prompt = ""
        self.ai_model = None
        self.session_id = session_id
        self.setup()

        self.session_dir = os.path.join(self.memory_path, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self.file_path_info = os.path.join(
            self.session_dir, "debug_info.md"
        )
        self.file_path_memory = os.path.join(
            self.session_dir, "memory.json"
        )

        self.input_num_tokens = 0
        self.output_num_tokens = 0
        self.elapsed_time = 0
        self.price = 0
        self.max_tool_json_retries = 2

    def setup(self):
        """
        Required method for each Agent to implement, initializes the agent's basic configuration.
        Typically sets the following attributes:
        agent_type: specifies the agent type (tool/plan)
        role: defines the agent's role and task
        user_info: information displayed to the user
        prompt: main prompt template
        """
        pass

    async def post_processing(
        self, output: Dict, memory: Dict
    ) -> Dict:
        """
        Method each Agent should implement to process LLM output. Main purposes:
        Parse and format the LLM output.
        Store results into memory.
        Decide next actions to execute.
        Must return a dict containing the processed results.
        """
        return output

    def parse_json_with_fallback(self, text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            json_matches = re.findall(r"(\{[\s\S]*\})", text)
            for match in json_matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    fixed_match = self.fix_common_json_errors(match)
                    try:
                        return json.loads(fixed_match)
                    except json.JSONDecodeError:
                        continue
            fixed_text = self.fix_common_json_errors(text)
            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                return text

    def fix_common_json_errors(self, text):
        text = re.sub(r"(?<=[:,\s])'([^']*)'", r'"\1"', text)
        text = re.sub(r'\\(["\\/bfnrt]|u[a-fA-F0-9]{4})', r"\1", text)
        text = re.sub(r"[\x00-\x1F\x7F]", "", text)
        text = re.sub(r",(\s*[}\]])", r"\1", text)
        text = text.strip()
        return text

    def clean_code_block(self, text):
        # strip markdown code block markers using regex
        code_block_pattern = r"^```(?:\w+)?\n(.*?)\n```$"
        match = re.match(code_block_pattern, text.strip(), re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()

    def _is_retryable_tool_output(self, output_parsed):
        if not isinstance(output_parsed, dict):
            return True
        if not output_parsed:
            return True
        if set(output_parsed.keys()) == {"text"}:
            text_value = output_parsed.get("text")
            return isinstance(text_value, str)
        return False

    def _build_retry_messages(self, base_messages, previous_response, retry_index):
        retry_messages = copy.deepcopy(base_messages)
        retry_messages.append({"role": "assistant", "content": previous_response})
        retry_messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous response is not a complete valid JSON object for this tool task. "
                    f"Retry #{retry_index}: regenerate the full result as ONE strict JSON object only. "
                    "Do not output markdown fences or explanations."
                ),
            }
        )
        return retry_messages

    async def build_messages(self):
        """
        Fill in variables in the prompt template.
        """
        if not self.prompt:
            return None

        format_dict = self.memory.copy()

        try:
            # replace prompt variables with data from memory
            formatted_prompt = self.prompt.format(**format_dict)
        except KeyError as e:
            # missing variable
            print(
                f"Error: {self.name} Missing key {e} in memory. Current memory: {self.memory}"
            )
            return None

        messages = [
            {"role": "system", "content": self.role},
            {"role": "user", "content": f"{formatted_prompt}"},
        ]
        return messages

    async def run_agent(self, return_outputs=True):
        """
        Execute the Agent's logic.
        """
        outputs = []  # initialize output list
        messages = await self.build_messages()
        # print(messages)
        # yield user_info first
        if return_outputs:
            yield self.user_info + "\n"
        else:
            outputs.append(self.user_info + "\n")  # Collect outputs

        if messages is None or self.prompt == "":
            processed_output = await self.post_processing({}, self.memory)
        else:
            ai_model = (
                self.ai_model
                if self.ai_model
                else self.memory.get("ai_model", {}).get("default", "gpt-4o")
            )

            with open(self.file_path_info, "a", encoding="utf-8") as file:
                title = f"\n## Agent Input: {self.name}"
                file.write(title + "\n")
                file.write(
                    f"```json\n{json.dumps(messages, indent=4, ensure_ascii=False)}\n```\n"
                )

            max_retries = self.max_tool_json_retries if self.agent_type == "tool" else 0
            current_messages = messages
            retry_count = 0
            output = {}

            while True:
                llm_result = LLMResult()
                response_generator = async_talk_to_llm(
                    ai_model, current_messages, result=llm_result
                )

                if return_outputs:
                    async for chunk in response_generator:
                        yield chunk  # stream LLM output
                else:
                    async for chunk in response_generator:
                        pass  # consume the generator; stats are collected in llm_result
                    outputs.append(llm_result.full_text)

                full_response = llm_result.full_text
                self.input_num_tokens += llm_result.input_tokens
                self.output_num_tokens += llm_result.output_tokens
                self.elapsed_time += llm_result.elapsed_time
                self.price += llm_result.price

                # clean LLM output
                cleaned_response = self.clean_code_block(full_response)

                # log output
                with open(self.file_path_info, "a", encoding="utf-8") as file:
                    title = f"\n## Agent Output: {self.name}"
                    if retry_count > 0:
                        title += f" [Retry {retry_count}]"
                    file.write(title + "\n")
                    file.write(full_response)

                if self.agent_type == "tool":
                    output_parsed = self.parse_json_with_fallback(cleaned_response)
                    should_retry = self._is_retryable_tool_output(output_parsed)
                    if should_retry and retry_count < max_retries:
                        retry_count += 1
                        print(
                            f"[{self.name}] Invalid/incomplete JSON detected, retrying ({retry_count}/{max_retries})..."
                        )
                        current_messages = self._build_retry_messages(
                            messages, full_response, retry_count
                        )
                        continue
                    if should_retry:
                        # Retry budget exhausted and the output is still not
                        # usable JSON. Fail loudly instead of writing a
                        # placeholder into memory.json that would silently
                        # corrupt every downstream step.
                        raise AgentOutputError(
                            f"[{self.name}] failed to produce valid JSON after "
                            f"{retry_count} retr{'y' if retry_count == 1 else 'ies'}. "
                            f"Last raw LLM response was:\n{full_response}"
                        )
                    output = output_parsed
                elif self.agent_type == "plan":
                    # for plan-type agents, may return only an action name
                    cleaned_response = cleaned_response.strip().strip('"').strip("'")
                    output = {"text": cleaned_response}
                else:
                    # unknown type, return cleaned text directly
                    output = {"text": cleaned_response}
                break

            processed_output = await self.post_processing(output, self.memory)

        async with self.memory_lock:
            has_actions = "actions" in processed_output
            new_actions = processed_output.get("actions")
            payload = {
                key: value
                for key, value in processed_output.items()
                if key != "actions"
            }
            self.memory.update(payload)

            if has_actions:
                if not isinstance(self.memory.get("actions", []), list):
                    self.memory["actions"] = (
                        [self.memory.get("actions")]
                        if self.memory.get("actions")
                        else []
                    )
                if isinstance(new_actions, dict):
                    if "sequential" in new_actions:
                        self.memory["actions"] = (
                            new_actions["sequential"] + self.memory["actions"]
                        )
                    else:
                        print(f"Invalid action dict: {new_actions}")
                elif isinstance(new_actions, list):
                    # prepend new actions to the action queue
                    self.memory["actions"] = new_actions + self.memory["actions"]
                elif isinstance(new_actions, str):
                    # insert single action at the front of the action queue
                    self.memory["actions"].insert(0, new_actions)
                else:
                    print(f"Invalid action type: {new_actions}")

            with open(self.file_path_memory, "w", encoding="utf-8") as file:
                file.write(json.dumps(self.memory, indent=4, ensure_ascii=False))

        # no streaming output needed; collected outputs are available in the outputs list
        if not return_outputs:
            pass
