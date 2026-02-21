import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent)
sys.path.append(ROOT_PATH)

from agent_os.workflow import AgentWorkflow


from utils.common import generate_unique_filename
from constants import *


DEFAULT_USER_INPUT = "Design an apartment about 100 square meters."
DEFAULT_AI_MODEL = "gemini-3-pro-preview"

parser = argparse.ArgumentParser(description="Run the LLM agents pipeline.")
parser.add_argument(
    "--input",
    "-i",
    type=str,
    default=DEFAULT_USER_INPUT,
    help=f'Textual design prompt (default: "{DEFAULT_USER_INPUT}")',
)
parser.add_argument(
    "--model",
    "-m",
    type=str,
    default=DEFAULT_AI_MODEL,
    help=f"AI model name to use (default: {DEFAULT_AI_MODEL})",
)
args = parser.parse_args()

agent_name = "workflow"
initial_input = {
    "user_input": {"text_input": args.input},
    "ai_model": {
        "default": args.model,
    },
}

session_id = generate_unique_filename()
session_dir = PATH_OF_MEMORY / session_id
os.makedirs(session_dir, exist_ok=True)
file_path_user_info_stream = f"{session_dir}/user_info.md"


async def main():
    workflow = AgentWorkflow(session_id=session_id)

    with open(file_path_user_info_stream, "a", encoding="utf-8") as f:
        async for output in workflow.run(agent_name, initial_input):
            print(output, end="")
            f.write(output)
            f.flush()
    print(
        f"\nConversation saved to files:\nI/O log: {workflow.file_path_info}\nMemory state log: {workflow.file_path_memory}\nStreaming output log: {file_path_user_info_stream}"
    )


asyncio.run(main())
