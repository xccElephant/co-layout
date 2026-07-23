import argparse
import asyncio
import datetime
import os
import sys
import time
from pathlib import Path

ROOT_PATH = str(Path(__file__).parent)
sys.path.append(ROOT_PATH)

from agent_os.workflow import AgentWorkflow
from run_optimization import synthesis
from constants import *

from utils.unique_filename import generate_unique_filename

async def main():
    parser = argparse.ArgumentParser(description="End-to-End Layout Generation and Optimization")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="Design an apartment about 100 square meters.",
        help='Textual design prompt',
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gemini-3-pro-preview",
        help="AI model name to use",
    )

    args = parser.parse_args()

    agent_name = "workflow"
    user_input = args.input
    ai_model = args.model

    initial_input = {
        "user_input": {"text_input": user_input},
        "ai_model": {
            "default": ai_model,
        },
    }

    session_id = generate_unique_filename()
    session_dir = os.path.join(PATH_OF_SESSIONS, session_id)
    os.makedirs(session_dir, exist_ok=True)
    file_path_user_info_stream = os.path.join(session_dir, "user_info.md")

    print(f"\n{'='*50}")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 1: Starting Agent Workflow (LLM Planning)")
    print(f"User Input: {user_input}")
    print(f"AI Model:   {ai_model}")
    print(f"Session ID: {session_id}")
    print(f"{'='*50}\n")

    workflow = AgentWorkflow(session_id=session_id)

    with open(file_path_user_info_stream, "a", encoding="utf-8") as f:
        async for output in workflow.run(agent_name, initial_input):
            print(output, end="")
            f.write(output)
            f.flush()
            
    print(
        f"\n\n[INFO] Conversation saved to files:\n"
        f"- I/O log: {workflow.file_path_info}\n"
        f"- Memory state log: {workflow.file_path_memory}\n"
        f"- Streaming output log: {file_path_user_info_stream}"
    )

    print(f"\n{'='*50}")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Step 2: Starting Synthesis (Floorplan Optimization Model)")
    print(f"{'='*50}\n")
    
    memory_json_path = workflow.file_path_memory
    
    if not os.path.exists(memory_json_path):
        print(f"[ERROR] Cannot find generated memory.json file: {memory_json_path}")
        print("Optimization stage cannot continue.")
        return

    start_time = time.time()
    try:
        print(f"[INFO] Reading {memory_json_path} and running synthesis optimization...")
        synthesis(session_id)
        end_time = time.time()
        print(f"\n[SUCCESS] Synthesis optimization completed! Elapsed: {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"\n[ERROR] Error during optimization: {e}")

if __name__ == "__main__":
    asyncio.run(main())