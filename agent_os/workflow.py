import os
import sys
import importlib.util
import asyncio
import json
from typing import Dict, List, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import *


class AgentWorkflow:
    def __init__(
        self,
        session_id="test",
        memory_path=str(PATH_OF_SESSIONS),
    ):
        self.agents = []
        self.memory = {}
        self.memory_lock = asyncio.Lock()
        self.session_id = session_id
        self.memory_path = memory_path
        if self.memory_path.startswith("./"):
            self.memory_path = self.memory_path[2:]
            self.memory_path = os.path.normpath(
                os.path.join(os.getcwd(), self.memory_path)
            )
        elif not os.path.isabs(self.memory_path):
            self.memory_path = os.path.normpath(
                os.path.join(os.getcwd(), self.memory_path)
            )
        print("Using memory path:", self.memory_path)
        self.session_dir = os.path.join(self.memory_path, self.session_id)
        os.makedirs(self.session_dir, exist_ok=True)
        self.file_path_info = os.path.join(
            self.session_dir, "debug_info.md"
        )
        self.file_path_memory = os.path.join(
            self.session_dir, "memory.json"
        )
        self.load_agents()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_elapsed_time = 0
        self.total_price = 0

    def load_agents(self):
        """
        Load agents from the built-in agents directory.
        """
        agent_dirs = [
            os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../agents")
            )
        ]

        for agents_dir in agent_dirs:
            if not os.path.isdir(agents_dir):
                print(f"Agent directory not found: {agents_dir}")
                continue
            sys.path.insert(0, agents_dir)
            for root, dirs, files in os.walk(agents_dir):
                for filename in files:
                    if filename.endswith(".py") and filename != "__init__.py":
                        module_path = os.path.join(root, filename)
                        module_name = os.path.splitext(os.path.basename(filename))[0]
                        relative_path = os.path.relpath(module_path, agents_dir)
                        module_spec_name = relative_path.replace(os.sep, ".")[:-3]

                        spec = importlib.util.spec_from_file_location(
                            module_spec_name, module_path
                        )
                        if spec is None:
                            continue
                        module = importlib.util.module_from_spec(spec)
                        try:
                            spec.loader.exec_module(module)
                        except Exception as e:
                            print(f"Failed to load module {module_spec_name}: {e}")
                            continue
                        class_name = (
                            "".join(
                                word.capitalize() for word in module_name.split("_")
                            )
                            + "Agent"
                        )
                        if hasattr(module, class_name):
                            agent_class = getattr(module, class_name)
                            agent = agent_class(
                                name=module_name,
                                memory_path=self.memory_path,
                                session_id=self.session_id,
                            )
                            self.agents.append(agent)
                        else:
                            print(
                                f"Module {module_spec_name} does not have class {class_name}"
                            )
            sys.path.pop(0)

    async def run(self, initial_actions: Union[str, List[str]], initial_input: Dict):
        self.memory = initial_input

        if isinstance(initial_actions, list):
            self.memory["actions"] = initial_actions
        else:
            self.memory["actions"] = [initial_actions]

        with open(self.file_path_memory, "w", encoding="utf-8") as file:
            file.write(json.dumps(self.memory, indent=4, ensure_ascii=False))

        while "actions" in self.memory and self.memory["actions"]:
            action = self.memory["actions"].pop(0)

            # iterate over outputs using an async generator
            async for output in self.process_single_action(action):
                yield output  # yield each Agent output

            if self.memory.get("workflow_complete"):
                return

        print("\n------\n")
        print("\n# Agent Workflow completed.")

        with open(self.file_path_info, "a", encoding="utf-8") as file:
            file.write("\n------\n")
            file.write("\n# Agent Workflow completed.\n")
            file.write("# Final memory state:\n")
            file.write(
                f"```json\n{json.dumps(self.memory, indent=4, ensure_ascii=False)}\n```\n"
            )
            file.write(f"\nTotal Input Tokens: {self.total_input_tokens}\n")
            file.write(f"Total Output Tokens: {self.total_output_tokens}\n")
            file.write(
                f"Total Elapsed Time: {self.total_elapsed_time:.2f} seconds\n"
            )
            file.write(f"Total Price: ${self.total_price:.4f}\n")
        with open(self.file_path_memory, "w", encoding="utf-8") as file:
            file.write(json.dumps(self.memory, indent=4, ensure_ascii=False))

    async def process_single_action(self, action):
        if isinstance(action, dict):
            async for output in self.execute_action_dict(action):
                yield output
        elif isinstance(action, str):
            async for output in self.execute_agent(action):
                yield output
        else:
            print(f"Invalid action: {action}")

    async def execute_action_dict(self, action_dict):
        if "sequential" in action_dict:
            for sub_action in action_dict["sequential"]:
                async for output in self.process_single_action(sub_action):
                    yield output
                if self.memory.get("workflow_complete"):
                    break
        else:
            print(f"Invalid action dict: {action_dict}")

    async def execute_agent(self, agent_name):
        agent = self.get_agent_by_name(agent_name)
        if not agent:
            print(f"Agent '{agent_name}' not found.")
            return  # use bare return to end the generator

        agent.memory = self.memory
        agent.memory_lock = self.memory_lock
        async for output in agent.run_agent():
            yield output

        self.total_input_tokens += agent.input_num_tokens
        self.total_output_tokens += agent.output_num_tokens
        self.total_elapsed_time += agent.elapsed_time
        self.total_price += agent.price

    def add_actions_to_memory(self, new_actions):
        # Ensure 'actions' in memory is a list
        if not isinstance(self.memory.get("actions", []), list):
            self.memory["actions"] = (
                [self.memory.get("actions")] if self.memory.get("actions") else []
            )
        if isinstance(new_actions, dict):
            if "sequential" in new_actions:
                # Insert sequential actions at the front of the actions list
                self.memory["actions"] = (
                    new_actions["sequential"] + self.memory["actions"]
                )
            else:
                print(f"Invalid action dict: {new_actions}")
        elif isinstance(new_actions, list):
            # Treat list as sequential actions
            self.memory["actions"] = new_actions + self.memory["actions"]
        elif isinstance(new_actions, str):
            # Single action, insert at the front
            self.memory["actions"].insert(0, new_actions)
        else:
            print(f"Invalid action type: {new_actions}")

    def get_agent_by_name(self, name):
        return next((agent for agent in self.agents if agent.name == name), None)


# Example usage
if __name__ == "__main__":
    import argparse

    from utils.unique_filename import generate_unique_filename

    parser = argparse.ArgumentParser(description="Run Agent Workflow")
    parser.add_argument("agent_name", type=str, help="Name of the agent to run")
    parser.add_argument(
        "user_text_input",
        type=str,
        nargs="?",
        default="",
        help="User text input (can be empty)",
    )
    parser.add_argument("ai_model", type=str, help="AI model to use", default="deepseek-chat")
    args = parser.parse_args()

    user_input = args.user_text_input

    initial_input = {
        "user_input": {"text_input": user_input},
        "ai_model": {
            "default": args.ai_model,
        },
    }

    client_data_str = json.dumps(initial_input, indent=4, ensure_ascii=False)

    session_id = generate_unique_filename()
    # create the sessions/session_id directory
    session_dir = PATH_OF_SESSIONS / session_id
    os.makedirs(session_dir, exist_ok=True)
    file_path_user_info_stream = f"{session_dir}/user_info.md"

    async def main():
        workflow = AgentWorkflow(session_id=session_id)

        with open(file_path_user_info_stream, "a", encoding="utf-8") as f:
            async for output in workflow.run(args.agent_name, initial_input):
                print(output, end="")
                f.write(output)
                f.flush()
        print(
            f"\nSession saved to files:\nInput/Output log: {workflow.file_path_info}\nMemory state log: {workflow.file_path_memory}\nStream output log: {file_path_user_info_stream}"
        )

    asyncio.run(main())