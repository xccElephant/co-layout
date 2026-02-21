from agent_os.base_agent import BaseAgent


class WorkflowAgent(BaseAgent):
    def setup(self):
        self.agent_type = "plan"
        self.user_info = "## Building being generated..."

    async def post_processing(self, output, memory):
        actions = {
            "sequential": [
                "basic_information_analyzer",
                "environment_analyzer",
                "outdoor_space_analyzer",
                "entrance_analyzer",
                "room_analyzer",
                "room_constraint_analyzer",
                "furniture_analyzer",
                "furniture_constraint_analyzer",
            ]
        }
        return {"actions": actions}
