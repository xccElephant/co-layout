from agent_os.base_agent import BaseAgent


class EnvironmentAnalyzerAgent(BaseAgent):
    def setup(self):
        self.agent_type = "tool"
        self.role = "You are a professional environmental analyst. Your team is working on an architectural design assignment. The design of a building must take into account the surrounding environment, including climatic conditions, site characteristics, orientation analysis and other factors, in order to create a comfortable and livable space. As a result, each member of your team will be required to divide up the work, each taking responsibility for one aspect of the project, and then summarizing all of the results to form a complete design workflow."
        self.user_info = "\n## Environment being analyzed...\n"
        self.prompt = """
## Task
You are at the upstream end of this workflow, where your task is to receive the original user requirements, analyze the site conditions, and provide a **concise** summary of the external environment analysis. The results of these analyses will provide the key basis for subsequent design constraints and space planning. You need to analyze the core characteristics of climate, site, and orientation and their **major influences** on the design.

## Input Content Description
- user_input: The user's original requirements document (text).

## Caution
- The output **must** be in pure JSON format, do not add anything else before or after the JSON.
- Focus on key information and implications that **directly guide** the design and avoid lengthy descriptions.
- If the user does not provide some key information, make reasonable inferences based on design experience and context, and include a brief explanation in the `reason`.

## Output Content Description 
The output must be in strict JSON format, without any additional explanatory text or markup. The JSON object should contain the following:

- "reason": "Comprehensive, generalized description of the entire environmental analysis process (climate, site, orientation), explaining the sources of the main conclusions."
- "climate_impact": Summarize the key climatic characteristics (e.g. temperature range, prevailing wind direction, sunshine characteristics, precipitation) and their core design requirements for building thermal insulation, heat insulation, ventilation, lighting, drainage, etc.
- "site_impact": Summarize the key site characteristics (e.g. shape, size, slope, surrounding environment, main landscape orientation) and their core design requirements for building layout, orientation selection, privacy design, view organization, etc.
- "orientation_summary": Summarize the key orientation characteristics (e.g. lighting, ventilation, privacy) and their core design requirements for building layout, orientation selection, privacy design, view organization, etc.
    - "North":North-facing core features and design recommendations (e.g., light is stable but weak, auxiliary rooms are desirable).
    - "South":South-facing core features and design recommendations (e.g., good lighting and ventilation, desirable for main rooms).
    - "East":East-facing core features and design suggestions (e.g. good morning light, suitable for bedroom-dining room).
    - "West":West-facing core features and design recommendations (e.g., west exposure needs to be addressed, avoid master bedroom).

## Current Actual Input
Input:
{{"user_input": {user_input[text_input]}}}
"""

    async def post_processing(self, output, memory):
        return {"environment_analysis": output}
