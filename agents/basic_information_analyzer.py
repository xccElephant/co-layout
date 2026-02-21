from agent_os.base_agent import BaseAgent


class BasicInformationAnalyzerAgent(BaseAgent):
    def setup(self):
        self.agent_type = "tool"
        self.role = "You are an experienced architectural designer. Your team is working on an architectural design assignment. Your primary responsibility is to extract key building parameters, user preferences, and design constraints from the client's original requirements to form a structured initial design basis for the team to follow up with detailed design."
        self.user_info = "\n## Analyzing Basic Information from User Input...\n"
        self.prompt = """
## Task
As the first step in the design process, you are tasked with carefully interpreting the original `user_input`, extracting and organizing the core parameters, user preferences, and constraints of the building project. You need to recognize explicit information and make reasonable inferences about missing or ambiguous information based on design experience and common sense.

## Input Content Description
- user_input: The user's original requirements document (text), which may contain descriptions of the building's dimensions, features, style, family members, etc.

## Core requirements and caveats
- Comprehensive information: not only extract the explicitly stated information, but also pay attention to exploring the implicit requirements and constraints.
- Reasonable Inference: When there is insufficient information, inferences must be made based on building design principles, codes, and common sense, and the logic and rationale for the inferences must be clearly stated in the `reason` field. Avoid ambiguous answers.
- Bounding box definition: `building_envelope` has `width` and `length` which define the maximum perimeter rectangular boundary of the house**, `length * width` is equal to the footprint.
- Data type: Make sure that the data type of the output value meets the requirements (Integer, Float, List, Dict, String). In particular, dimensions, area, etc. are usually required to be integers.
- JSON format: The output must be in pure JSON format, without any comments, explanatory text, or other non-JSON content.
- Sufficiently justified: The `reason` field is critical, it explains your process of judgment and inference, and is key to the transparency and traceability of the design process.

## Output Content Description
The output must be in strict JSON format, without any additional explanatory text or markup. The JSON object should contain the following:

- "building_parameters": Basic quantitative parameters.
    - "building_envelope":
        - "reason": "Analyze whether the user mentions a specific plot size or building footprint, if so, then the footprint indicates the area of land covered by the building, including indoor and outdoor facilities. If not, estimate a **maximum perimeter bounding box** size that can accommodate the entire project based on the desired building form. length is the east-west length, width is the north-south length in meters, and is required to be an integer."
        - "value": A **Dict** for defining the overall bounding box size: `{{"width": Integer, "length": Integer}}`
    - "target_floor_height":
        - "reason": "Analyze whether the user has special requirements for floor height (such as high living rooms). If not, then based on building specifications and common practices, set standard floor heights (e.g., residential 2.8-3.0 meters). If the floor functions differ greatly (e.g., containing commercial, public spaces), it may be necessary to set different floor heights. Specify the basis for setting the heights."
        - "value":The height of the floors (meters, **Float**).
- "user_preferences":
    - "reason": "Summarize the key preferences and lifestyles of the users as reflected in the description of their needs. For example, requirements for light, ventilation, views, and quietness; preferences for specific styles (modern, Chinese); important family activities (cooking, gathering, working from home, working out); and considerations for privacy, openness, environmental protection, and technology integration. Describe how these preferences can be distilled from the original inputs."
    - "value": A text summary of the user's key preferences and lifestyles. (**String**)
- "family_structure":
    - "reason": "Analyze the user's family structure, including the number of permanent residents, age layering (adults, children, seniors), member relationships, and whether there are pets. If the information is incomplete, infer based on room requirements or other clues, and specify the basis for the inference."
    - "value": A **Dict** containing family member information.

## Current Actual Input
Input:
{{"user_input": {user_input[text_input]}}}
"""

    async def post_processing(self, output, memory):
        return {"basic_information": output}
