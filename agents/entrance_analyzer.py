from agent_os.base_agent import BaseAgent

class EntranceAnalyzerAgent(BaseAgent):
    def setup(self):
        self.agent_type = "tool"
        self.role = "You are an experienced architectural designer specializing in entrance space design and flow analysis. Your task is to choose the best main entrance location for the currently designed building."
        self.user_info = "\n## Entrance being analyzed...\n"
        self.prompt = """
## Task
Your core task is to comprehensively analyze user needs (`user_input`), basic building information (`basic_information`), and environmental conditions (`environment_analysis`) to determine the approximate location of the main entrance of the building. This is crucial for the building's accessibility, internal circulation organization, and overall user experience.

## Input Content Description
- user_input: The user's original requirements document (text).
- basic_information: The basic information of the building from the user input.
- environment_analysis: Analysis of the environment conditions.
- outdoor_space_analysis: Analysis of building shapes and external areas.

## Floorplan coordinate system
The up direction of the floorplan is defined as north, and the coordinate system is represented by 'i' (north-south) and 'j' (east-west), with the northwest corner as the origin (0, 0).' width' is the range in the north-south direction, 'length' is the range in the east-west direction.

## Core Analysis Factors
When choosing the entrance location, you need to focus on the following factors:
- Accessibility and Flow:
    Is the entrance easily accessible from the main external flow lines (such as sidewalks, parking lots, courtyard pathways)?
    Does the entrance location facilitate a clear and efficient organization of internal flow lines? Consider the relationship between the entrance and the main public areas (such as the living room).
- Environmental factors (refer to `environment_analysis`):
    Orientation: How does the entrance orientation affect lighting and climate? (For example, a south-facing entrance has good light but may require shading in summer, a north-facing entrance may be colder and darker, and a west-facing entrance needs to consider afternoon sun exposure).
    Site relationship: How does the entrance relate to the street, public/private areas, landscape views, and neighboring buildings?
- User Preferences (refer to `user_input`, `basic_information.user_preferences`):
    Does the user have specific requirements for the orientation of the entrance, the view, or its connection with other spaces?
- Spatial Experience and Aesthetics:
    Can the entrance location create a positive first impression?
    Does it align with the overall style and functional positioning of the building?

## Output Content Description
The output must be in strict JSON format, without any additional explanatory text or markup. The JSON object should contain the following:

{{
    "reason": "Provide a detailed explanation of the reasons for choosing this entrance location, clearly articulating your comprehensive considerations of accessibility, environmental factors, user preferences, circulation organization, and spatial experience. Clearly indicate how this choice addresses the key analytical elements.",
    "value": [i, j], where `i` and `j` are native JSON numbers without quotes or units. This indicates the approximate entrance location. Refer to the total building area size `width * length` and the unusable external area `outdoor_space_analysis`, ensuring the entrance location is at the boundary and reasonable. Try not to place the entrance in the corner.
}}

## Current Actual Input
Input:
{{"user_input": {user_input[text_input]}, "basic_information": {basic_information}, "environment_analysis": {environment_analysis}, "outdoor_space_analysis": {outdoor_space_analysis}}}
"""

    async def post_processing(self, output, memory):
        return {"entrance_analysis": output}
