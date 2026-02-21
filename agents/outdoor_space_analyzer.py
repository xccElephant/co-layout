from agent_os.base_agent import BaseAgent


class OutdoorSpaceAnalyzerAgent(BaseAgent):
    def setup(self):
        self.agent_type = "tool"
        self.role = "You are a professional architectural modeling analyst. Your team is working on a building design. The architectural design requires a precise definition of the boundaries of the buildable areas, especially when site constraints, user requirements or design shapes prevent the use of some areas for the layout of the interior spaces (e.g., recessed corners of an L-shaped building, interior courtyards, necessary setbacks, etc.). Your task is to analyze these situations and determine which areas do not belong to the buildable interior space."
        self.user_info = "\n## Outdoor space being analyzed...\n"
        self.prompt = """
## Task
Your task is to analyze user requirements, site basic information (`basic_information`) and environmental conditions (`environment_analysis`) to precisely identify and define which grid cells in the total planning area (within `width * length`) of each floor belong to the **not available used for internal building layout** (defined as `outdoor_space`). These areas may be areas that must be left open due to site constraints, code setbacks, preserved open space, user-specified special shapes (e.g., courtyards, patios), or areas that must be left open in order to create a particular building form (e.g., L-shape).

Your output will clearly define these "outdoor_space" areas, and in turn, the remaining area (`width * length` - `outdoor_space` area) will be the **effective building occupancy (footprint)** that can be used for internal room layout on that floor. This clear boundary is essential for subsequent space planning and room layout.

## Floorplan coordinate system and slice description 
The up direction of the floorplan is defined as north, and the coordinate system is represented by 'i' (north-south) and 'j' (east-west), with the northwest corner as the origin (0, 0).' width' is the range in the north-south direction, 'length' is the range in the east-west direction.
“outdoor_space” is represented by a number of rectangular regions (a list of coordinate slices). Each rectangular region is represented by a slice `"i_start:i_end,j_start:j_end"`, with the range `[i_start, i_end]` in the i-direction and `[j_start, j_end]` in the j-direction. Note the check for plausibility: `i_start < i_end` and `j_start < j_end` must be satisfied.

## Input Content Description
- user_input: The user's original requirements document (text).
- basic_information: The basic information of the building from the user input.
- environment_analysis: The analysis of the environment conditions.

## Core requirements
- Identify constraints: Prioritize the identification of areas that must be left open, e.g., courtyards/patios explicitly requested by the user, necessary fire/sunlight setback distances, topographically constrained areas of the site itself, protection objects to be avoided, etc.
- Shaping the form: According to the user's requirements or design concepts (e.g., L-shape, U-shape), determine the areas that need to be set aside for the formation of a specific architectural form.
- Guarantee Usability: Ensure that after the "outdoor_space" is set aside, the remaining **valid building occupancy area** is relatively intact and connected, so as to facilitate the subsequent reasonable room layout. Avoid creating too many fragmented and difficult to utilize corners.
- Reasonable Inference: If the input information is insufficient, a reasonable inference based on building design principles and experience must be made and stated in the `reason`.
- Pay attention to the difference between the building sizes (width x length) and the actual required area. You can set some areas as outdoor areas to make the two(width*length and user's requirement) more similar.

## Output Content Description
The output must be in strict JSON format, without any additional explanatory text or markup. The JSON object should contain the following:

- "reason": (String) A detailed analysis of the reasons and thought processes that led to the identification of these "outdoor_spaces", reflecting the core requirements and analysis points above. Set the outdoor_space as concise as possible and try not to overlap.
- "value": (List[String], e.g. ["0:2,0:2", "10:12,0:10"]) Define a number of rectangular areas for the "outdoor_space". The list can be empty `[]` (indicating that building completely fills the `width * length` area).


## Current Actual Input
Input:
{{"user_input": {user_input[text_input]}, "basic_information": {basic_information}, "environment_analysis": {environment_analysis}}}
"""

    async def post_processing(self, output, memory):
        return {"outdoor_space_analysis": output}
