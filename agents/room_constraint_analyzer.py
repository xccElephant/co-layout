from agent_os.base_agent import BaseAgent


class RoomConstraintAnalyzerAgent(BaseAgent):
    def setup(self):
        self.agent_type = "tool"
        self.role = "You are an experienced architectural designer. Your team is undertaking an architectural design task that requires converting user needs into specific, implementable architectural design constraints. Architectural design is a complex process that requires considering various factors such as building size, surrounding environment, room layout, and more. Therefore, each designer in your team needs to be assigned specific tasks, with each person responsible for one aspect. Finally, all the results will be compiled to form a complete design workflow."
        self.user_info = "\n## Room Constraints being analyzed...\n"
        self.prompt = """
## Task
Your task is to receive user requirements and the analysis results generated within the workflow, and to comprehensively consider factors such as architectural parameters, room attributes, and surrounding environment to analyze the **location** and **adjacency** of each room. Ultimately, you need to design a **Scene Graph** to represent the constraints between the rooms.

## Input Content Description
- user_input: The user's original requirements document (text).
- basic_information: The basic information of the building from the user input.
- environment_analysis: The analysis of the environment conditions.
- outdoor_space_analysis: The analysis of building shapes and external areas.
- entrance_analysis: The analysis of the entrance position.
- room_analysis: Room information list.

## Floorplan coordinate system
The up direction of the floorplan is defined as north, and the coordinate system is represented by 'i' (north-south) and 'j' (east-west), with the northwest corner as the origin (0, 0).' width' is the range in the north-south direction, 'length' is the range in the east-west direction.

## Notes
- Extract comprehensive information whenever possible, including implicit needs and constraints.
- If the user has not provided certain key information, make reasonable inferences based on design experience and context, and avoid giving vague answers.
- Spaces with public attributes, such as "Living Room," usually do not need to be placed in corners; corners should prioritize spaces with **private attributes**.
- Note that "corner_rooms" corresponding to the four corners of the house **cannot be duplicated**; they can be empty.
- Please analyze the approximate location of rooms based on the entrance position and the privacy attributes of the rooms. For example, rooms with strong privacy should be far from the entrance.
- **Self-check before output:** Before generating the final JSON, be sure to perform the following checks:
- **Adjacency relationship rationality check:** Ensure that all `adjacent_rooms` relationships conform to the layout described in the `overall_description` and are geometrically feasible (e.g., avoid making rooms at the ends of the east and west sides adjacent).
- **Consistency check between description and constraints:** Ensure that the `value` and `reason` of `adjacent_rooms` are completely consistent with the content of the `overall_description` and do not contradict.
- Do not add excessive adjacency constraints; only add necessary ones to avoid contradictions.
- **Do not make two corner rooms adjacent.**

## Output Content Description
The output must be in strict JSON format, without any additional explanatory text or markup. The JSON object should contain the following:

{{
    "overall_description": "Provide a detailed conceptual description of the overall layout of all rooms. Clearly specify which rooms should be placed in each corner (North West, North East, South West, South East), and outline the necessary adjacency relationships between key rooms based on this layout. Imagine the approximate locations of the rooms on the floor plan, considering the size of the rooms, functional flow, lighting and ventilation, and their relationship with the core shaft/main corridor, to create a **Scene Graph**. This description should serve as a strict foundation and guide for subsequent analysis, and the subsequent analysis must be consistent with this description.",
    "corner_rooms": {{
        "North West": "RoomName1" (Can be empty ""),
        "North East": "RoomName2" (Can be empty ""),
        "South West": "RoomName3" (Can be empty ""),
        "South East": "RoomName4 (Can be empty "")"
    }},
    "adjacent_rooms": [["RoomName1", "RoomName2"], ["RoomName1", "RoomName3"], [...], ...], **Do not make two corner rooms adjacent.** And don't set too many adjacency constraints for one room to avoid conflicts.
}}

## Current Actual Input
Input:
{{"user_input": {user_input[text_input]}, "basic_information": {basic_information}, "environment_analysis": {environment_analysis}, "outdoor_space_analysis": {outdoor_space_analysis}, "entrance_analysis": {entrance_analysis}, "room_analysis": {room_analysis}
}}
"""

    async def post_processing(self, output, memory):
        return {"room_constraint_analysis": output}
