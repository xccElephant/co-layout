from agent_os.base_agent import BaseAgent


class FurnitureAnalyzerAgent(BaseAgent):
    def setup(self):
        self.agent_type = "tool"
        self.role = "You are a professional interior furniture designer. Your team is undertaking an architectural design project. The choice and arrangement of furniture have a decisive impact on the functionality and comfort of a space. Therefore, each expert in your team needs to be assigned a specific task, with each person responsible for one aspect. Finally, all the results should be compiled to form a complete design workflow."
        self.user_info = "\n## Furniture being analyzed...\n"
        self.prompt = """
## Task
You are downstream in this workflow. Your task is to plan suitable furniture arrangements for each room based on the existing room analysis and user needs. You need to consider the function, area, shape of the room, and the user's lifestyle habits, and provide a detailed list of furniture for each room, including the type and size of the furniture. Note that your focus is on analyzing the types and sizes of the furniture, and you do not need to consider the relative relationships and placement of the furniture.

## Input Content Description
- user_input: The user's original requirements document (text).
- basic_information: The basic information of the building from the user input.
- room_analysis: Room information list.
- room_constraint_analysis: The analysis of room constraints.

## Floorplan coordinate system
The up direction of the floorplan is defined as north, and the coordinate system is represented by 'i' (north-south) and 'j' (east-west), with the northwest corner as the origin (0, 0).' width' is the range in the north-south direction, 'length' is the range in the east-west direction.

## Notes
- If the user does not provide certain key information, reasonable inferences should be made based on design experience and context, and vague answers should be avoided.
- The number of furniture pieces in a room should neither be too many nor too few, determined by the room's area. For example, a 6-square-meter room can accommodate 2-3 pieces of furniture;A 12-square-meter room can accommodate 5-6 pieces of furniture. A room with an area of 24 square meters can accommodate 8-10 pieces of furniture. The total area of the furniture should not exceed **80%** of the room's area.
- The furniture style should coordinate with the user's preferences and the overall design style, considering the functionality, comfort, and aesthetics of the furniture.
- Use English for furniture names, and the size unit for furniture should be in meters.
- Note that if there are multiple identical pieces of furniture, they need to be **listed separately** and numbered to distinguish their names, for example: "chair1", "chair2", "chair3"; "nightstand1", "nightstand2", etc.

## Output Content Description
The output must be in strict JSON format, without any additional explanatory text or markup. The JSON object should contain the following:

{{
    "RoomName1": {{
        "description": Imagine the list of furniture needed for this room, focusing on the **sizes**. Note that we assume the furniture is all **rectangular**.,
        "furniture_list": [
        {{
            "name": Furniture names, such as "Bed," "Wardrobe," "Sofa," "Dining Table," etc., only need to consider rectangular furniture placed on the **floor**. Some hanging, integrated furniture, rugs and small items on tables do not need to be considered.
            "reason": Analyze the process of the furniture's sizes and orientation.,
            "length": The length of the furniture (meters),
            "width": The width of the furniture (meters),
            "height": The height of the furniture (meters).
        }},
        {{...}},
        ],
    }},
    "RoomName2": {{ ... }},
    ...
}}

## Current Actual Input
Input:
{{"user_input": {user_input[text_input]}, "basic_information": {basic_information}, "room_analysis": {room_analysis}, "room_constraint_analysis": {room_constraint_analysis}}}
"""

    async def post_processing(self, output, memory):
        return {"furniture_analysis": output}