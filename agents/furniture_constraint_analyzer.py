from agent_os.base_agent import BaseAgent


class FurnitureConstraintAnalyzerAgent(BaseAgent):
    def setup(self):
        self.agent_type = "tool"
        self.role = "You are a professional furniture layout designer. Your team is undertaking an architectural design project. The layout and interrelationship of furniture have a decisive impact on the functionality and comfort of a space. Therefore, each expert in your team needs to be assigned a specific cast, with each person responsible for one pleat, and finally, all results should Discounted to form a complete design workflow."
        self.user_info = "\n## Furniture constraints being analyzed...\n"
        self.prompt = """
## Task
You are downstream in this workflow. Your task is to establish constraint relationships between the furniture in each room based on the existing room analysis and furniture list. You need to consider the function, size, usage, and the user's living habits of the furniture, and provide detailed positional relationship constraints for each pair of related furniture.

## Input Content Description
- user_input: The user's original requirements document (text).
- basic_information: The basic information of the building from the user input.
- room_analysis: Room information list.
- room_constraint_analysis: The analysis of room constraints.
- furniture_analysis: Furniture information list.

## Key requirements
- Extract comprehensive information as much as possible, including implicit needs and constraints.
- If the user has not provided certain key information, make reasonable inferences based on design experience and context; do not give vague answers.
- The spacing between furniture should consider ergonomic principles and user comfort.
- Relative positioning should consider the functional relevance and usage flow of the furniture, taking into account their functionality, comfort, and aesthetics.
- Note that the distance constraint is defined relative to the furniture orientation. For example, ['chair','table',1,0] means the **chair should be 1 meter in front of the table** and **0 meters to the right**. If it is ['chair','table',-1,-0.2], it means the chair is 1 meter behind the table and 0.2 meters to the left. Another example: ['coffee_table','sofa',1,0] means the coffee table is 0.5 meters in front of the sofa and 0 meters to the right. ['tv_stand','sofa', 3, 0.5] means the TV stand is 3 meters in front of the sofa and 0.5 meter to the right.
- Note that the distance constraint refers to the distance between the **centers of two objects**, so it is necessary to fully consider the **sizes** of the furniture itself. For example, two 0.5m x 0.5m nightstands should be two sides(left/right) of a 1m x 2m bed, so considering the dimensions, the horizontal parameter can be set to 0.5/2 + 1/2 + spacing of about 1m, and the vertical parameter can be set to 2/2 - 0.5/2 of about 0.75m (Nightstands are generally placed at the head of the bed, and therefore **behind** the center of the bed), and so the constraints can be **['nightstand1', 'bed', -0.75, 1] and ['nightstand2', 'bed', -0.75, -1]**.
- Boundary constraints indicate whether the furniture is against the wall. For example, in general, nightstands and beds should be placed on boundary.
- Align constraint indicates that two pieces of furniture should be oriented in the same direction. (For example, Bed and Nightstand)
- Facing constraint indicates that one piece of furniture should face another piece of furniture, for example ["chair", "table"] means the chair should face to the table.
- Every distance offset must be a native JSON number in meters, without quotes or unit suffixes. Negative values and zero are allowed.

## Output Content Description
The output must be in strict JSON format, without any additional explanatory text or markup. The JSON object should contain the following:

{{
    "RoomName1": {{
        "overall_description": Provide a detailed overall description of the furniture layout in this room. You can imagine the approximate positions and orientations of each piece of furniture in the room, as well as the relationships between the furniture. Consider the functional flow of the room and the direction of the pathways. This description should serve as the foundation and guide for subsequent analysis, and future analyses must be consistent with this description. When setting furniture constraints, ensure that these constraints are **sufficiently comprehensive** to achieve the envisioned furniture layout of the room, but be careful **not to add redundant or contradictory constraints**.,
        "boundary": ["Name1", "Name2", ...],
        "distance": [["Name1", "Name2", 1, 0.5], [...], ...], where the third and fourth values are JSON numbers,
        "align": [["Name1", "Name2"], [...], ...],
        "facing":[["Name1", "Name2"], [...], ...]
    }},
    "RoomName2": {{ ... }},
    ...
}}

## Current Actual Input
Input:
{{"user_input": {user_input[text_input]}, "basic_information": {basic_information}, "room_analysis": {room_analysis}, "room_constraint_analysis": {room_constraint_analysis}, "furniture_analysis": {furniture_analysis}}}
"""

    async def post_processing(self, output, memory):
        return {"furniture_constraint_analysis": output}