from agent_os.base_agent import BaseAgent


class RoomAnalyzerAgent(BaseAgent):
    def setup(self):
        self.agent_type = "tool"
        self.role = "You are an experienced interior designer. Your team is undertaking an architectural design task that requires converting user needs into specific, implementable room design plans. Architectural design is a complex process that requires consideration of various factors such as building dimensions, surrounding environment, and room layout. Therefore, each designer in your team needs to be assigned specific tasks, with each person responsible for one aspect. Finally, all the results will be compiled to form a complete design workflow."
        self.user_info = "\n## Rooms being analyzed...\n"
        self.prompt = """
## Task
Your task is to plan the rooms for the building based on user requirements and basic architectural information. You need to set up suitable rooms for the building and provide detailed room information.

## Input Content Description
- user_input: The user's original requirements document (text).
- basic_information: The basic information of the building from the user input.
- environment_analysis: Analysis of the environment conditions.

## Floorplan coordinate system
The up direction of the floorplan is defined as north, and the coordinate system is represented by 'i' (north-south) and 'j' (east-west), with the northwest corner as the origin (0, 0).' width' is the range in the north-south direction, 'length' is the range in the east-west direction.

## Core requirements
- Extract comprehensive information as much as possible, including implicit needs and constraints.
- If the user has not provided certain key information, make reasonable inferences based on design experience and context, and avoid giving vague answers.
- Ensure the reasonableness of room areas and numbers (generally, no more than 7 rooms, with a total area of about **90%** of the total building area).
- The requirements for each room (privacy, lighting, etc.) should be reasonably set according to their function and location, with a range of 1-3, where 1 is the lowest and 3 is the highest.
- Do not provide rooms with outdoor attributes, such as 'Roof Terrace','Balcony' etc. Do not provide rooms like 'Entrance Hall', 'Entrance Room'.

## Output Content Description
The output must be in strict JSON format, without any additional explanatory text or markup. The JSON object should contain the following:
{{
    "RoomName1": {{
        "area": {{
            "reason": "Analyze the required area for the room based on user needs, number of residents, and building standards. Analyze the room's function (such as bedroom, living room, kitchen, etc.), determine what furniture needs to be placed in the room, and pay attention to the area requirements for this type of room as specified in the architectural design standards.",
            "value": "Room area(Square meter)"
        }},
        "open_room": true/false, Some open rooms sometimes serve as passages, such as the Living Room. For these rooms, set to true; for other rooms, set to false.
        "reason": "Analyze the thought process of the following attributes.",
        "activities": ["List of activities that may take place in this room"],
        "privacy": "Privacy (1-3)",
        "natural_light": "Natural light requirement (1-3)",
        "ventilation": "Ventilation requirement (1-3)",
        "noise_control": "Noise control requirement (1-3)",
        "orientation_suggestion": "Orientation suggestion (combined with environment)"
    }},
    "RoomName2": {{ ... }},
    ...
}}

## Current Actual Input
Input:
{{"user_input": {user_input[text_input]}, "basic_information": {basic_information}, "environment_analysis": {environment_analysis}}}
"""

    async def post_processing(self, output, memory):
        return {"room_analysis": output}
