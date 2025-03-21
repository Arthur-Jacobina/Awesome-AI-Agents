import random
from langchain_core.tools import BaseTool

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get the current weather for a location. The input should be a string specifying a location."
    
    def _run(self, location: str) -> str:
        conditions = ["Sunny", "Cloudy", "Partly Cloudy", "Rainy", "Thunderstorm", "Snowy", "Windy", "Clear"]
        temperature = random.randint(-10, 35)
        
        result = {
            "location": location,
            "temperature": f"{temperature}°C",
            "conditions": random.choice(conditions)
        }
        
        return f"Weather for {location}: {result['temperature']}, {result['conditions']}"
        
    async def _arun(self, location: str) -> str:
        return self._run(location)