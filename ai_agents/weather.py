from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from schemas import RouterState
from utils import get_api_key
from service import WeatherTool

load_dotenv()

class WeatherAgent:
    def __init__(self, api_key: str | None = None):
        try:
            self.api_key = api_key or get_api_key()
        except ValueError as e:
            raise ValueError(
                f"{e!s} Please provide it directly "
                "or set the OPENAI_API_KEY env variable"
            ) from e
    
        self.agent_llm = ChatOpenAI(
            model="gpt-4o", 
            temperature=0.5,
            api_key=self.api_key
        )
        
        self.router_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key=self.api_key
        )

        self.weather_tool = WeatherTool()

        self.agent_prompt = """
            You are a helpful weather assistant. When asked about weather, use the get_weather tool 
            to fetch information. Respond in a friendly, conversational tone, focusing on the weather details requested.
            
            Always include the location, temperature, and conditions in your final response.
        """

        self.agent = create_react_agent(
            self.agent_llm,
            tools=[self.weather_tool],
            prompt=self.agent_prompt,
        )
    
    def _create_nodes(self): 
        return {
            "router": self._router,
            "agent": self.agent, 
            "direct_weather": self._weather_node,
            "general": self._general_node
        }
    
    def _weather_node(self, state: RouterState):
        last_message = state["messages"][-1].content
        weather_info = self.weather_tool._run(last_message.replace("What's the weather in ", "").replace("?", ""))
        return {"messages": state["messages"] + [AIMessage(content=weather_info)]}      
     
    def _general_node(self, state: RouterState):
        last_message = state["messages"][-1].content
        response = self.agent_llm.invoke(
            f"""You are a helpful assistant answering general questions.
                
            User question: {last_message}
                
            Provide a concise, informative response.
            """
            )
        return {"messages": state["messages"] + [AIMessage(content=response.content)]}
    
    def build_graph(self):
        nodes = self._create_nodes()
        
        workflow = StateGraph(RouterState)
        
        for node_name, node_func in nodes.items():
            workflow.add_node(node_name, node_func)
        
        workflow.add_conditional_edges(
            "router",
            lambda state: state["next"],
            {
                "agent": "agent",
                "direct_weather": "direct_weather",
                "general": "general"
            }
        )
        
        workflow.add_edge("agent", END)
        workflow.add_edge("direct_weather", END)
        workflow.add_edge("general", END)
        
        workflow.set_entry_point("router")
        
        return workflow.compile()
    
    def _router(self, state: RouterState) -> RouterState:
            last_message = state["messages"][-1].content
            
            router_prompt = f"""
            Analyze this user request: "{last_message}"
            
            Determine the appropriate handler by responding with exactly ONE of these options:
            - "direct_weather": If this is a simple, direct request for current weather in a specific location.
            - "agent": If this is a weather-related query that may need more reasoning or context.
            - "general": If this is not related to weather at all.
            
            Output only one of these three options with no additional text.
            """
            
            response = self.router_llm.invoke(router_prompt)
            decision = response.content.strip().lower()
            
            if decision not in ["agent", "direct_weather", "general"]:
                decision = "agent" 
                
            state["next"] = decision
            return state 
    
    def query(self, user_input: str):
        graph = self.build_graph()
        config = {"messages": [HumanMessage(content=user_input)]}
        result = graph.invoke(config)
        return result["messages"][-1].content


# RUN python weather.py
if __name__ == "__main__":
        weather_agent = WeatherAgent()
        
        weather_queries = [
            "Should I brind an umbrella to work? I'm currently in NYC",
            "What is the capital of France?",
            "What is the weather in SF?",
        ]
        
        for query in weather_queries:
            print(f"\nQuery: {query}")
            result = weather_agent.query(query)
            print(f"Response: {result}")
        