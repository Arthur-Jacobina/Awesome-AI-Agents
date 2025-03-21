from typing import TypedDict, Annotated, Literal, List

class RouterState(TypedDict):
    messages: List
    next: Annotated[Literal["agent", "direct_weather", "general"], "The next node to call"]