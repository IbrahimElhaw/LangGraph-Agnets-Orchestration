import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph, START, END

llm = init_chat_model(
    model="gemma3:1b",
    model_provider="ollama",
    temperature=0,
)

def prompt_llm(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph_builder = StateGraph(MessagesState)

graph_builder.add_node(prompt_llm)
graph_builder.add_edge(START, "prompt_llm")
graph_builder.add_edge("prompt_llm", END)

checkpoint = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpoint)
config = {
    "configurable":{"thread_id": str(uuid.uuid4())},
}

user_message=""
while user_message != "exit":
    user_message = input("Enter your message: ")
    answer = graph.invoke({"messages": [{"role": "user", "content": user_message}]},
                          config=config)
    print( answer["messages"][-1].content)