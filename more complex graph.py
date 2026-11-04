import uuid
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from rag_helper import vector_store
from pydantic import  BaseModel, Field
from typing import TypedDict, Annotated, Literal
from langgraph.checkpoint.memory import InMemorySaver
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command

CLASSIFYING_PROPMT = ("classify the intent of the user only to these 3 things"
                      " \"chat\", \"knowledge\", \"code\" without other words. Chat is only"
                      "for chatting with out asking for infos. Knowledge for asking about information with a question mark."
                      "and code if there is some issue with code in programming, or the user asks for programming"
                      "functions")

HANDLE_CODE_PROMPT = ("classify if the user wants to accept, deny or change the code. Answer with 'yes', 'no', 'change'"
                      " without including any other words in your answer")



llm = init_chat_model(
    model="gemma3:1b",
    model_provider="ollama",
    temperature=0,
)

class IntentClassification(BaseModel):
    message_intent: Literal["chat", "knowledge", "code"] = Field(..., description=CLASSIFYING_PROPMT)

class HandleCode(BaseModel):
    accept_code: Literal["yes", "no", "change"] = Field(..., description=HANDLE_CODE_PROMPT)

class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    message_intent: str | None
    agent_name: str |   None
    next_node: str | None



classifying_llm = llm.with_structured_output(IntentClassification)
def classify_intent(state: MyState):
    result = classifying_llm.invoke([
        {
            "role": "system", "content": CLASSIFYING_PROPMT

        },
        {
            "role": "user", "content": state["messages"][-1].content
        }
    ])
    return {"message_intent": result.message_intent}


def propmt_llm_chat(state: MyState):
    messages = [
        {
            "role": "system", "content": "you are a talkative chatbot for fun, be nice."
        }

    ] + state["messages"]
    response = llm.invoke(messages)
    return {'messages':[{'role': 'assistant', 'content': response.content}],
            "agent_name": "chatbot"}

def propmt_llm_rag(state: MyState):
    query = state["messages"][-1].content
    documents = vector_store.similarity_search(query, k=3)
    context = "\n".join(f"- {doc.page_content}" for doc in documents)
    messages = [
        {
            "role": "system", "content": "you are a RAG agent. Answer the user using only the context below"
                                         f"If the answer is not in the context, say 'I do not know'. The context is: {context}"
        }

    ] + state["messages"]
    response = llm.invoke(messages)
    return {'messages':[{'role': 'assistant', 'content': response.content}],
            "agent_name": "rag_agent"}

def propmt_llm_code(state: MyState):
    messages = [
        {"role": "system", "content": "You are a coding agent. Write the code the user asks for. Do not include in the"
                                      "answer any thing except the code, so I can run the file directly. Do not include explaining or any thing else"}
    ] + state["messages"]
    response = llm.invoke(messages)

    workspace = Path(__file__).parent / "workspace"
    workspace.mkdir(exist_ok=True)
    path = workspace / f"code.txt"
    path.write_text(str(response.content), encoding="utf-8")

    return {
        "messages": [{"role": "assistant", "content": f"{response.content}\n\nSaved to: {path}"}],
        "agent_name": "code_agent",
    }

accepting_code_llm = llm.with_structured_output(HandleCode)
def accept_coding(state: MyState):
    user_prompt = state["messages"][-1].content
    decision = interrupt(f"do you want to accept the code with the request: \n {user_prompt}\n (yes/no) "
                         "or type a revised request")
    text = str(decision).strip().lower()
    result = accepting_code_llm.invoke([
        {
            "role": "system", "content": HANDLE_CODE_PROMPT

        },
        {
            "role": "user", "content": text
        }
    ])
    print(result.accept_code)
    if result.accept_code == "yes":
        return {"next_node": "accepted",
                "agent_name": "code_agent"}
    if result.accept_code == "no":
        return {"messages":[{"role": "assistant", "content": "Coding was denied by user"}],
                "next_node": "denied",
                "agent_name": "code_agent"
        }
    return {
        "messages":[{"role": "user", "content": text}],
        "next_node": "change",
        "agent_name": "code_agent"
    }
##################################################################

graph_builder = StateGraph(MyState)

graph_builder.add_node("classifier", classify_intent)
graph_builder.add_node("chat_agent", propmt_llm_chat)
graph_builder.add_node("rag_agent", propmt_llm_rag)
graph_builder.add_node("code_agent", propmt_llm_code)
graph_builder.add_node("accept_coding", accept_coding)

graph_builder.add_edge(START, "classifier")
graph_builder.add_conditional_edges("classifier",
                                    lambda state: state["message_intent"],
                                    {'chat': 'chat_agent',
                                            'knowledge': 'rag_agent',
                                            'code': 'accept_coding'})
graph_builder.add_edge("chat_agent", END)
graph_builder.add_conditional_edges(
    "accept_coding",
    lambda state: state.get("next_node"),
    {
        "denied": END,
        "accepted": "code_agent",
        "change": "classifier",
    },
)
graph_builder.add_edge("rag_agent", END)
graph_builder.add_edge("code_agent", END)

checkpoint = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpoint)
config : RunnableConfig = {
    "configurable":{"thread_id": str(uuid.uuid4())},
}

# Path("graph.png").write_bytes(graph.get_graph().draw_mermaid_png())
from langgraph.types import Command

STREAMING_NODES = ("chat_agent", "rag_agent", "code_agent")


def run(input_data):
    """Stream one run. Returns the interrupt prompt if the graph paused, else None."""
    interrupt_prompt = None
    printed_agent = None

    for mode, data in graph.stream(
        input_data,
        config=config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            token, metadata = data
            node = metadata.get("langgraph_node")
            if node in STREAMING_NODES and token.content:
                if printed_agent != node:
                    print(f"\n[{node}] ", end="", flush=True)
                    printed_agent = node
                print(token.content, end="", flush=True)

        elif mode == "updates" and "__interrupt__" in data:
            interrupt_prompt = data["__interrupt__"][0].value

    print()
    return interrupt_prompt


while True:
    user_message = input("_________\nEnter your message: ")
    if user_message.strip().lower() == "exit":
        break

    prompt = run({"messages": [{"role": "user", "content": user_message}]})

    while prompt:
        decision = input(f"\n{prompt}\n> ")
        prompt = run(Command(resume=decision))