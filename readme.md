# Multi-Agent Chatbot with LangGraph

A small learning project that routes each user message to a specialized agent using [LangGraph](https://langchain-ai.github.io/langgraph/). Everything runs **locally** with [Ollama](https://ollama.com), so no API key is needed.

## Features

- **Intent classification**: a structured-output classifier sends each message to one of three agents (`chat`, `knowledge`, `code`).
- **Chat agent**: casual conversation.
- **RAG agent**: answers questions from a small knowledge base (in-memory vector store with Ollama embeddings).
- **Code agent**: writes code and saves it to `workspace/code.txt`.
- **Human-in-the-loop**: `interrupt()` asks the user to accept, reject or change the code request.
- **Conversation memory**: `InMemorySaver` checkpointer with a `thread_id`.
- **Streaming**: tokens are printed as the model generates them.

## Graph

```
START -> classifier -+-> chat_agent -----------------------> END
                     |
                     +-> rag_agent ------------------------> END
                     |
                     +-> accept_coding -+-> code_agent ----> END
                                        |
                                        +-> END (denied)
```

You can draw the real graph at any time:

```python
print(graph.get_graph().draw_mermaid())   # paste into https://mermaid.live
```

## Project Structure

```
someLangGraphProject/
├── more complex graph.py   # main script: state, nodes, graph, chat loop
├── rag_helper.py           # knowledge text + in-memory vector store
├── workspace/              # output folder for generated code (git-ignored)
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

## Requirements

- Python 3.12
- [Ollama](https://ollama.com) installed and running

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/IbrahimElhaw/<repository-name>.git
   cd <repository-name>
   ```

2. **Create and activate a virtual environment**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**

   ```bash
   python -m pip install -r requirements.txt
   ```

4. **Download the Ollama models**

   ```bash
   ollama pull gemma3:1b
   ollama pull nomic-embed-text
   ```

   `gemma3:1b` is the chat model (small, low power use) and `nomic-embed-text` creates the embeddings for the RAG agent.

## Usage

Make sure Ollama is running, then start the program:

```bash
python "more complex graph.py"
```

Type a message and press Enter. Type `exit` to quit.

Example prompts:

| Prompt | Routed to |
|---|---|
| `hi, can we talk` | chat agent |
| `What does Facebook say about selling animals?` | RAG agent |
| `write a python function that reverses a string` | code agent (asks for confirmation first) |

When the code agent is selected you are asked whether to continue. Answer `yes`, `no`, or type a revised request.

## Configuration

- **Model**: change `model="gemma3:1b"` in `init_chat_model(...)` to try another Ollama model (for example `qwen2.5:3b`).
- **Knowledge base**: edit `RAG_KNOWLEDGE` in `rag_helper.py`.
- **Output file**: generated code is saved to `workspace/code.txt` and overwritten on each request.

## Notes and Limitations

- `gemma3:1b` is very small. It can misclassify intents, ignore instructions, or hallucinate. It is meant for learning the graph structure, not for accurate answers.
- The vector store and the conversation memory are **in RAM only** and are lost when the program stops.
- Generated code is only saved, never executed.

## Tech Stack

Python, LangGraph, LangChain, Ollama, Pydantic

## License

Add a license of your choice (for example MIT).