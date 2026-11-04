from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

RAG_KNOWLEDGE = (
    "Facebook, owned by Meta, has specific policies to protect animals "
    "and prevent animal exploitation. Its rules prohibit the sale of endangered species, "
    "wildlife products and animal parts, as well as content promoting poaching or animal "
    "cruelty. The platform generally prohibits private individuals from selling live animals, "
    "although certain exceptions apply to legitimate businesses, breeders and animal shelters. "
    "Posts showing graphic animal abuse, suffering or violence may also be removed. "
    "Users can report content that violates these rules, and Facebook may remove posts or take action "
    "against responsible accounts."
)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = InMemoryVectorStore(embeddings)

documents = [
    Document(page_content=RAG_KNOWLEDGE)
]

vector_store.add_documents(documents)