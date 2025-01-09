import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
from phi.model.groq import Groq
from phi.embedder.ollama import OllamaEmbedder
#from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] =  os.getenv("GROQ_API_KEY")

db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

knowledgeBase = PDFUrlKnowledgeBase(
    
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url,embedder=OllamaEmbedder(model="llama3.2", dimensions=4096)),#OllamaEmbedder()),SentenceTransformerEmbedder()
    #model=Groq(id="llama-3.1-70b-versatile",api_key=os.getenv("GROQ_API_KEY")),dimensions=4096
)

knowledgeBase.load()

print("KNOWLEDGE BASE LOADED!!!!!")
storage = PgAssistantStorage(table_name="pdf_assistant",db_url=db_url)

def pdf_assistant(new: bool=False, user: str="user"):
    run_id: Optional[str] = None

    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids)>0:
            run_id=existing_run_ids[0]
    
    assistant = Assistant(
        llm=Groq(id="llama-3.1-8b-instant",api_key=os.getenv("GROQ_API_KEY")),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledgeBase,
        storage=storage,
        #show tool calls in the response
        show_tool_calls=True,
        #Enable the assistant to search the knowledge
        search_knowledge=True,
        #Enable the assistant to read the chat history
        read_chat_history=True,
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run:{run_id}\n")
    else:
        print(f"Continuing Run:{run_id}\n")

    assistant.cli_app(markdown=True)

if __name__ == "__main__":
    typer.run(pdf_assistant)