import openai
from dotenv import load_dotenv
import os
#fast api
import httpx
from fastapi import FastAPI
#database
from fastapi import FastAPI, HTTPException, Depends
import models
from typing import Annotated
from database import engine, SessionLocal
from sqlalchemy.orm import Session
#open ai chat
from schemas.schema import UserBase, PostBase, BotBase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

app = FastAPI()

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY') #OpenAI API key

#dataabse utils
models.Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

#Docs https://platform.openai.com/docs/api-reference/chat/streaming


async def generate_llmanswer(input, seed_number, user_query):
    url = "https://blog.langchain.dev/langgraph/"
    vector_store = get_vector_from_url(url) #PARAM#
    
    async with httpx.AsyncClient() as client:
        url = f"http://127.0.0.1:8000/chatbot/{seed_number}"
        response = await client.get(url)
        response.raise_for_status()  # Raise an exception if the request fails
        chathistory = response.json()  # Assuming the endpoint returns JSON
    print(chathistory)    

    #print(chathistory)
    # Dynamically construct messages with appropriate structures
    langmessages = []
    for message_dict in chathistory:
        message_type = HumanMessage if message_dict["role"] == "user" else AIMessage
        message = message_type(content=message_dict["content"])
        langmessages.append(message) #PARAM# if langmessages is empty because there is no chathistory response...
    print(langmessages)

    #Según el chat-history y user_query va a llamar a todos los vectores relevantes (un vector por cada document-chunk)
    retrieved_documents = get_context_retriever_chain(vector_store)
    """
    .invoke({    
        "chat_history": langmessages,
        "input": user_query
    })
    """
    response = get_conversational_rag_chain(retrieved_documents).invoke({
            "chat_history": langmessages,
            "input": user_query
    })
    print(response['answer'])
    #print(response['context'])

    return response['answer']

       


#want to be persistant
def get_vector_from_url(url):
    #get the textin document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

#This function essentially creates a mechanism for the chatbot to 
#actively search and retrieve information from the website (source doc chunks), making the conversation more dynamic and informative.
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    # convert vector store into a retriever object. This allows searching the vectorstore for relevant information
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    #Combines the LLM, retriever, and prompt into a single chain
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain

#this function get an answer using the source-doc-chunks and chat-history (based on your query)    
def get_conversational_rag_chain(retriever_chain): 
    
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
      ("system", "Answer the user's questions. Give me a concise answer since my target byte size is maximum of 450 bytes. Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)






#usando el metodo openai.ChatCompletion, donde no se necesita pasar un web_url como fuente para que LLM pueda responderte
async def generate_llmanswer_sin_URLSource(input, seed_number, user_query):
    async with httpx.AsyncClient() as client:
        url = f"http://127.0.0.1:8000/chatbot/{seed_number}"
        response = await client.get(url)
        response.raise_for_status()  # Raise an exception if the request fails
        chathistory = response.json()  # Assuming the endpoint returns JSON

    # Convert custom messages to API-expected format (openai.ChatCompletion)
    api_messages = []
    for message_dict in chathistory:
        api_message = {"role": message_dict["role"], "content": message_dict["content"]}
        api_messages.append(api_message)
    #print(api_messages)    

    messagesInstruction = {"role": "user", "content": """Please help me with this question. Give me a concise answer since my target byte size is maximum of 450 bytes' \n"""}
    
    api_messages.append(messagesInstruction)

    api_messages.append({"role": "user", "content": f"{input}"})
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=api_messages
    )
    reply = completion.choices[0].message.content
    return reply

