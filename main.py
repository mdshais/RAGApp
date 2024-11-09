import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint

# Import library
from langchain_community.document_loaders import PyPDFLoader
# Import the recursive character splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


load_dotenv()

key=os.environ['HUGGINGFACE_API_KEY']



# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"

repo_id = "meta-llama/Llama-3.2-1B"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_length=128,
    temperature=0.5,
    huggingfacehub_api_token=key,
)

# Create a document loader for rag_vs_fine_tuning.pdf
loader = PyPDFLoader("resume.pdf")

# Load the document
data = loader.load()
# print(data)


chunk_size = 150
chunk_overlap = 30

# Create an instance of the splitter class
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap)

# Split the document and print the chunks
docs = splitter.split_documents(data)
# print(docs)


# Embed the documents in a persistent Chroma vector database
model_name = "sentence-transformers/all-mpnet-base-v2"
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_function = HuggingFaceEmbeddings(model_name=model_name)

vectorstore = Chroma.from_documents(
    docs,
    embedding=embedding_function,
    persist_directory=os.getcwd()
)

# Configure the vector store as a retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# Add placeholders to the message string
message = """
Answer the following question using the context provided:

Context:
{context}

Question:
{question}

Answer:
"""

# Create a chat prompt template from the message string
prompt_template = ChatPromptTemplate.from_messages([("human", message)])

# Create a chain to link retriever, prompt_template, and llm
rag_chain = ({"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm)

# Invoke the chain
response = rag_chain.invoke("what is the age of the person?")
print(response)

