import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.llms import OpenAI

OPEN_AI_LLM_MODEL = 'gpt-3.5-turbo-0301'
llm_model = OPEN_AI_LLM_MODEL

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

from langchain.indexes import VectorstoreIndexCreator

#This is an example of RetrievalQA chain
#In this we make use of a document to answer questions more accurately, with the help of a document.
#This document acts as a source of Truth/Fact.
#LLMs have limitations on the number of tokens that can use to give the answer, hence we make use of vectore stores and embeddings

# What is embeddings?
# Embeddings create numerical representation of pieces of text
# This numerical representation captures the symantic meaning of the pieces of text
# Text with similar content will have similar meaning

# What is vector Database?
# Store the vector representation that we generated in the previous steps
# The document is first breakdown into smaller chunks. We then create embeddings for each of these chunks and store them in the vector database
# The above happens when we create the index.
# During runtime we pass this index and the most similar embeddings are returned


# When a query comes in we first create an embedding for that query
# We compare this to all the vector in vector database and return the most similar  

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

llm_replacement_model = OpenAI(temperature=0, model=llm_model)

response = index.query(query, llm = llm_replacement_model)



#Lets execute the above step by step to understand it better
from langchain.document_loaders import CSVLoader
loader = CSVLoader(file_path=file)

docs = loader.load()

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

#Example of adding a custom embedding
embed = embeddings.embed_query("Hi my name is Harrison")

#Created a vector store of the documents loaded
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)


query = "Please suggest a shirt with sunblocking"

#This returns the similar documents based on the query
docs = db.similarity_search(query)


retriever = db.as_retriever()
llm = ChatOpenAI(temperature = 0.0, model=llm_model)

qdocs = "".join([docs[i].page_content for i in range(len(docs))])


#Given all the content of the document to the LLM, to answer the query
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

#Below you are using the RetrievalQA chain and the chain_type stuff means that you are passing all the content of the documents.
#This works here fine because the documents are small but there are other chain_type 
#E.g: ,map_reduce, refine, map_rerank
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)