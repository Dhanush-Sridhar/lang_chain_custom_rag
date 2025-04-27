

## Load the documents

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("DocSource/PED_Spezifikation_V2.3.0.pdf")

documents = loader.load()

print(documents)


## Split the documents into chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Splitting the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)

# Creating the embedding model for embedding our text data
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(model_name=model_name)

from langchain_chroma import Chroma

# Storing our data into vector database
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

