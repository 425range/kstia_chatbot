from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. Load the document
loader = TextLoader("data/kstia.md", encoding="utf-8")
documents = loader.load()

# 2. Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)

docs = text_splitter.split_documents(documents)

print(f"Number of chunks: {len(docs)}");

# 3. Create embeddings for the chunks
embeddings = OpenAIEmbeddings()

# 4. Store the embeddings in a vector store
vectorstore = FAISS.from_documents(docs, embeddings)

# 5. Save the vector store to disk
vectorstore.save_local("faiss_index")

print("Vector store saved to disk.")