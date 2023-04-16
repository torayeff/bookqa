import os

from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
if not os.path.exists("book_index"):
    print("[Building index]")
    # load book
    loader = PyMuPDFLoader("book.pdf")
    documents = loader.load()

    # split and create documents
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(documents)

    # create embeddings and store
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("book_index")

# load book
vectorstore = FAISS.load_local("book_index", embeddings)

# intialize
qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0), vectorstore.as_retriever()
)
query = ""
chat_history = []

# start chat
while query != "exit":
    query = input("Human: ")
    result = qa({"question": query, "chat_history": chat_history})
    chat_history.append((query, result["answer"]))

    print(f"AI: {result['answer']}")
