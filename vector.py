# database locally on our computer - chroma

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os
import pandas as pd


df = pd.read_csv("reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    print("📥 Creating documents")
    documents = []
    ids = []
    for i, row in df.iterrows():
        doc = Document(
            page_content=row["Title"] + " " + row["Review"],
            metadata={"rating": row["Rating"], "date": row["Date"]}
        )
        ids.append(str(i))
        documents.append(doc)

    vector_store.add_documents(documents=documents, ids=ids)
    print("✅ Database created")
else:
    print("✅ Database found")

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

