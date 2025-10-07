"""
Vector database management for product reviews using ChromaDB and Ollama embeddings.
"""
import logging
import os
import pandas as pd
import chromadb
from langchain_chroma import Chroma
from typing import List, Tuple
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def _df_to_documents(df: pd.DataFrame) -> Tuple[List[Document], List[str]]:
    #transforms a dataframe to a list of documents and ids
    documents = []
    ids = []
    for i,row in df.iterrows():
        content = f"{row['Title']} - {row['Review']}" if pd.notna(row['Title']) else row['Review']
        doc = Document(
            page_content=content,
            metadata = {
                "rating": float(row["Rating"]) if not pd.isna(row["Rating"]) else None,
                "date": str(row["Date"]),
                "title": row["Title"] if pd.notna(row["Title"]) else "No Title"
            }
        )
        documents.append(doc)
        ids.append(str(i))
    return documents, ids

class ReviewsVectorStore:

    def __init__(self, csv_file_path: str = "reviews.csv", db_location: str = "./chroma_db", embedding_model: str = "mxbai-embed-large", collection_name: str = "gaming_reviews"):
        self.csv_file_path = csv_file_path
        self.db_location = db_location
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        self.vector_store = Chroma(
            client=chromadb.PersistentClient(path=db_location),
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    def load_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")
        else:
            df = pd.read_csv(self.csv_file_path)
            required_columns = ["Title","Date","Rating","Review"]
            missing_columns = [c for c in required_columns if c not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            return df

    def init_database(self, auto_recreate: bool = False) -> None:
        if not os.path.exists(self.db_location):
            raise FileNotFoundError(f"Database directory not found: {self.db_location}")
        if auto_recreate:
            try:
                self.vector_store.delete_collection()
                documents, ids = _df_to_documents(self.load_csv())
                self.vector_store.add_documents(documents=documents, ids=ids) # vector store is a ChromaDB object used to store documents and their embeddings
            except Exception as e:
                logging.error(f"Error recreating database: {e}")
                raise e
        else:
            try:
                columns = self.vector_store.get()
                size = len(columns.get("ids", []))
                logging.info(f"Database size: {size}")
                if size == 0:
                    logging.info("Database is empty. Loading data from CSV and adding to ChromaDB.")
                    documents, ids = _df_to_documents(self.load_csv()) #_ before the name of the function means that it is private
                    self.vector_store.add_documents(documents=documents, ids=ids)
            except Exception as e:
                logging.error(f"Error initializing database: {e}")
                raise e

    def get_number_of_vectors(self):
        return len(self.vector_store.get().get("ids", []))

    def get_retriever(self, k: int = 10):
        k = max(1, min(50, int(k)))
        return self.vector_store.as_retriever(search_kwargs={"k": k})