"""
Vector database management for product reviews using ChromaDB and Ollama embeddings.
"""
import os
import pandas as pd
from typing import List
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


class ReviewsVectorStore:
    """Manages the vector store for product reviews."""

    def __init__(self,
                 csv_file_path: str = "reviews.csv",
                 db_location: str = "./chrome_langchain_db",
                 embedding_model: str = "mxbai-embed-large",
                 collection_name: str = "restaurant_reviews"):
        """
        Initialize the vector store manager.

        Args:
            csv_file_path: Path to the CSV file containing reviews
            db_location: Directory for ChromaDB storage
            embedding_model: Ollama embedding model to use
            collection_name: Name for the ChromaDB collection
        """
        self.csv_file_path = csv_file_path
        self.db_location = db_location
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)

        # Initialize vector store
        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=db_location,
            embedding_function=self.embeddings
        )

    def should_recreate_database(self) -> bool:
        if not os.path.exists(self.db_location):
            print("Database not found. Creating new database...")
            return True

        user_input = input("Recreate internal database (loading reviews.csv)? 'y' or 'n': ")
        return user_input.lower() == 'y'

    def load_reviews_from_csv(self) -> pd.DataFrame:
        try:
            if not os.path.exists(self.csv_file_path):
                raise FileNotFoundError(f"CSV file not found: {self.csv_file_path}")

            df = pd.read_csv(self.csv_file_path)

            # Validate required columns
            required_columns = ["Title", "Review", "Rating", "Date"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            print(f"Loaded {len(df)} reviews from {self.csv_file_path}")
            return df

        except Exception as e:
            print(f"Error loading CSV file: {e}")
            raise

    def create_documents_from_dataframe(self, df: pd.DataFrame) -> tuple[List[Document], List[str]]:
        """Convert dataframe to LangChain documents."""
        documents = []
        ids = []

        print("Creating documents...")
        for i, row in df.iterrows():
            # Combine title and review for better context
            content = f"{row['Title']} {row['Review']}" if pd.notna(row['Title']) else row['Review']

            doc = Document(
                page_content=content,
                metadata={
                    "rating": row["Rating"],
                    "date": row["Date"],
                    "title": row["Title"] if pd.notna(row["Title"]) else "No Title"
                }
            )
            documents.append(doc)
            ids.append(str(i))

        return documents, ids

    def initialize_database(self) -> None:
        """Initialize or recreate the vector database."""
        if self.should_recreate_database():
            # Load and process data
            df = self.load_reviews_from_csv()
            documents, ids = self.create_documents_from_dataframe(df)

            # Add documents to vector store
            print("Adding documents to vector store...")
            self.vector_store.add_documents(documents=documents, ids=ids)
            print(f"Database created with {len(documents)} documents")
        else:
            # Check existing database
            collection = self.vector_store.get()
            print(f"Database found with {len(collection['ids'])} documents")

    def get_retriever(self, k: int = 50):
        """Get a retriever for the vector store."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# Initialize the vector store and retriever
def initialize_vector_store() -> ReviewsVectorStore:
    """Factory function to initialize and return vector store."""
    v_store_manager = ReviewsVectorStore()
    v_store_manager.initialize_database()
    return v_store_manager