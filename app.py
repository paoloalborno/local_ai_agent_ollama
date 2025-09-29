"""
Reviews Q&A System using Ollama and LangChain.
"""

import sys
from typing import Optional
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import initialize_vector_store


class OllamaAgentQASystem:

    def __init__(self):
        self.llm: Optional[OllamaLLM] = None
        self.chain = None
        self.retriever = None
        self.vector_store_manager = None

        # Available models
        self.available_models = {
            "1": "qwen3:8b",
            "2": "llama3.2:latest"
        }

        # System prompt template
        self.template = """
            You are an expert assistant specialized in analyzing e-commerce product reviews.
            Based on the provided review texts, answer the user's question accurately and comprehensively.
            Review texts: {reviews_texts}
            User question: {question}
            Please provide a detailed and helpful response based on the review data.
        """

    @staticmethod
    def display_welcome_message():
        """Display welcome message and system information."""
        print("=" * 60)
        print("Ollama LLM Agent - Q&A System")
        print("=" * 60)
        print()

    def select_model(self) -> bool:
        """Allow user to select the AI model."""
        print("Available AI Models:")
        print("1. Qwen3:8b - Optimized multilingual model")
        print("2. Llama3.2:latest - Meta AI conversational model")
        print()

        while True:
            choice = input("Please select a model (1 or 2): ").strip()

            if choice in self.available_models:
                model_name = self.available_models[choice]
                try:
                    self.llm = OllamaLLM(model=model_name)
                    print(f"Successfully loaded model: {model_name}")
                    return True
                except Exception as e:
                    print(f"Error loading model {model_name}: {e}")
                    print("Please ensure Ollama is running and the model is installed.")
                    return False
            else:
                print("Invalid selection. Please choose 1 or 2.")

    def initialize_system(self) -> bool:
        """Initialize the complete system."""
        try:
            # Initialize vector store
            print("\nInitializing vector database...")
            self.vector_store_manager = initialize_vector_store()
            self.retriever = self.vector_store_manager.get_retriever()

            # Create the processing chain
            prompt = ChatPromptTemplate.from_template(self.template)
            self.chain = prompt | self.llm

            print("System initialization completed successfully.")
            return True

        except Exception as e:
            print(f"System initialization failed: {e}")
            return False

    def process_question(self, question: str) -> str:
        """Process a user question and return the answer."""
        try:
            # Retrieve relevant reviews
            docs = self.retriever.invoke(question)
            reviews_texts = "\n\n".join([doc.page_content for doc in docs])

            # Generate answer
            print("\nGenerating response with the LLM model, please wait...")
            result = self.chain.invoke({
                "reviews_texts": reviews_texts,
                "question": question
            })

            if result.count("</think>") == 1:
                result = result.split("</think>", 1)[1]

            return result.strip()

        except Exception as e:
            return f"Error processing question: {e}"

    @staticmethod
    def display_answer(answer: str):
        """Display the answer with proper formatting."""
        print("\nLLM Generated Response:")
        print("-" * 60)
        print(answer)
        print("-" * 60)
        print()

    def run_interactive_session(self):
        """Run the main interactive Q&A session."""
        print("\nStarting session...")
        print("You can ask questions about the product reviews.")
        print("Type 'quit', 'exit', or 'q' to end the session.")
        print()

        while True:
            try:
                print("-" * 60)
                question = input("Ask a question about the products: ").strip()

                # Check for exit commands
                if question.lower() in ['q', 'quit', 'exit']:
                    print("\nThank you for using the Product Reviews Q&A System!")
                    break

                # Skip empty questions
                if not question:
                    print("Please enter a valid question.")
                    continue

                # Process the question
                answer = self.process_question(question)
                self.display_answer(answer)

            except KeyboardInterrupt:
                print("\n\nSession interrupted by user.")
                print("Thank you for using the Product Reviews Q&A System!")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                print("Please try again or restart the system.")

    def start(self):
        self.display_welcome_message()

        if not self.select_model():
            print("Failed to load AI model. Exiting...")
            sys.exit(1)

        if not self.initialize_system():
            print("Failed to initialize system. Exiting...")
            sys.exit(1)

        self.run_interactive_session()


def main():
    qa_system = OllamaAgentQASystem()
    qa_system.start()

if __name__ == "__main__":
    main()
