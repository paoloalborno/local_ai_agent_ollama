import csv

import pandas as pd
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

reviews_texts = []

#df = pd.read_csv("reviews.csv")
#for row in df.iterrows():
#    reviews_texts.append(row["Title"] + " " + row["Review"])

model_choice = input("Choose model : '1' = qwen3:8b '2' = llama3.2:latest : ")
if model_choice == "1":
    llm = OllamaLLM(model="qwen3:8b")
elif model_choice == "2":
    llm = OllamaLLM(model="llama3.2:latest")
else:
    print("Invalid model")
    exit()
print(f"Model selected is {llm.model}")

template = """
    You are an expert in answering questions about the ecommerce product reviews:
    Here are the reviews texts: {reviews_texts}
    Here is the user question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm
while True:
    print("\n\n--------------------------------------------------------")
    question = input("Ask a question about the products (q to quit): ")
    if question == "q":
        break
    #select relevant reviews by vector
    docs = retriever.invoke(question)
    reviews_texts = "\n\n".join([d.page_content for d in docs])

    result = chain.invoke({"reviews_texts": reviews_texts, "question": question})
    if result.count("</think>") == 1:
        result = result.split("</think>", 1)[1]
    print(result)