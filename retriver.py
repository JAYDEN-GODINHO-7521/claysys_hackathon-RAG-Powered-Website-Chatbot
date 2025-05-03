from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def main(query):
    embeddings = OpenAIEmbeddings(check_embedding_ctx_length=False,  openai_api_key="sk-1234", base_url="http://localhost:8080/v1",model="text-embedding-nomic-embed-text-v1.5")

    persist_directory = "chroma_persist" 

    loaded_vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory) 

    retriever = loaded_vectordb.as_retriever(search_kwargs={"k": 3})

    # Prompt
    template = """answer using the text below.
    {context}

    Question: {question}
    Don't mention that you are referring to the text in your answer.
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(base_url="http://127.0.0.1:8080/v1",model="llama-3.2-1b-instruct", api_key="LM")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response=rag_chain.invoke(query)

    return response