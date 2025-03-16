from typing import Dict, List
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import uuid

class RAGService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_stores: Dict[str, Chroma] = {}
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.llm = ChatOpenAI(temperature=0)

    def create_user_collection(self, user_id: str) -> str:
        collection_id = f"user_{user_id}_{str(uuid.uuid4())}"
        self.vector_stores[user_id] = Chroma(
            collection_name=collection_id,
            embedding_function=self.embeddings
        )
        return collection_id

    def add_documents(self, user_id: str, documents: List[str]) -> None:
        if user_id not in self.vector_stores:
            self.create_user_collection(user_id)
        
        docs = [Document(page_content=text) for text in documents]
        texts = self.text_splitter.split_documents(docs)
        self.vector_stores[user_id].add_documents(texts)

    def query(self, user_id: str, query: str) -> str:
        if user_id not in self.vector_stores:
            raise ValueError("User has no documents indexed")
            
        retriever = self.vector_stores[user_id].as_retriever()
        prompt = PromptTemplate.from_template(
            """Answer the following question based on the provided context:

            Context: {context}
            Question: {input}

            Answer:"""
        )
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        qa_chain = RunnablePassthrough() | retriever | document_chain
        
        return qa_chain.invoke(query)
