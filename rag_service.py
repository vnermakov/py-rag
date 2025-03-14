from typing import Dict, List, Optional
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
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
        
        texts = self.text_splitter.split_documents(documents)
        self.vector_stores[user_id].add_documents(texts)

    def query(self, user_id: str, query: str) -> str:
        if user_id not in self.vector_stores:
            raise ValueError("User has no documents indexed")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_stores[user_id].as_retriever()
        )
        
        return qa_chain.run(query)
