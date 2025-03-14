from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List
from rag_service import RAGService

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
rag_service = RAGService()

class DocumentRequest(BaseModel):
    documents: List[str]

class QueryRequest(BaseModel):
    query: str

@app.post("/documents")
async def add_documents(
    request: DocumentRequest,
    user_id: str = Depends(oauth2_scheme)
):
    try:
        rag_service.add_documents(user_id, request.documents)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(
    request: QueryRequest,
    user_id: str = Depends(oauth2_scheme)
):
    try:
        response = rag_service.query(user_id, request.query)
        return {"response": response}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
