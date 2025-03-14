# py-rag

# Multi-user RAG System

## Setup

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

2. Run the server:
    ```sh
    python main.py
    ```

## Usage

Send a POST request to `http://127.0.0.1:5000/query` with JSON body:
```json
{
    "user_id": "user1",
    "query": "Your query here"
}
```