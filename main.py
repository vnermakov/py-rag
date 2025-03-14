from flask import Flask, request, jsonify
from rag_system import RAGSystem

app = Flask(__name__)
rag_system = RAGSystem()

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_id = data.get('user_id')
    query_text = data.get('query')
    response = rag_system.handle_query(user_id, query_text)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
