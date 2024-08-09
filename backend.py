'''
just a demo
'''

from flask import Flask, request, jsonify
import os
from demo_api_mode import single_response, bot

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    chat_type = data.get('type')
    
    history = data.get('history', [])
    response = bot(history + [[user_message, None]], chat_type)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
