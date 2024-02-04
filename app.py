from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chatting import get_response 

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  

@app.route("/predict", methods=["POST"])  
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    answer = {"answer": response}
    return jsonify(answer)

if __name__ == "__main__":
    app.run(debug=True)
