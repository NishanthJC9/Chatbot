import os
from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Set your Google API Key (ensure it's set up properly in your environment)
os.environ["GOOGLE_API_KEY"] = "AIzaSyC14gcwPiQaxHlnTahD3QI9v_wYmfaDH44"

# Initialize Google Gemini client using Langchain's ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.1)

# Route to handle chat requests
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()  # Get the JSON data sent in the POST request

    # Check if 'question' key is in the request data
    if "question" not in data:
        return jsonify({"error": "Missing 'question' in request data"}), 400

    user_input = data["question"]

    # Set up the prompt template
    prompt_template = PromptTemplate(input_variables=["question"], template="{question}")

    # Create the LLM chain
    chain = LLMChain(llm=llm, prompt=prompt_template)

    try:
        # Get the response from the chain
        response = chain.run({"question": user_input})

        # Return the response as JSON
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
