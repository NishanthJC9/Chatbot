import os
from flask import Flask, request, jsonify, render_template
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from flask_cors import CORS  # Import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Set your Google API Key (ensure it's set up properly in your environment)
os.environ["GOOGLE_API_KEY"] = "AIzaSyBdSjDXgh6i3ojwEskmigar4412yD-PjgM"

# Initialize Google Gemini client using Langchain's ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.8)

# Define the prompt template for movie recommendation (no memory)
demo_template = '''You should act as a simple chatbot to answer all the questions which is asked.
Human Input: {human_input}'''

# Create the prompt using the template
prompt = PromptTemplate(input_variables=['human_input'], template=demo_template)

# Route to render the HTML page
@app.route("/")
def index():
    return render_template("index.html")  # This will render the HTML file from the templates folder

# Route to handle chat requests
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()  # Get the JSON data sent in the POST request

    # Check if 'question' key is in the request data
    if "question" not in data:
        return jsonify({"error": "Missing 'question' in request data"}), 400

    user_input = data["question"]

    # Create the LLM chain without memory
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    try:
        # Get the response from the chain, passing in the human input
        response = chain.predict(human_input=user_input)

        # Return the response as JSON
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
