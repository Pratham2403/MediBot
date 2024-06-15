from flask import Flask, request, make_response, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
import io

# Load environment variables
load_dotenv()

# Constants
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "mixtral-8x7b-32768"
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a medical chatbot that provides answers based on the context and finds relavance in questions using both context and conversation history.
Your responses are medically relevant, however you can have general conversations as well as medically relavnt conversations.

Context:
{context}

Conversation History:
{history}

---

Make sure to check both Context and Conversation History to maximize output efficiency.

Important Notes:
- Your name is MediBot and you only say it when asked so in the question.
- When thanked or bidding farewell, do not refer to the conotext, your response should only be 'Welcome' and 'Bye', nothing else.
- When having a general conversation, do not refer to the context at all, and your response should be extremely small and concise.
- You should first distinguish whether the question is related to medical field or a general conversation, then respond correspondingly.
- Your response should align well with the question, and should not contain any extra information that may not be required.

Questions based on the context:
- {question}
"""

# In-memory conversation history storage
conversation_history = []

app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/detect', methods=['POST'])
def detect():
    if request.content_type != 'application/json':
        return 'Unsupported Media Type', 415

    data = request.get_json()
    query_text = data.get('input', '')

    try:
        # Sanity checks for environment variables
        if not COHERE_API_KEY:
            raise ValueError("COHERE_API_KEY is not set")
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set")

        # Prepare the DB
        print("Initializing Cohere Embeddings...")
        embedding_function = CohereEmbeddings()

        print("Connecting to Chroma database...")
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB
        print(f"Searching the database for query: {query_text}")
        results = db.similarity_search_with_score(query_text, k=3)

        # Construct the context text
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        # Construct the conversation history text
        history_text = "\n".join(conversation_history)

        # Create the prompt
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, history=history_text, question=query_text)

        # Get the response from the model
        print("Invoking the ChatGroq model...")
        model = ChatGroq(api_key=GROQ_API_KEY, model=MODEL)
        response = model.invoke(prompt)
        
        # Extract the content and sources
        response_text = response.content

        # Update the conversation history
        conversation_history.append(f"Client: {query_text}")
        conversation_history.append(f"Response: {response_text}")

        # Format the final response
        formatted_response = f"{response_text}"

        return formatted_response
    except Exception as e:
        # Log the error details
        print(f"Error during detection: {e}")
        return jsonify({'error': 'Internal Server Error', 'message': str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset():
    global conversation_history
    conversation_history = []
    return "Conversation history reset."

@app.route('/download', methods=['GET'])
def download():
    # Create a formatted text from the conversation history
    formatted_history = "\n".join(conversation_history)
    
    # Create a text file in memory
    buffer = io.StringIO()
    buffer.write("Conversation History:\n")
    buffer.write("-----------------------------\n")
    buffer.write(formatted_history)
    buffer.seek(0)
    
    # Create a response with the text file
    response = make_response(buffer.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=conversation_history.txt"
    response.headers["Content-Type"] = "text/plain"
    
    return response


if __name__ == '__main__':
    app.run(port=5000)
