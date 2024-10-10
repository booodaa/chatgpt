import os
import json
import logging
import uuid
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, session
from flask_cors import CORS
from openai import OpenAI
import io
import docx
import fitz
import base64

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config['SECRET_KEY'] = '369'  # Change this to a secure random string

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

api_key = 'sk-Ws0RC7y2pZ5E8dSlz9LArCjsGQjFMpi__g4ujQnzL5T3BlbkFJwZEm2zMLzU0Hv9FV9kGWPMXJY2hipYuLp2VrzKpCIA'
client = OpenAI(api_key=api_key)

MAX_MESSAGES = 10
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'doc', 'png', 'jpg', 'jpeg'}

# In-memory storage for conversations and remaining messages
conversations = {}
remaining_messages_store = {}

# Define the system prompt variable
system_prompt = "You are an intelligent assistant specialized in helping users learn from and analyze provided data. You can generate insightful questions, answer questions, or provide insights based strictly on the text and visual content extracted from the files uploaded, offering short and concise responses. Your ability to generate questions helps users explore the data more effectively."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def encode_image(file_content):
    return base64.b64encode(file_content).decode('utf-8')

def extract_text_from_file(file_content, filename):
    _, file_extension = os.path.splitext(filename)
    file_extension = file_extension.lower()

    try:
        if file_extension == '.pdf':
            return extract_text_from_pdf(file_content)
        elif file_extension == '.docx':
            return extract_text_from_docx(file_content)
        elif file_extension == '.txt':
            return file_content.decode('utf-8')
        else:
            return "Unsupported file format"
    except Exception as e:
        logger.error(f"Error extracting text from file: {e}")
        return "Error extracting text from file"

def extract_text_from_pdf(file_content):
    text = ""
    pdf = fitz.open(stream=file_content, filetype='pdf')
    for page in pdf:
        text += page.get_text()
    return text

def extract_text_from_docx(file_content):
    file_stream = io.BytesIO(file_content)
    doc = docx.Document(file_stream)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def get_or_create_conversation(session_id):
    if session_id not in conversations:
        conversations[session_id] = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
    return conversations[session_id]

@app.route('/')
def index():
    logger.info("Serving index.html")
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), 'index.html')


@app.route('/api/create_session', methods=['GET'])
def create_session():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    conversations[session_id] = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    remaining_messages_store[session_id] = MAX_MESSAGES
    return jsonify({"session_id": session_id, "remaining_messages": MAX_MESSAGES}), 200

@app.route('/api/query', methods=['POST'])
def query():
    session_id = session.get('session_id')
    if not session_id:
        # Create a new session if none exists
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
        conversations[session_id] = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        remaining_messages_store[session_id] = MAX_MESSAGES

    if remaining_messages_store.get(session_id, 0) <= 0:
        return jsonify({"error": "No remaining messages. Please reset the conversation."}), 403

    user_query = request.form.get('query', '')
    logger.info(f"Query from session {session_id}: {user_query}")

    conversation = get_or_create_conversation(session_id)

    # Process user query and any uploaded files
    user_message_content = [{"type": "text", "text": user_query}]
    
    files = request.files.getlist('file')
    if files:
        for file in files[:5]:  # Limit to 5 files
            if file and allowed_file(file.filename):
                file_content = file.read()
                mime_type = file.mimetype

                if mime_type and mime_type.startswith('image/'):
                    base64_image = encode_image(file_content)
                    user_message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    })
                else:
                    extracted_text = extract_text_from_file(file_content, file.filename)
                    user_message_content.append({
                        "type": "text",
                        "text": f"Extracted text from file ({file.filename}):\n{extracted_text}"
                    })

    # Append user message to conversation
    conversation.append({
        "role": "user",
        "content": user_message_content
    })

    # Ensure conversation doesn't exceed max messages
    if len(conversation) > MAX_MESSAGES:
        conversation = conversation[-MAX_MESSAGES:]
        conversations[session_id] = conversation

    def generate(conversation, session_id):
        try:
            logger.info(f"Starting OpenAI stream for session {session_id}")
            stream = client.chat.completions.create(
                model='chatgpt-4o-latest',
                messages=conversation,
                temperature=0.7,
                stream=True
            )

            assistant_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    assistant_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"

            # Append assistant response to conversation
            conversation.append({"role": "assistant", "content": assistant_response})

            # Ensure conversation doesn't exceed max messages
            if len(conversation) > MAX_MESSAGES:
                conversation = conversation[-MAX_MESSAGES:]
                conversations[session_id] = conversation

            # Decrement remaining messages
            remaining_messages_store[session_id] = max(0, remaining_messages_store.get(session_id, MAX_MESSAGES) - 1)

            # Send remaining message count
            remaining_messages = remaining_messages_store[session_id]
            yield f"data: {json.dumps({'remaining_messages': remaining_messages})}\n\n"

        except Exception as e:
            logger.error(f"Error in OpenAI stream for session {session_id}: {str(e)}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate(conversation, session_id)), content_type='text/event-stream')

@app.route('/api/reset_conversation', methods=['POST'])
def reset_conversation():
    old_session_id = session.get('session_id')
    if old_session_id:
        # Delete the old session data
        conversations.pop(old_session_id, None)
        remaining_messages_store.pop(old_session_id, None)
        session.pop('session_id', None)

    # Create a new session
    new_session_id = str(uuid.uuid4())
    session['session_id'] = new_session_id
    conversations[new_session_id] = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    remaining_messages_store[new_session_id] = MAX_MESSAGES

    return jsonify({
        "message": "Conversation reset",
        "session_id": new_session_id,
        "remaining_messages": MAX_MESSAGES
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
