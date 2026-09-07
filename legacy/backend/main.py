import re
from youtube_transcript_api import YouTubeTranscriptApi
import openai
import anthropic
import google.generativeai as genai
from transformers import pipeline
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import logging

# Load environment variables from .env file
load_dotenv()

# Set API keys from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY') 
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

required_env_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
for var in required_env_vars:
    if not os.getenv(var):
        logging.error(f"Missing required environment variable: {var}")
        raise EnvironmentError(f"Missing required environment variable: {var}")

def get_youtube_transcript(video_url):
    print(f"Attempting to fetch transcript for URL: {video_url}")
    video_id = extract_video_id(video_url)
    print(f"Extracted video ID: {video_id}")
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([entry['text'] for entry in transcript])
        print(f"Successfully fetched transcript. Length: {len(full_transcript)}")
        
        # Save transcript with a distinct name
        save_transcript(video_id, full_transcript)
        
        return full_transcript
    except Exception as e:
        print(f"Error fetching transcript: {str(e)}")
        raise ValueError(f"Error fetching transcript: {str(e)}")

def save_transcript(video_id, transcript):
    # Create transcripts directory if it doesn't exist
    os.makedirs('transcripts', exist_ok=True)
    
    # Save transcript with video ID as filename
    filename = f"transcripts/{video_id}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(transcript)
    print(f"Transcript saved as {filename}")

def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        str: The video ID.

    Raises:
        ValueError: If the URL is invalid.
    """
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:embed\/|v\/|youtu.be\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Could not extract video ID from URL")

def summarize_transcript(transcript, model="gpt-4"):
    """
    Summarizes a YouTube video transcript using the specified model.

    Args:
        transcript (str): The transcript of the video.
        model (str): The model to use for summarization.

    Returns:
        str: The summary of the transcript.

    Raises:
        ValueError: If the transcript is empty or the model is unsupported.
    """
    if not transcript:
        raise ValueError("Transcript is empty or None")
    
    if model.startswith("gpt"):
        return summarize_with_gpt(transcript, model)
    elif model == "claude":
        return summarize_with_claude(transcript)
    elif model == "gemini":
        return summarize_with_gemini(transcript)
    elif model == "llama":
        return summarize_with_llama(transcript)
    else:
        raise ValueError(f"Unsupported model: {model}")

# Initialize OpenAI client
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def summarize_with_gpt(transcript, model):
    """
    Summarizes a transcript using the GPT model.

    Args:
        transcript (str): The transcript of the video.
        model (str): The GPT model to use.

    Returns:
        str: The summary of the transcript.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes YouTube video transcripts."},
            {"role": "user", "content": f"Please summarize the following transcript and provide bullet-pointed highlights:\n\n{transcript}"}
        ]
    )
    return response.choices[0].message.content

def summarize_with_claude(transcript):
    """
    Summarizes a transcript using the Claude model.

    Args:
        transcript (str): The transcript of the video.

    Returns:
        str: The summary of the transcript.
    """
    client = anthropic.Client(api_key=anthropic_api_key)
    response = client.completion(
        prompt=f"Human: Please summarize the following transcript and provide bullet-pointed highlights:\n\n{transcript}\n\nAssistant:",
        model="claude-2",
        max_tokens_to_sample=1000
    )
    return response.completion

def summarize_with_gemini(transcript):
    """
    Summarizes a transcript using the Gemini model.

    Args:
        transcript (str): The transcript of the video.

    Returns:
        str: The summary of the transcript.
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Please summarize the following transcript and provide bullet-pointed highlights:\n\n{transcript}")
    return response.text

def summarize_with_llama(transcript):
    """
    Summarizes a transcript using the LLaMA model.

    Args:
        transcript (str): The transcript of the video.

    Returns:
        str: The summary of the transcript.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(transcript, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def answer_question(transcript, question, model="gpt-4"):
    """
    Answers a question based on a YouTube video transcript using the specified model.

    Args:
        transcript (str): The transcript of the video.
        question (str): The question to answer.
        model (str): The model to use for answering the question.

    Returns:
        str: The answer to the question.

    Raises:
        ValueError: If the transcript or question is empty, or the model is unsupported.
    """
    if not transcript or not question:
        raise ValueError("Transcript or question is empty or None")
    
    if model.startswith("gpt"):
        return answer_with_gpt(transcript, question, model)
    elif model == "claude":
        return answer_with_claude(transcript, question)
    elif model == "gemini":
        return answer_with_gemini(transcript, question)
    elif model == "llama":
        return answer_with_llama(transcript, question)
    else:
        raise ValueError(f"Unsupported model: {model}")

def answer_with_gpt(transcript, question, model):
    """
    Answers a question using the GPT model.

    Args:
        transcript (str): The transcript of the video.
        question (str): The question to answer.
        model (str): The GPT model to use.

    Returns:
        str: The answer to the question.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions about YouTube video transcripts."},
            {"role": "user", "content": f"Based on the following transcript, please answer the question:\n\nTranscript: {transcript}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message.content

def answer_with_claude(transcript, question):
    """
    Answers a question using the Claude model.

    Args:
        transcript (str): The transcript of the video.
        question (str): The question to answer.

    Returns:
        str: The answer to the question.
    """
    client = anthropic.Client(api_key=anthropic_api_key)
    response = client.completion(
        prompt=f"Human: Based on the following transcript, please answer the question:\n\nTranscript: {transcript}\n\nQuestion: {question}\n\nAssistant:",
        model="claude-2",
        max_tokens_to_sample=1000
    )
    return response.completion

def answer_with_gemini(transcript, question):
    """
    Answers a question using the Gemini model.

    Args:
        transcript (str): The transcript of the video.
        question (str): The question to answer.

    Returns:
        str: The answer to the question.
    """
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f"Based on the following transcript, please answer the question:\n\nTranscript: {transcript}\n\nQuestion: {question}")
    return response.text

def answer_with_llama(transcript, question):
    """
    Answers a question using the LLaMA model.

    Args:
        transcript (str): The transcript of the video.
        question (str): The question to answer.

    Returns:
        str: The answer to the question.
    """
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=transcript)
    return result['answer']

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://youtube-app-x18r.onrender.com", "http://localhost:3000"]}})

@app.route('/api/transcript', methods=['POST'])
def get_transcript():
    print("Received request for transcript")  # Debug log
    data = request.json
    print(f"Video URL: {data['video_url']}")  # Debug log
    try:
        transcript = get_youtube_transcript(data['video_url'])
        print("Transcript fetched successfully")  # Debug log
        return jsonify({'transcript': transcript})
    except Exception as e:
        print(f"Error: {str(e)}")  # Debug log
        return jsonify({'error': str(e)}), 500

def generate_summary(transcript, model):
    if model.startswith("gpt"):
        return summarize_with_gpt(transcript, model)
    elif model == "claude":
        return summarize_with_claude(transcript)
    elif model == "gemini":
        return summarize_with_gemini(transcript)
    elif model == "llama":
        return summarize_with_llama(transcript)
    else:
        raise ValueError(f"Unsupported model: {model}")

@app.route('/api/summarize', methods=['POST'])
def summarize():
    data = request.json
    transcript = data['transcript']
    model = data.get('model', 'gpt-4')
    summary = generate_summary(transcript, model)
    return jsonify({'summary': summary})

@app.route('/api/answer', methods=['POST'])
def answer():
    data = request.json
    transcript = data['transcript']
    question = data['question']
    model = data.get('model', 'gpt-4')
    answer = answer_question(transcript, question, model)
    return jsonify({'answer': answer})

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"An error occurred: {str(e)}")
    return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test():
    """
    Test endpoint to verify the Flask app is running.

    Returns:
        JSON response with a success message.
    """
    return jsonify({'message': 'Test successful'})

@app.route('/test-form', methods=['GET', 'POST'])
def test_form():
    if request.method == 'POST':
        video_url = request.form['video_url']
        try:
            transcript = get_youtube_transcript(video_url)
            return jsonify({'transcript': transcript})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template_string('''
        <form method="post">
            <input type="text" name="video_url" placeholder="YouTube URL">
            <input type="submit" value="Get Transcript">
        </form>
    ''')

@app.route('/')
def home():
    return "Welcome to the YouTube Transcript Analyzer API!"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)