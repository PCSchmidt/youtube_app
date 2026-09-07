# YouTube Transcript Analyzer

This project is a web application that allows users to analyze YouTube video transcripts. It provides features such as transcript retrieval, summarization, and question answering based on the video content.

## Features

- Fetch transcripts from YouTube videos
- Summarize video transcripts using various AI models
- Answer questions based on video transcripts
- Support for multiple AI models (GPT, Claude, Gemini, LLaMA)

## Technologies Used

### Backend
- Python
- Flask
- OpenAI API
- Anthropic API
- Google Generative AI
- Hugging Face Transformers
- YouTube Transcript API

### Frontend
- React
- Material-UI
- Axios

## Project Structure

```
youtube_app/
├── backend/
│   ├── main.py
│   └── transcripts/
├── frontend/
│   ├── public/
│   ├── src/
│   └── package.json
├── .env
├── .gitignore
├── package.json
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/youtube_app.git
   cd youtube_app
   ```

2. Set up the backend:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
   pip install -r requirements.txt
   ```
   Note: This project requires Python 3.7 or higher.

3. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```
   You'll need to obtain your own API keys from OpenAI, Anthropic, and Google to use their respective services.

4. Set up the frontend:
   ```
   cd frontend
   npm install
   ```

## Running the Application

1. Start the backend server:
   ```
   cd backend
   python main.py
   ```
   The backend will run on `http://localhost:5000`.

2. Start the frontend development server:
   ```
   cd frontend
   npm start
   ```
   The frontend will run on `http://localhost:3000`.

3. Open your browser and navigate to `http://localhost:3000` to use the application.

## Using the Application

1. Enter a YouTube video URL in the provided input field.
2. Choose an operation: Summarize or Ask a Question.
3. If summarizing, select the AI model you want to use.
4. If asking a question, enter your question in the provided field.
5. Click the submit button to process your request.
6. View the results displayed on the page.

## API Endpoints

- `POST /api/transcript`: Fetch a YouTube video transcript
- `POST /api/summarize`: Summarize a video transcript
- `POST /api/answer`: Answer a question based on a video transcript
- `GET /test`: Test endpoint to verify the Flask app is running

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
