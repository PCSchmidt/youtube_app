import React, { useState } from 'react';
import { Container, TextField, Button, Select, MenuItem, Typography, Paper } from '@mui/material';
import axios from 'axios';

function App() {
  // State variables
  const [videoUrl, setVideoUrl] = useState('');
  const [model, setModel] = useState('gpt-4');
  const [transcript, setTranscript] = useState('');
  const [summary, setSummary] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');

  // Function to fetch the transcript of a YouTube video
  const handleGetTranscript = async () => {
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/transcript`, { video_url: videoUrl });
      setTranscript(response.data.transcript);
    } catch (error) {
      console.error('Error fetching transcript:', error);
    }
  };

  // Function to summarize the transcript
  const handleSummarize = async () => {
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/summarize`, { transcript, model });
      setSummary(response.data.summary);
    } catch (error) {
      console.error('Error summarizing transcript:', error);
    }
  };

  // Function to answer a question based on the transcript
  const handleAskQuestion = async () => {
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/answer`, { transcript, question, model });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('Error answering question:', error);
    }
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>YouTube Transcript Analyzer</Typography>
      <Paper elevation={3} style={{ padding: '20px', marginBottom: '20px' }}>
        <TextField
          fullWidth
          label="YouTube Video URL"
          value={videoUrl}
          onChange={(e) => setVideoUrl(e.target.value)}
          margin="normal"
        />
        <Select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          fullWidth
          margin="normal"
        >
          <MenuItem value="gpt-4">GPT-4</MenuItem>
          <MenuItem value="claude">Claude</MenuItem>
          <MenuItem value="gemini">Gemini</MenuItem>
          <MenuItem value="llama">Llama</MenuItem>
        </Select>
        <Button variant="contained" onClick={handleGetTranscript} style={{ marginTop: '10px' }}>
          Get Transcript
        </Button>
      </Paper>

      {transcript && (
        <Paper elevation={3} style={{ padding: '20px', marginBottom: '20px' }}>
          <Typography variant="h6" gutterBottom>Transcript</Typography>
          <Typography variant="body1">{transcript}</Typography>
          <Button variant="contained" onClick={handleSummarize} style={{ marginTop: '10px' }}>
            Summarize
          </Button>
        </Paper>
      )}

      {summary && (
        <Paper elevation={3} style={{ padding: '20px', marginBottom: '20px' }}>
          <Typography variant="h6" gutterBottom>Summary</Typography>
          <Typography variant="body1">{summary}</Typography>
        </Paper>
      )}

      <Paper elevation={3} style={{ padding: '20px' }}>
        <TextField
          fullWidth
          label="Ask a question"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          margin="normal"
        />
        <Button variant="contained" onClick={handleAskQuestion} style={{ marginTop: '10px' }}>
          Ask Question
        </Button>
        {answer && (
          <Typography variant="body1" style={{ marginTop: '10px' }}>{answer}</Typography>
        )}
      </Paper>
    </Container>
  );
}

export default App;