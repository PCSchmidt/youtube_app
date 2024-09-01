import React, { useState } from 'react';
import { Container, TextField, Button, Select, MenuItem, Typography, Paper, CircularProgress, Box } from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';
import axios from 'axios';
import backgroundImage from './images/myimage2.jpg';

// Create a custom theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#4CAF50',
    },
    background: {
      default: 'rgba(245, 247, 250, 0.8)',
    },
  },
});

function App() {
  // State variables
  const [videoUrl, setVideoUrl] = useState('');
  const [model, setModel] = useState('gpt-4');
  const [transcript, setTranscript] = useState('');
  const [summary, setSummary] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Function to fetch the transcript of a YouTube video
  const handleGetTranscript = async () => {
    setIsLoading(true);
    try {
      const transcriptResponse = await axios.post(`${process.env.REACT_APP_API_URL}/api/transcript`, { video_url: videoUrl });
      setTranscript(transcriptResponse.data.transcript);
      
      const summaryResponse = await axios.post(`${process.env.REACT_APP_API_URL}/api/summarize`, { 
        transcript: transcriptResponse.data.transcript,
        model: model
      });
      setSummary(summaryResponse.data.summary);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to answer a question based on the transcript
  const handleAskQuestion = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/answer`, { transcript, question, model });
      setAnswer(response.data.answer);
    } catch (error) {
      console.error('Error answering question:', error);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to download the transcript as a .txt file
  const handleDownloadTranscript = () => {
    const element = document.createElement("a");
    const file = new Blob([transcript], { type: 'text/plain' });
    element.href = URL.createObjectURL(file);
    element.download = `transcript_${videoUrl.split('v=')[1]}.txt`;
    document.body.appendChild(element);
    element.click();
  };

  // Handle Enter key press
  const handleKeyPress = (event, action) => {
    if (event.key === 'Enter') {
      action();
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <Box 
        className="app-background"
        sx={{ 
          minHeight: '100vh', 
          backgroundImage: `url(${backgroundImage})`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
          backgroundAttachment: 'fixed',
          display: 'flex', 
          flexDirection: 'column', 
          justifyContent: 'center',
          '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            right: 0,
            bottom: 0,
            left: 0,
            backgroundColor: 'rgba(255, 255, 255, 0.7)',
          }
        }}
      >
        <Container maxWidth="md" sx={{ position: 'relative', zIndex: 1 }}>
          <Typography variant="h4" gutterBottom align="center" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
            YouTube Transcript Analyzer
          </Typography>
          <Paper elevation={3} sx={{ padding: '20px', marginBottom: '20px', backgroundColor: 'background.default' }}>
            <TextField
              fullWidth
              label="YouTube Video URL"
              value={videoUrl}
              onChange={(e) => setVideoUrl(e.target.value)}
              onKeyPress={(e) => handleKeyPress(e, handleGetTranscript)}
              margin="normal"
            />
            <Select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              fullWidth
              sx={{ mt: 2, mb: 2 }}
            >
              <MenuItem value="gpt-4">GPT-4</MenuItem>
              <MenuItem value="claude">Claude</MenuItem>
              <MenuItem value="gemini">Gemini</MenuItem>
              <MenuItem value="llama">Llama</MenuItem>
            </Select>
            <Button 
              variant="contained" 
              onClick={handleGetTranscript} 
              sx={{ mt: 1 }}
              disabled={isLoading}
              fullWidth
            >
              {isLoading ? <CircularProgress size={24} /> : 'Get Transcript'}
            </Button>
          </Paper>

          {transcript && (
            <Paper elevation={3} sx={{ padding: '20px', marginTop: '20px', backgroundColor: 'background.default' }}>
              <Typography variant="h6">Transcript</Typography>
              <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>{transcript}</Typography>
              <Button variant="contained" onClick={handleDownloadTranscript} sx={{ mt: 1 }}>
                Download Transcript
              </Button>
            </Paper>
          )}

          {summary && (
            <Paper elevation={3} sx={{ padding: '20px', marginY: '20px', backgroundColor: 'background.default' }}>
              <Typography variant="h6" gutterBottom>Summary</Typography>
              <Typography variant="body1">{summary}</Typography>
            </Paper>
          )}

          <Paper elevation={3} sx={{ padding: '20px', backgroundColor: 'background.default' }}>
            <TextField
              fullWidth
              label="Ask a question"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyPress={(e) => handleKeyPress(e, handleAskQuestion)}
              margin="normal"
            />
            <Button 
              variant="contained" 
              onClick={handleAskQuestion} 
              sx={{ mt: 1 }}
              disabled={isLoading}
              fullWidth
            >
              {isLoading ? <CircularProgress size={24} /> : 'Ask Question'}
            </Button>
            {answer && (
              <Typography variant="body1" sx={{ mt: 2 }}>{answer}</Typography>
            )}
          </Paper>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;