import unittest
from main import extract_video_id, summarize_transcript, answer_question

class TestYouTubeApp(unittest.TestCase):
    def test_extract_video_id(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        video_id = extract_video_id(url)
        self.assertEqual(video_id, "dQw4w9WgXcQ")

    def test_summarize_transcript(self):
        transcript = "This is a test transcript."
        summary = summarize_transcript(transcript, model="gpt-3.5-turbo")
        self.assertIn("test transcript", summary)

    def test_answer_question(self):
        transcript = "This is a test transcript."
        question = "What is this?"
        answer = answer_question(transcript, question, model="gpt-3.5-turbo")
        self.assertIn("test transcript", answer)

if __name__ == '__main__':
    unittest.main()
