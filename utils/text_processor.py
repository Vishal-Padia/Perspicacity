import nltk
from typing import List, Optional
from transformers import pipeline
from config import Config


class TextProcessor:
    def __init__(self):
        self.text_generator = pipeline(
            "text2text-generation",
            model=Config.TEXT_GENERATION_MODEL,
            max_length=Config.MAX_SEQUENCE_LENGTH,
        )
        self.query_processor = pipeline(
            "text2text-generation",
            model=Config.QUERY_PROCESSING_MODEL,
            max_length=Config.MAX_SEQUENCE_LENGTH,
        )
        nltk.download("punkt", quiet=True)

    def preprocess_query(self, query: str) -> str:
        """Process the query to make it more search-friendly"""
        prompt = f"Convert this query into a detailed search query: {query}"
        processed_query = self.query_processor(prompt, max_length=100, min_length=20)[
            0
        ]["generated_text"]
        return processed_query

    def create_text_chunks(self, text: str) -> List[str]:
        """Break text into manageable chunks"""
        if not text.strip():
            return []

        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if current_length + sentence_tokens <= Config.MAX_CHUNK_TOKENS:
                current_chunk.append(sentence)
                current_length += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_tokens

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def process_chunk(self, chunk: str, query: str) -> Optional[str]:
        """Process a single chunk of text"""
        if len(chunk.split()) < 30:
            return None

        chunk_prompt = f"""Transform this text segment to address: "{query}"
        Text: {chunk}
        Provide a clear and relevant response."""

        try:
            response = self.text_generator(
                chunk_prompt,
                max_length=250,
                min_length=50,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                num_beams=4,
            )[0]["generated_text"]
            return response
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
            return None

    def synthesize_responses(self, responses: List[str], query: str) -> str:
        """Combine multiple chunk responses into a final response"""
        if not responses:
            return "Unable to generate response from the available content."

        final_prompt = f"""Synthesize these segment responses into a coherent answer for: "{query}"
        Responses: {' '.join(responses)}"""

        try:
            final_response = self.text_generator(
                final_prompt,
                max_length=400,
                min_length=100,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                num_beams=4,
            )[0]["generated_text"]
            return final_response
        except Exception as e:
            print(f"Error in final synthesis: {str(e)}")
            return "Error generating final response."
