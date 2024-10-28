import os
import time
import json
import nltk
import requests
import random

from bs4 import BeautifulSoup
from googlesearch import search
from transformers import pipeline
from urllib.parse import urlparse
from newspaper import Article, Config


class SearchSummarizer:
    def __init__(self):
        # List of commonly blocking domains
        self.blocked_domains = {
            "cloudflare.com",
            "crunchbase.com",
            "pitchbook.com",
            "linkedin.com",
            "facebook.com",
            "instagram.com",
            "twitter.com",
        }

        # List of user agents to rotate
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
        ]

        # Configure newspaper with more robust settings
        self.config = Config()
        self.config.browser_user_agent = random.choice(self.user_agents)
        self.config.request_timeout = 10
        self.config.fetch_images = False
        self.config.memoize_articles = False
        self.config.follow_meta_refresh = True

        # Initialize the summarizer
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        # Download the required NLTK data
        nltk.download("punkt", quiet=True)

        # Initialize session with custom headers
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )

    def is_blocked_domain(self, url):
        """Check if the domain is in the blocked list"""
        domain = urlparse(url).netloc.lower()
        return any(blocked in domain for blocked in self.blocked_domains)

    def rotate_user_agent(self):
        """Rotate user agent for the next request"""
        self.config.browser_user_agent = random.choice(self.user_agents)
        self.session.headers["User-Agent"] = self.config.browser_user_agent

    def search_and_crawl(self, query, num_results=7):
        """
        Perform a Google search and crawl the top search results with improved handling
        """
        results = []
        attempted_urls = 0
        max_attempts = num_results * 2  # Try more URLs to account for failures

        try:
            # Get more results than needed to account for blocked sites
            search_results = list(
                search(query, num=max_attempts, stop=max_attempts, pause=2)
            )

            for url in search_results:
                if len(results) >= num_results:
                    break

                if self.is_blocked_domain(url):
                    continue

                # Rotate user agent for each request
                self.rotate_user_agent()

                # Add randomized delay between requests
                time.sleep(random.uniform(1.5, 3.5))

                content = self.crawl_page(url)
                if content and len(content.strip()) > 200:  # Ensure meaningful content
                    results.append({"url": url, "content": content})

                attempted_urls += 1
                if attempted_urls >= max_attempts:
                    break

            return results

        except Exception as e:
            print(f"Error during search and crawl: {str(e)}")
            return results

    def crawl_page(self, url):
        """
        Crawl a single page with improved error handling and fallback methods
        """
        try:
            # First try with newspaper3k
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            content = article.text

            # If content is too short, try direct requests + BeautifulSoup as fallback
            if len(content.strip()) < 200:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove unwanted elements
                for element in soup(
                    ["script", "style", "nav", "header", "footer", "iframe"]
                ):
                    element.decompose()

                content = " ".join(soup.stripped_strings)

            return content if len(content.strip()) > 200 else None

        except Exception as e:
            if "403" in str(e):
                print(f"Access denied for {url} - skipping...")
            else:
                print(f"Error crawling {url}: {str(e)}")
            return None

    def combine_content(self, crawled_results):
        """
        Combine the content from all the crawled pages with improved text cleaning
        """
        combined_text = ""
        for result in crawled_results:
            if result["content"]:
                # Clean the content
                content = result["content"]
                content = " ".join(content.split())  # Remove extra whitespace
                content = content.replace("\n", " ").replace("\r", " ")
                combined_text += content + "\n\n"

        return combined_text.strip()

    def get_optimal_length(self, text):
        """
        Calculate optimal summary length based on input length with increased output length
        """
        input_length = len(text.split())

        max_length = max(min(int(input_length * 0.5), 1024), 100)
        min_length = max(int(max_length * 0.6), 50)
        return max_length, min_length

    def summarize_content(self, text):
        """
        Summarize the content with improved chunking and longer summaries
        """
        try:
            if not text.strip():
                return "No content to summarize."

            # Increased chunk size from 500 to 800
            sentences = nltk.sent_tokenize(text)
            chunks = []
            current_chunk = []
            current_length = 0

            for sentence in sentences:
                sentence_length = len(sentence.split())
                if current_length + sentence_length <= 1000:  # Increased from 500
                    current_chunk.append(sentence)
                    current_length += sentence_length
                else:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [sentence]
                    current_length = sentence_length

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Summarize chunks with longer length
            chunk_summaries = []
            print(f"Number of chunks: {len(chunks)}")
            for chunk in chunks:
                # Increased minimum chunk size from 50 to 80
                if len(chunk.split()) < 100:
                    continue

                max_length, min_length = self.get_optimal_length(chunk)
                try:
                    summary = self.summarizer(
                        chunk,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        length_penalty=1.5,  # Added length penalty
                        num_beams=4,  # Increased beam search
                    )[0]["summary_text"]

                    # Append the summary if it is not too short
                    chunk_summaries.append(summary)
                except Exception as e:
                    print(f"Error summarizing chunk: {str(e)}")
                    continue

            if not chunk_summaries:
                return "Unable to generate summary from the available content."

            # Final summarization with longer length
            final_text = " ".join(chunk_summaries)
            return final_text

        except Exception as e:
            print(f"Error during summarization: {str(e)}")
            return "Error generating summary."

    def search_and_summarize(self, query):
        """
        Main function with improved error handling and user feedback
        """
        print(f"\nSearching and analyzing content for: {query}")
        print("This may take a few moments...\n")

        results = self.search_and_crawl(query)

        if not results:
            return {
                "query": query,
                "summary": "No accessible results found. Please try a different search query.",
                "sources": [],
            }

        combined_content = self.combine_content(results)
        summary = self.summarize_content(combined_content)

        return {
            "query": query,
            "summary": summary,
            "sources": [r["url"] for r in results],
        }


def main():
    summarizer = SearchSummarizer()

    print("Welcome to the PerSpicacity")
    print("Enter your search queries below. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Enter a search query (or 'quit' to exit): ").strip()

            if not query:
                print("Please enter a valid query.")
                continue

            if query.lower() == "quit":
                print("\nThank you for using PerSpicacity!")
                break

            result = summarizer.search_and_summarize(query)

            print("\nSummary:")
            print("-" * 80)
            print(result["summary"])
            print("-" * 80)
            print("\nSources:")
            for idx, url in enumerate(result["sources"], 1):
                print(f"{idx}. {url}")
            print("\n")

        except KeyboardInterrupt:
            print("\n\nExiting PerSpicacity...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again with a different query.\n")


if __name__ == "__main__":
    main()
