import time
import random
import requests
from typing import Optional, List, Dict
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from newspaper import Article, Config as NewsConfig
from googlesearch import search
from config import Config


class WebCrawler:
    def __init__(self):
        self.config = NewsConfig()
        self.session = requests.Session()
        self.session.headers.update(Config.DEFAULT_HEADERS)
        self._rotate_user_agent()

    def _rotate_user_agent(self):
        """Rotate user agent for requests"""
        new_agent = random.choice(Config.USER_AGENTS)
        self.config.browser_user_agent = new_agent
        self.session.headers["User-Agent"] = new_agent

    def _is_blocked_domain(self, url: str) -> bool:
        """Check if domain is blocked"""
        domain = urlparse(url).netloc.lower()
        return any(blocked in domain for blocked in Config.BLOCKED_DOMAINS)

    def crawl_page(self, url: str) -> Optional[str]:
        """Crawl a single webpage"""
        try:
            article = Article(url, config=self.config)
            article.download()
            article.parse()
            content = article.text

            if len(content.strip()) < Config.MIN_CONTENT_LENGTH:
                response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, "html.parser")

                for element in soup(
                    ["script", "style", "nav", "header", "footer", "iframe"]
                ):
                    element.decompose()

                content = " ".join(soup.stripped_strings)

            return (
                content if len(content.strip()) >= Config.MIN_CONTENT_LENGTH else None
            )

        except Exception as e:
            print(f"Error crawling {url}: {str(e)}")
            return None

    def search_and_crawl(
        self, query: str, num_results: int = 5
    ) -> List[Dict[str, str]]:
        """Perform search and crawl results"""
        results = []
        attempted_urls = 0
        max_attempts = num_results * 2

        try:
            search_results = list(
                search(
                    query,
                    num=max_attempts,
                    stop=max_attempts,
                    pause=Config.SEARCH_PAUSE,
                )
            )

            for url in search_results:
                if len(results) >= num_results:
                    break

                if self._is_blocked_domain(url):
                    continue

                self._rotate_user_agent()
                time.sleep(random.uniform(1.5, 3.5))

                content = self.crawl_page(url)
                if content:
                    results.append({"url": url, "content": content})

                attempted_urls += 1
                if attempted_urls >= max_attempts:
                    break

            return results

        except Exception as e:
            print(f"Error during search and crawl: {str(e)}")
            return results

    def combine_content(self, crawled_results: List[Dict[str, str]]) -> str:
        """Combine content from multiple crawled pages"""
        combined_text = ""
        for result in crawled_results:
            if result["content"]:
                content = result["content"]
                content = " ".join(content.split())
                content = content.replace("\n", " ").replace("\r", " ")
                combined_text += content + "\n\n"

        return combined_text.strip()
