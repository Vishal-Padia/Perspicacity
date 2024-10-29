from typing import Dict, Any
from utils.text_processor import TextProcessor
from utils.web_crawler import WebCrawler


class SearchTextGenerator:
    def __init__(self):
        self.text_processor = TextProcessor()
        self.web_crawler = WebCrawler()

    def search_and_generate(self, query: str) -> Dict[str, Any]:
        """Main function to process query and generate response"""
        print(f"\nQuery: {query}")
        processed_query = self.text_processor.preprocess_query(query)

        print("Searching and analyzing content...")
        results = self.web_crawler.search_and_crawl(processed_query)

        if not results:
            return {
                "query": query,
                "processed_query": processed_query,
                "response": "No accessible results found. Please try a different search query.",
                "sources": [],
            }

        combined_content = self.web_crawler.combine_content(results)
        chunks = self.text_processor.create_text_chunks(combined_content)

        chunk_responses = []
        for chunk in chunks:
            response = self.text_processor.process_chunk(chunk, query)
            if response:
                chunk_responses.append(response)

        final_response = self.text_processor.synthesize_responses(
            chunk_responses, query
        )

        return {
            "query": query,
            "processed_query": processed_query,
            "response": final_response,
            "sources": [r["url"] for r in results],
        }


def main():
    generator = SearchTextGenerator()

    print("Welcome to the PerSpicacity")
    print("Enter your queries below. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Enter a query (or 'quit' to exit): ").strip()

            if not query:
                print("Please enter a valid query.")
                continue

            if query.lower() == "quit":
                print("\nThank you for using PerSpicacity!")
                break

            result = generator.search_and_generate(query)

            print("\nGenerated Response:")
            print("-" * 80)
            print(result["response"])
            print("-" * 80)
            print("\nSources:")
            for idx, url in enumerate(result["sources"], 1):
                print(f"{idx}. {url}")
            print("\n")

        except KeyboardInterrupt:
            print("\n\nExiting the system...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again with a different query.\n")


if __name__ == "__main__":
    main()
