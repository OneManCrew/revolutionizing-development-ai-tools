import os
import asyncio
from langchain import LangChain
from openai import OpenAI
from typing import List, Dict

class AIEnhancedCodeReview:
    """
    A class that utilizes LangChain and OpenAI API to perform AI-enhanced code reviews.
    """
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.openai = OpenAI(api_key=api_key)

    async def analyze_code(self, code_snippet: str) -> Dict[str, str]:
        """
        Analyzes the provided code snippet using OpenAI's API and returns a structured review.

        Args:
            code_snippet (str): The code snippet to be analyzed.

        Returns:
            Dict[str, str]: A dictionary containing analysis results.
        """
        try:
            response = await self.openai.create_completion(
                model="gpt-4-code",
                prompt=f"Review the following code and provide improvements:\n{code_snippet}",
                max_tokens=150
            )
            return {"review": response.choices[0].text.strip()}
        except Exception as e:
            return {"error": str(e)}

    async def perform_code_review(self, code_snippets: List[str]) -> List[Dict[str, str]]:
        """
        Performs code reviews on a list of code snippets asynchronously.

        Args:
            code_snippets (List[str]): A list of code snippets to be reviewed.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing review results for each snippet.
        """
        tasks = [self.analyze_code(snippet) for snippet in code_snippets]
        return await asyncio.gather(*tasks)


async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    code_snippets = [
        "def add(a, b): return a + b",
        "def multiply(a, b): return a * b"
    ]

    reviewer = AIEnhancedCodeReview(api_key)
    reviews = await reviewer.perform_code_review(code_snippets)
    for review in reviews:
        print(review)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())