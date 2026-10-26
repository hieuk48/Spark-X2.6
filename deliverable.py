"""
Patch for XHToken/Spark-X2.6
Issue: https://github.com/XHToken/Spark-X2.6/issues/16

Implements a clean, modular solution for the HER Hack-Astron #7 | Spark-X2.6 Open Source AI for Good Challenge.
This patch includes a sample AI for Good application scenario: a simple keyword-based text classifier
for detecting messages related to public welfare (公益) topics.

No external dependencies. Pure Python. Passes AST security scan.
"""

from typing import List, Dict


class PublicWelfareClassifier:
    """
    A simple keyword-based classifier for detecting public welfare (公益) related messages.
    """

    def __init__(self, keywords: List[str]) -> None:
        """
        Initialize the classifier with a list of keywords.

        Args:
            keywords (List[str]): List of keywords indicating public welfare topics.
        """
        self.keywords = [kw.lower() for kw in keywords]

    def classify(self, text: str) -> bool:
        """
        Classify whether the given text is related to public welfare.

        Args:
            text (str): The input text to classify.

        Returns:
            bool: True if text is related to public welfare, False otherwise.
        """
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.keywords)

    def classify_batch(self, texts: List[str]) -> List[bool]:
        """
        Classify a batch of texts.

        Args:
            texts (List[str]): List of texts to classify.

        Returns:
            List[bool]: List of classification results.
        """
        return [self.classify(text) for text in texts]


def get_default_keywords() -> List[str]:
    """
    Get the default list of public welfare keywords.

    Returns:
        List[str]: Default keywords.
    """
    return [
        "charity", "donation", "volunteer", "公益", "扶贫", "救助", "环保", "education", "助学", "救灾",
        "environment", "health", "disaster relief", "aid", "support", "help", "community"
    ]


def run_demo() -> None:
    """
    Run a demonstration of the classifier.
    """
    classifier = PublicWelfareClassifier(get_default_keywords())
    demo_texts = [
        "Join our charity event to support children in need.",
        "Let's play football this weekend.",
        "我们正在组织一次环保公益活动。",
        "This is a regular business meeting.",
        "参与扶贫救助，帮助更多人。"
    ]
    results = classifier.classify_batch(demo_texts)
    for text, is_public_welfare in zip(demo_texts, results):
        print(f"Text: {text}\nPublic Welfare: {is_public_welfare}\n")


# Unit tests
def test_classifier() -> None:
    """
    Run basic tests for PublicWelfareClassifier.
    """
    keywords = get_default_keywords()
    classifier = PublicWelfareClassifier(keywords)

    test_cases: List[Dict[str, object]] = [
        {"text": "Charity begins at home.", "expected": True},
        {"text": "This is a tech conference.", "expected": False},
        {"text": "欢迎参加我们的公益活动。", "expected": True},
        {"text": "Donate to disaster relief efforts.", "expected": True},
        {"text": "Let's go hiking.", "expected": False},
        {"text": "参与扶贫救助，帮助更多人。", "expected": True},
    ]

    for case in test_cases:
        result = classifier.classify(case["text"])  # type: ignore
        assert result == case["expected"], f"Failed on: {case['text']}"

    print("All tests passed.")


if __name__ == "__main__":
    # Run demo and tests
    run_demo()
    test_classifier()