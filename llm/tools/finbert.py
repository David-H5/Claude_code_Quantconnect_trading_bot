"""
FinBERT Sentiment Analysis Tool

Wrapper for FinBERT financial sentiment model.

QuantConnect Compatible: Yes (can be run in Initialize for model loading)
"""

from dataclasses import dataclass

from models.exceptions import AgentConfigurationError


@dataclass
class SentimentScore:
    """Sentiment analysis result."""

    label: str  # "positive", "negative", "neutral"
    score: float  # -1.0 (very negative) to +1.0 (very positive)
    confidence: float  # 0.0 to 1.0
    probabilities: dict[str, float]  # {"positive": X, "negative": Y, "neutral": Z}


class FinBERTAnalyzer:
    """
    FinBERT sentiment analyzer for financial text.

    Features:
    - Specialized for financial domain
    - Pre-trained on financial news
    - More accurate than generic sentiment models
    - Fast inference (<100ms per text)

    Example:
        analyzer = FinBERTAnalyzer()
        result = analyzer.analyze("Apple reports record earnings, stock surges")
        print(f"Sentiment: {result.label}, Score: {result.score:.2f}")
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT analyzer.

        Args:
            model_name: HuggingFace model name (default: ProsusAI/finbert)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.loaded = False

    def load(self) -> None:
        """
        Load FinBERT model and tokenizer.

        This should be called once during initialization, not on every request.
        """
        if self.loaded:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Use GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()

            self.model.eval()  # Set to evaluation mode
            self.loaded = True

        except ImportError:
            raise ImportError("transformers and torch required. Run: pip install transformers torch")

    def analyze(self, text: str) -> SentimentScore:
        """
        Analyze sentiment of financial text.

        Args:
            text: Financial text to analyze (news, tweet, etc.)

        Returns:
            SentimentScore with label, score, confidence, probabilities

        Raises:
            AgentConfigurationError: If model not loaded
        """
        if not self.loaded:
            raise AgentConfigurationError(
                agent_name="FinBERTAnalyzer",
                config_key="model",
                reason="Model not loaded. Call load() first.",
            )

        import torch

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1)
        probs = probs.cpu().numpy()[0]

        # FinBERT labels: positive, negative, neutral
        labels = ["positive", "negative", "neutral"]
        probabilities = {labels[i]: float(probs[i]) for i in range(len(labels))}

        # Get predicted label
        predicted_idx = probs.argmax()
        predicted_label = labels[predicted_idx]
        confidence = float(probs[predicted_idx])

        # Calculate sentiment score (-1 to +1)
        # Weighted by probabilities: positive = +1, negative = -1, neutral = 0
        score = probabilities["positive"] * 1.0 + probabilities["negative"] * (-1.0) + probabilities["neutral"] * 0.0

        return SentimentScore(
            label=predicted_label,
            score=score,
            confidence=confidence,
            probabilities=probabilities,
        )

    def analyze_batch(self, texts: list[str]) -> list[SentimentScore]:
        """
        Analyze sentiment for multiple texts.

        Args:
            texts: List of financial texts

        Returns:
            List of SentimentScore results
        """
        return [self.analyze(text) for text in texts]

    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self.tokenizer = None
        self.loaded = False


# Global instance
_finbert: FinBERTAnalyzer | None = None


def get_finbert_analyzer() -> FinBERTAnalyzer:
    """
    Get the global FinBERT analyzer instance.

    Returns:
        FinBERTAnalyzer instance (loads model if not already loaded)
    """
    global _finbert
    if _finbert is None:
        _finbert = FinBERTAnalyzer()
        _finbert.load()
    return _finbert


def analyze_financial_text(text: str) -> SentimentScore:
    """
    Convenience function to analyze financial text.

    Args:
        text: Financial text to analyze

    Returns:
        SentimentScore with sentiment analysis results
    """
    analyzer = get_finbert_analyzer()
    return analyzer.analyze(text)
