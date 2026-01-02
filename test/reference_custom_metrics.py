import os
import json
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.metrics.bias.bias import BiasMetric
from deepeval.metrics.toxicity.toxicity import ToxicityMetric
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.hallucination.hallucination import HallucinationMetric
from deepeval.test_case.llm_test_case import LLMTestCaseParams
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.models.llms.gemini_model import GeminiModel


def get_value(filepath, key: str, default: str = ""):
    """Get value from JSON file with error handling."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data.get(key, default)
    except FileNotFoundError:
        print(f"Warning: Config file not found: {filepath}")
        return default
    except json.JSONDecodeError:
        print(f"Warning: Invalid JSON in config file: {filepath}")
        return default
    except Exception as e:
        print(f"Warning: Error reading config file: {e}")
        return default


script_dir = os.path.dirname(os.path.abspath(__file__))
metrics_filepath = os.path.join(script_dir, "configs", "metrics.json")
prompts_filepath = os.path.join(script_dir, "configs", "prompts.json")

model = GeminiModel(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

ToneMetric = GEval(
    name="Tone",
    criteria=get_value(metrics_filepath, "TONE"),
    model=model,
    threshold=0.6,
    async_mode=False,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

CompletenessMetric = GEval(
    name="Completeness",
    criteria=get_value(metrics_filepath, "COMPLETENESS"),
    model=model,
    threshold=0.5,
    async_mode=False,
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
)

ClarityMetric = GEval(
    name="Clarity",
    threshold=0.6,
    criteria=get_value(metrics_filepath, "CLARITY"),
    model=model,
    async_mode=False,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
)

# Additional custom metrics for RAG/Graph evaluation
ContextRelevanceMetric = GEval(
    name="ContextRelevance",
    criteria=get_value(metrics_filepath, "CONTEXT_RELEVANCE"),
    model=model,
    threshold=0.5,
    async_mode=False,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
)

RetrievalQualityMetric = GEval(
    name="RetrievalQuality",
    criteria=get_value(metrics_filepath, "RETRIEVAL_QUALITY"),
    model=model,
    threshold=0.5,
    async_mode=False,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

SourceAttributionMetric = GEval(
    name="SourceAttribution",
    criteria=get_value(metrics_filepath, "SOURCE_ATTRIBUTION"),
    model=model,
    threshold=0.3,
    async_mode=False,
    evaluation_params=[
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
    ],
)

metrics = [
    # HallucinationMetric(model=model, async_mode=False),  # ACCURACY
    FaithfulnessMetric(model=model, threshold=0.5, async_mode=False),  # ACCURACY
    AnswerRelevancyMetric(model=model, threshold=0.5, async_mode=False),  # RELEVANCE
    ToxicityMetric(model=model, async_mode=False),  # SAFETY
    BiasMetric(model=model),  # SAFETY
    ToneMetric,  # custom metric
    ClarityMetric,  # custom metric
    CompletenessMetric,  # custom metric
    ContextRelevanceMetric,  # RAG/Graph custom metric
    RetrievalQualityMetric,  # RAG/Graph custom metric
    SourceAttributionMetric,  # RAG/Graph custom metric
]
