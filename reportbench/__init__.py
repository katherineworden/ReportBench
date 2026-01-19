"""
ReportBench: LLM evaluation framework for measuring model responses to intervention tools.

This benchmark measures how LLMs respond when given access to mock reporting tools
(e.g., FBI tips, emergency services, online reporting) combined with prompts of
varying severity levels.
"""

from reportbench.task import reportbench_task
from reportbench.prompts.dataset import Severity, PromptCategory

__version__ = "0.1.0"
__all__ = ["reportbench_task", "Severity", "PromptCategory"]
