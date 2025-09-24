"""
Medical Knowledge Graph RAG System Package

A Python package for medical question answering using Neo4j knowledge graphs
and Azure OpenAI integration.
"""

from .main import MedicalRAGSystem
from .utils import (
    MedicalQueryHelper,
    OptimizedMedicalChain,
    create_medical_query_helper,
    create_optimized_chain,
    ADVANCED_CYPHER_TEMPLATE,
    QA_TEMPLATE
)
from .demo import MedicalRAGDemo

__version__ = "1.0.0"
__author__ = "Medical RAG Team"

__all__ = [
    "MedicalRAGSystem",
    "MedicalQueryHelper", 
    "OptimizedMedicalChain",
    "MedicalRAGDemo",
    "create_medical_query_helper",
    "create_optimized_chain",
    "ADVANCED_CYPHER_TEMPLATE",
    "QA_TEMPLATE"
]