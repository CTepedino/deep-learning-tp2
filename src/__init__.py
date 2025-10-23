"""
RAG System for Academic Exercise Generation
Sistema RAG para generación de ejercicios académicos usando LangChain

Módulos principales:
- data_loading: Carga de documentos PDF, TXT, TEX
- text_processing: Fragmentación inteligente de textos
- vector_store: Base de datos vectorial con ChromaDB
- retriever: Búsqueda semántica con LangChain
- generator: Generación de ejercicios con ChatOpenAI
- rag_pipeline: Pipeline completo RAG con LCEL
- evaluation: Evaluación con métricas RAGAS
- query_utils: Utilidades para consultas
"""

from .rag_pipeline import RAGPipeline, create_rag_pipeline
from .generator import ExerciseGenerator
from .retriever import Retriever, create_retriever
from .vector_store import VectorStore, create_vector_store
from .data_loading import DocumentLoader
from .text_processing import AcademicTextSplitter, split_documents
from .evaluation import RAGEvaluator
from .query_utils import prepare_search_query
from .export_utils import ExerciseExporter, export_exercises

__all__ = [
    # Main pipeline
    'RAGPipeline',
    'create_rag_pipeline',
    # Components
    'ExerciseGenerator',
    'Retriever',
    'create_retriever',
    'VectorStore',
    'create_vector_store',
    'DocumentLoader',
    'AcademicTextSplitter',
    'split_documents',
    'RAGEvaluator',
    'prepare_search_query',
    # Export
    'ExerciseExporter',
    'export_exercises',
]

__version__ = '2.0.0'
__author__ = 'Deep Learning TP2 - ITBA'

