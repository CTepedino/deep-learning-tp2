"""
RAG Pipeline Module using LangChain
Pipeline principal de RAG usando LangChain para máxima flexibilidad
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Imports de LangChain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Imports locales
from .data_loading import DataLoader
from .text_processing import TextProcessor
from .vector_store import create_vector_store
from .retriever import create_retriever
from .generator import ExerciseGenerator
from .query_utils import prepare_search_query

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Pipeline RAG completo usando LangChain para máxima flexibilidad
    """
    
    def __init__(
        self,
        collection_name: str = "itba_ejercicios_collection",
        persist_directory: str = "./data/processed/chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
        generator_model_name: str = "gpt-4o",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        reset_collection: bool = False
    ):
        """
        Inicializa el pipeline RAG con LangChain
        
        Args:
            collection_name: Nombre de la colección ChromaDB
            persist_directory: Directorio de persistencia
            embedding_model: Modelo de embeddings
            generator_model_name: Modelo de generación
            chunk_size: Tamaño de chunks
            chunk_overlap: Overlap entre chunks
            reset_collection: Si resetear la colección
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.generator_model_name = generator_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Inicializar componentes
        self.data_loader = DataLoader()
        self.text_processor = TextProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Vector store
        self.vector_store = create_vector_store(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            reset_collection=reset_collection
        )
        
        # Retriever
        self.retriever = create_retriever(
            vector_store=self.vector_store,
            k=5,
            score_threshold=0.0
        )
        
        # Generador de ejercicios
        self.generator = ExerciseGenerator(generator_model_name)
        
        logger.info("RAGPipeline inicializado correctamente")
    
    def load_materials(
        self,
        data_directory: str,
        file_extensions: List[str] = None
    ) -> Dict[str, Any]:
        """
        Carga materiales académicos desde un directorio
        
        Args:
            data_directory: Directorio con los materiales
            file_extensions: Extensiones de archivo a procesar
            
        Returns:
            Diccionario con estadísticas de carga
        """
        try:
            if file_extensions is None:
                file_extensions = ['.txt', '.pdf', '.tex']
            
            # Cargar documentos
            documents = self.data_loader.load_documents(
                directory=data_directory,
                file_extensions=file_extensions
            )
            
            if not documents:
                logger.warning(f"No se encontraron documentos en {data_directory}")
                return {"status": "error", "message": "No documents found"}
            
            # Procesar texto
            processed_docs = self.text_processor.process_documents(documents)
            
            # Convertir a formato LangChain Document
            langchain_docs = []
            for doc in processed_docs:
                langchain_doc = Document(
                    page_content=doc['content'],
                    metadata=doc['metadata']
                )
                langchain_docs.append(langchain_doc)
            
            # Agregar al vector store
            doc_ids = self.vector_store.add_documents(langchain_docs)
            
            # Estadísticas
            stats = {
                "status": "success",
                "documents_loaded": len(documents),
                "chunks_created": len(processed_docs),
                "documents_added": len(doc_ids),
                "collection_info": self.vector_store.get_collection_info()
            }
            
            logger.info(f"Materiales cargados: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Error cargando materiales: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def generate_exercises(
        self,
        query_params: Dict[str, Any],
        k_retrieval: int = 5,
        use_filters: bool = True
    ) -> Dict[str, Any]:
        """
        Genera ejercicios basados en los parámetros de consulta
        
        Args:
            query_params: Parámetros de la consulta
            k_retrieval: Número de documentos a recuperar
            use_filters: Si usar filtros de metadata
            
        Returns:
            Diccionario con ejercicios generados
        """
        try:
            # Preparar consulta de búsqueda
            search_query = prepare_search_query(query_params)
            
            # Construir filtros si es necesario
            filter_dict = None
            if use_filters:
                filter_dict = {}
                if query_params.get("materia"):
                    filter_dict["materia"] = query_params["materia"]
                if query_params.get("tipo_documento"):
                    filter_dict["tipo_documento"] = query_params["tipo_documento"]
                if query_params.get("nivel_dificultad"):
                    filter_dict["difficulty_hint"] = query_params["nivel_dificultad"]
            
            # Recuperar contexto relevante
            context_docs = self.retriever.retrieve(
                query=search_query,
                k=k_retrieval,
                filter_dict=filter_dict
            )
            
            if not context_docs:
                return {
                    "status": "error",
                    "message": "No se encontró contexto relevante para generar ejercicios"
                }
            
            # Generar ejercicios
            exercises = self.generator.generate_exercises(
                query_params=query_params,
                context_documents=context_docs
            )
            
            # Agregar información de contexto
            exercises["context_info"] = {
                "documents_retrieved": len(context_docs),
                "search_query": search_query,
                "filters_applied": filter_dict
            }
            
            logger.info(f"Ejercicios generados: {len(exercises.get('ejercicios', []))}")
            return exercises
            
        except Exception as e:
            logger.error(f"Error generando ejercicios: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def search_materials(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Busca materiales en la base de datos vectorial
        
        Args:
            query: Consulta de búsqueda
            k: Número de resultados
            filter_dict: Filtros de metadata
            
        Returns:
            Lista de documentos encontrados
        """
        try:
            # Buscar documentos
            results = self.retriever.retrieve(
                query=query,
                k=k,
                filter_dict=filter_dict
            )
            
            # Convertir a formato de respuesta
            search_results = []
            for doc in results:
                search_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"Búsqueda completada: {len(search_results)} resultados")
            return search_results
            
        except Exception as e:
            logger.error(f"Error en búsqueda: {str(e)}")
            return []
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Obtiene información del sistema
        
        Returns:
            Diccionario con información del sistema
        """
        try:
            collection_info = self.vector_store.get_collection_info()
            
            return {
                "pipeline_type": "RAG",
                "vector_store": {
                    "type": "ChromaDB with LangChain",
                    "collection_name": self.collection_name,
                    "persist_directory": self.persist_directory,
                    "embedding_model": self.embedding_model,
                    "document_count": collection_info.get("document_count", 0)
                },
                "retriever": {
                    "type": "Retriever",
                    "k": self.retriever.k,
                    "score_threshold": self.retriever.score_threshold
                },
                "generator": {
                    "type": "OpenAI",
                    "model": self.generator_model_name
                },
                "text_processing": {
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo información del sistema: {str(e)}")
            return {}
    
    def update_retriever_settings(
        self,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ):
        """
        Actualiza configuraciones del retriever
        
        Args:
            k: Número de documentos a recuperar
            score_threshold: Umbral de score
        """
        if k is not None:
            self.retriever.set_k(k)
        if score_threshold is not None:
            self.retriever.set_score_threshold(score_threshold)
        
        logger.info("Configuraciones del retriever actualizadas")
    
    def get_retrieval_stats(
        self,
        query: str,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de recuperación
        
        Args:
            query: Consulta de prueba
            k: Número de documentos a recuperar
            
        Returns:
            Estadísticas de recuperación
        """
        return self.retriever.get_retrieval_stats(query=query, k=k)


def create_rag_pipeline(
    collection_name: str = None,
    persist_directory: str = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    generator_model_name: str = "gpt-4o",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    reset_collection: bool = False
) -> RAGPipeline:
    """
    Función de conveniencia para crear un pipeline RAG con LangChain
    
    Args:
        collection_name: Nombre de la colección
        persist_directory: Directorio de persistencia
        embedding_model: Modelo de embeddings
        generator_model_name: Modelo de generación
        chunk_size: Tamaño de chunks
        chunk_overlap: Overlap entre chunks
        reset_collection: Si resetear la colección
        
    Returns:
        Instancia del pipeline RAG
    """
    # Usar variables de entorno si no se especifican
    if collection_name is None:
        collection_name = os.getenv('CHROMA_COLLECTION_NAME', 'itba_ejercicios_collection')
    
    if persist_directory is None:
        persist_directory = os.getenv('CHROMA_PERSIST_DIRECTORY', './data/processed/chroma_db')
    
    return RAGPipeline(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        generator_model_name=generator_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reset_collection=reset_collection
    )


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Crear pipeline RAG
    print("Creando pipeline RAG...")
    rag_pipeline = create_rag_pipeline(
        embedding_model="all-MiniLM-L6-v2",
        reset_collection=True
    )
    
    # Mostrar información del sistema
    system_info = rag_pipeline.get_system_info()
    print("Información del Sistema:")
    print("=" * 50)
    for component, info in system_info.items():
        print(f"\n{component.upper()}:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    
    # Crear documentos de ejemplo
    from langchain_core.documents import Document
    
    sample_docs = [
        Document(
            page_content="La distribución normal es fundamental en estadística. Se usa para modelar muchos fenómenos naturales.",
            metadata={"materia": "Probabilidad y estadística", "tipo_documento": "apunte", "difficulty_hint": "introductorio"}
        ),
        Document(
            page_content="Calcular P(X < 2) para X ~ N(0, 1) usando tablas de distribución normal estándar.",
            metadata={"materia": "Probabilidad y estadística", "tipo_documento": "ejercicio", "difficulty_hint": "intermedio"}
        ),
        Document(
            page_content="Los algoritmos de clustering como K-means son importantes en machine learning.",
            metadata={"materia": "Sistemas de Inteligencia Artificial", "tipo_documento": "concepto", "difficulty_hint": "avanzado"}
        )
    ]
    
    # Agregar documentos
    print("\nAgregando documentos...")
    doc_ids = rag_pipeline.vector_store.add_documents(sample_docs)
    print(f"Documentos agregados: {len(doc_ids)}")
    
    # Ejemplo de generación de ejercicios
    print("\nGenerando ejercicios...")
    query_params = {
        "materia": "Probabilidad y estadística",
        "unidad": "Distribuciones continuas",
        "tipo_ejercicio": "multiple_choice",
        "cantidad": 2,
        "nivel_dificultad": "intermedio"
    }
    
    exercises = rag_pipeline.generate_exercises(query_params)
    print(f"Ejercicios generados: {exercises}")
    
    # Ejemplo de búsqueda
    print("\nBuscando materiales...")
    search_results = rag_pipeline.search_materials("distribución normal", k=2)
    for i, result in enumerate(search_results):
        print(f"Resultado {i+1}: {result['content'][:100]}...")
    
    # Estadísticas de recuperación
    print("\nEstadísticas de recuperación:")
    stats = rag_pipeline.get_retrieval_stats("probabilidad")
    print(f"Estadísticas: {stats}")
