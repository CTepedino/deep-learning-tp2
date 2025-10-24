"""
RAG Pipeline Module using LangChain
Pipeline principal de RAG usando LangChain
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Imports de LangChain
from langchain_core.documents import Document

# Imports locales
from .data_loading import DocumentLoader
from .text_processing import TextProcessor
from .vector_store import create_vector_store
from .retriever import create_retriever
from .generator import ExerciseGenerator
from .query_utils import prepare_search_query, normalize_text

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Pipeline RAG completo usando LangChain
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: str = None,
        generator_model_name: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        reset_collection: bool = False
    ):
        """
        Inicializa el pipeline RAG
        
        Args:
            collection_name: Nombre de la colección ChromaDB
            persist_directory: Directorio de persistencia
            embedding_model: Modelo de embeddings
            generator_model_name: Modelo de generación
            chunk_size: Tamaño de chunks
            chunk_overlap: Overlap entre chunks
            reset_collection: Si resetear la colección
        """
        # Leer valores de variables de entorno si no se especifican
        self.collection_name = collection_name or os.getenv('CHROMA_COLLECTION_NAME', 'itba_ejercicios_collection')
        self.persist_directory = persist_directory or os.getenv('CHROMA_PERSIST_DIRECTORY', './data/processed/chroma_db')
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.generator_model_name = generator_model_name or os.getenv('LLM_MODEL', 'gpt-4o-mini')
        self.chunk_size = chunk_size if chunk_size is not None else int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else int(os.getenv('CHUNK_OVERLAP', '200'))
        
        # Inicializar componentes
        self.data_loader = DocumentLoader()
        self.text_processor = TextProcessor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        # Vector store
        self.vector_store = create_vector_store(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_model=self.embedding_model,
            reset_collection=reset_collection
        )
        
        # Retriever
        self.retriever = create_retriever(
            vector_store=self.vector_store,
            k=10,
            score_threshold=0.0
        )
        
        # Generador de ejercicios
        self.generator = ExerciseGenerator(self.generator_model_name)
        
        logger.info("RAGPipeline inicializado correctamente")
    
    def load_materials(
        self,
        data_directory: str,
        file_extensions: List[str] = None,
        skip_on_error: bool = True
    ) -> Dict[str, Any]:
        """
        Carga materiales académicos desde un directorio
        
        Args:
            data_directory: Directorio con los materiales
            file_extensions: Extensiones de archivo a procesar
            skip_on_error: Si continuar cuando hay errores en documentos individuales
            
        Returns:
            Diccionario con estadísticas de carga
        """
        try:
            if file_extensions is None:
                file_extensions = ['.txt', '.pdf', '.tex']
            
            # Cargar documentos con metadata académica de la jerarquía de carpetas
            documents = self.data_loader.load_documents_from_directory(
                directory_path=data_directory,
                metadata_extractor=self.data_loader.extract_academic_metadata
            )
            
            if not documents:
                logger.warning(f"No se encontraron documentos en {data_directory}")
                # No es un error crítico si ya hay docs en la DB
                collection_info = self.vector_store.get_collection_info()
                existing_count = collection_info.get('document_count', 0)
                if existing_count > 0:
                    return {
                        "status": "warning",
                        "message": f"No se encontraron nuevos documentos, pero hay {existing_count} documentos existentes",
                        "documents_loaded": 0,
                        "chunks_created": 0,
                        "documents_added": 0,
                        "existing_documents": existing_count
                    }
                return {"status": "error", "message": "No documents found"}
            
            # Procesar texto (fragmentar en chunks)
            processed_docs = self.text_processor.split_documents(documents)
            
            # Agregar al vector store
            doc_ids = self.vector_store.add_documents(processed_docs)
            
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
            
            # Si skip_on_error, no abortar - reportar estado parcial
            if skip_on_error:
                collection_info = self.vector_store.get_collection_info()
                existing_count = collection_info.get('document_count', 0)
                
                if existing_count > 0:
                    logger.warning(f"Continuando con {existing_count} documentos existentes")
                    return {
                        "status": "partial_error",
                        "message": str(e),
                        "documents_loaded": 0,
                        "chunks_created": 0,
                        "documents_added": 0,
                        "existing_documents": existing_count,
                        "can_continue": True
                    }
            
            return {"status": "error", "message": str(e), "can_continue": False}
    
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
                    # Normalizar materia para búsqueda robusta
                    materia_normalizada = normalize_text(query_params["materia"])
                    filter_dict["materia"] = materia_normalizada
            
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
            tipo_ejercicio = query_params.get("tipo_ejercicio", "multiple_choice")
            exercises = self.generator.generate_exercises(
                query_params=query_params,
                context_documents=context_docs,
                tipo_ejercicio=tipo_ejercicio
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
    embedding_model: str = None,
    generator_model_name: str = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
    reset_collection: bool = False
) -> RAGPipeline:
    """
    Función de conveniencia para crear un pipeline RAG
    
    Args:
        collection_name: Nombre de la colección (por defecto de CHROMA_COLLECTION_NAME)
        persist_directory: Directorio de persistencia (por defecto de CHROMA_PERSIST_DIRECTORY)
        embedding_model: Modelo de embeddings (por defecto de EMBEDDING_MODEL)
        generator_model_name: Modelo de generación (por defecto de LLM_MODEL)
        chunk_size: Tamaño de chunks (por defecto de CHUNK_SIZE)
        chunk_overlap: Overlap entre chunks (por defecto de CHUNK_OVERLAP)
        reset_collection: Si resetear la colección
        
    Returns:
        Instancia del pipeline RAG
    """
    return RAGPipeline(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        generator_model_name=generator_model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reset_collection=reset_collection
    )

