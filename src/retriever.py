"""
Retriever Module using LangChain
Búsqueda semántica con filtrado por metadata usando LangChain nativo
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

logger = logging.getLogger(__name__)


class Retriever:
    """
    Wrapper del Retriever nativo de LangChain con capacidades de filtrado avanzado
    """
    
    def __init__(self, vector_store, k: int = 5, score_threshold: float = 0.0):
        """
        Inicializa el retriever usando el retriever nativo de LangChain
        
        Args:
            vector_store: Instancia del vector store de LangChain
            k: Número de documentos a recuperar por defecto
            score_threshold: Umbral mínimo de similitud
        """
        self.vector_store = vector_store
        self.k = k
        self.score_threshold = score_threshold
        
        # Crear retriever nativo de LangChain
        self._create_native_retriever()
    
    def _create_native_retriever(self):
        """Crea el retriever nativo de LangChain"""
        # Usar similarity_score_threshold como tipo de búsqueda
        self.native_retriever = self.vector_store.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.k,
                "score_threshold": self.score_threshold
            }
        )
        logger.info(f"Retriever nativo de LangChain creado con k={self.k}, threshold={self.score_threshold}")
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[str] = None,
        include_scores: bool = False
    ) -> List[Document]:
        """
        Recupera documentos relevantes usando el retriever nativo de LangChain
        
        Args:
            query: Consulta de búsqueda
            k: Número de documentos a recuperar
            filter_dict: Filtros de metadata
            include_scores: Si incluir scores de similitud
            
        Returns:
            Lista de documentos recuperados
        """
        try:
            k = k or self.k
            
            if include_scores:
                # Búsqueda con scores usando método directo
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter_dict=filter_dict
                )
                
                # Filtrar por score threshold
                filtered_results = [
                    doc for doc, score in results 
                    if score >= self.score_threshold
                ]
                
                logger.info(f"Recuperados {len(filtered_results)} documentos (con scores)")
                return filtered_results
            else:
                # Usar retriever nativo si no hay filtros especiales
                if filter_dict is None and k == self.k:
                    # Usar retriever nativo directamente
                    results = self.native_retriever.invoke(query)
                    logger.info(f"Recuperados {len(results)} documentos (retriever nativo)")
                    return results
                else:
                    # Búsqueda con parámetros personalizados
                    results = self.vector_store.similarity_search(
                        query=query,
                        k=k,
                        filter_dict=filter_dict
                    )
                    logger.info(f"Recuperados {len(results)} documentos")
                    return results
                
        except Exception as e:
            logger.error(f"Error en recuperación: {str(e)}")
            raise
    

    def get_native_retriever(self) -> VectorStoreRetriever:
        """
        Retorna el retriever nativo de LangChain
        
        Returns:
            Retriever nativo de LangChain
        """
        return self.native_retriever

def create_retriever(
    vector_store,
    k: int = 5,
    score_threshold: float = 0.0
) -> Retriever:
    """
    Función de conveniencia para crear un retriever con LangChain
    
    Args:
        vector_store: Instancia del vector store
        k: Número de documentos a recuperar
        score_threshold: Umbral de score
        
    Returns:
        Instancia del retriever
    """
    return Retriever(
        vector_store=vector_store,
        k=k,
        score_threshold=score_threshold
    )



