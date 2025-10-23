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
                    # Usar retriever nativo directamente (ideal para LCEL chains)
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
    
    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Recupera documentos con scores de similitud
        
        Args:
            query: Consulta de búsqueda
            k: Número de documentos a recuperar
            filter_dict: Filtros de metadata
            
        Returns:
            Lista de tuplas (documento, score)
        """
        try:
            k = k or self.k
            
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter_dict=filter_dict
            )
            
            # Filtrar por score threshold
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= self.score_threshold
            ]
            
            logger.info(f"Recuperados {len(filtered_results)} documentos con scores")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error en recuperación con scores: {str(e)}")
            raise
    
    def retrieve_by_materia(
        self,
        query: str,
        materia: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Recupera documentos filtrados por materia
        
        Args:
            query: Consulta de búsqueda
            materia: Materia a filtrar
            k: Número de documentos a recuperar
            
        Returns:
            Lista de documentos de la materia especificada
        """
        filter_dict = {"materia": materia}
        return self.retrieve(query=query, k=k, filter_dict=filter_dict)
    
    def get_retrieval_stats(
        self,
        query: str,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la recuperación
        
        Args:
            query: Consulta de búsqueda
            k: Número de documentos a recuperar
            
        Returns:
            Diccionario con estadísticas
        """
        try:
            k = k or self.k
            
            # Recuperar con scores
            results_with_scores = self.retrieve_with_scores(query=query, k=k)
            
            if not results_with_scores:
                return {"total_documents": 0}
            
            scores = [score for _, score in results_with_scores]
            
            # Estadísticas por metadata
            materias = {}
            tipos = {}
            dificultades = {}
            
            for doc, _ in results_with_scores:
                # Contar materias
                materia = doc.metadata.get('materia', 'No especificada')
                materias[materia] = materias.get(materia, 0) + 1
                
                # Contar tipos
                tipo = doc.metadata.get('tipo_documento', 'No especificado')
                tipos[tipo] = tipos.get(tipo, 0) + 1
                
                # Contar dificultades
                dificultad = doc.metadata.get('difficulty_hint', 'No especificada')
                dificultades[dificultad] = dificultades.get(dificultad, 0) + 1
            
            return {
                "total_documents": len(results_with_scores),
                "score_stats": {
                    "min": min(scores),
                    "max": max(scores),
                    "avg": sum(scores) / len(scores)
                },
                "materias": materias,
                "tipos_documento": tipos,
                "dificultades": dificultades
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {str(e)}")
            return {}
    
    def get_native_retriever(self) -> VectorStoreRetriever:
        """
        Retorna el retriever nativo de LangChain para uso en LCEL chains
        
        Returns:
            Retriever nativo de LangChain
        """
        return self.native_retriever
    
    def set_k(self, k: int):
        """Actualiza el número de documentos a recuperar por defecto"""
        self.k = k
        # Recrear retriever con nuevo k
        self._create_native_retriever()
        logger.info(f"K actualizado a: {k}")
    
    def set_score_threshold(self, threshold: float):
        """Actualiza el umbral de score"""
        self.score_threshold = threshold
        # Recrear retriever con nuevo threshold
        self._create_native_retriever()
        logger.info(f"Score threshold actualizado a: {threshold}")


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


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Importar dependencias
    from vector_store import create_vector_store
    
    # Crear vector store y retriever
    vector_store = create_vector_store(
        embedding_model="all-MiniLM-L6-v2",
        reset_collection=True
    )
    
    retriever = create_retriever(vector_store, k=3)
    
    # Crear documentos de ejemplo
    from langchain_core.documents import Document
    
    sample_docs = [
        Document(
            page_content="La distribución normal es fundamental en estadística.",
            metadata={"materia": "Probabilidad y estadística", "tipo_documento": "apunte", "difficulty_hint": "introductorio"}
        ),
        Document(
            page_content="Calcular P(X < 2) para X ~ N(0, 1) usando tablas.",
            metadata={"materia": "Probabilidad y estadística", "tipo_documento": "ejercicio", "difficulty_hint": "intermedio"}
        ),
        Document(
            page_content="Los algoritmos de clustering son importantes en ML.",
            metadata={"materia": "Sistemas de Inteligencia Artificial", "tipo_documento": "concepto", "difficulty_hint": "avanzado"}
        )
    ]
    
    # Agregar documentos al vector store
    vector_store.add_documents(sample_docs)
    
    # Ejemplos de recuperación
    print("=== Recuperación básica ===")
    results = retriever.retrieve("distribución normal")
    for i, doc in enumerate(results):
        print(f"Resultado {i+1}: {doc.page_content}")
    
    print("\n=== Recuperación por materia ===")
    results = retriever.retrieve_by_materia("probabilidad", "Probabilidad y estadística")
    for i, doc in enumerate(results):
        print(f"Por materia {i+1}: {doc.page_content}")
    
    print("\n=== Recuperación con contexto académico ===")
    results = retriever.retrieve_academic_context(
        "ejercicios",
        materia="Probabilidad y estadística",
        tipo_documento="ejercicio"
    )
    for i, doc in enumerate(results):
        print(f"Contexto académico {i+1}: {doc.page_content}")
    
    print("\n=== Estadísticas de recuperación ===")
    stats = retriever.get_retrieval_stats("distribución normal")
    print(f"Estadísticas: {stats}")
