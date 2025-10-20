"""
Vector Store Module using LangChain ChromaDB
Gestión de base de datos vectorial usando LangChain con ChromaDB
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Gestor de base de datos vectorial usando LangChain con ChromaDB
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: str = None,
        reset_collection: bool = False
    ):
        """
        Inicializa el vector store con LangChain
        
        Args:
            collection_name: Nombre de la colección
            persist_directory: Directorio para persistencia
            embedding_model: Modelo de embeddings
            reset_collection: Si resetear la colección existente
        """
        # Leer valores de variables de entorno si no se especifican
        self.collection_name = collection_name or os.getenv('CHROMA_COLLECTION_NAME', 'itba_ejercicios_collection')
        self.persist_directory = persist_directory or os.getenv('CHROMA_PERSIST_DIRECTORY', './data/processed/chroma_db')
        self.embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.vectorstore = None
        
        # Crear directorio de persistencia si no existe
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        self._initialize_vectorstore(reset_collection)
    
    def _initialize_vectorstore(self, reset_collection: bool = False):
        """Inicializa el vector store de LangChain con ChromaDB"""
        try:
            from langchain_chroma import Chroma
            
            # Configurar embeddings según el modelo especificado
            if self.embedding_model.lower() == "openai" or self.embedding_model.startswith("text-embedding"):
                # Usar embeddings de OpenAI
                from langchain_openai import OpenAIEmbeddings
                
                # Si solo dice "openai", usar el modelo por defecto
                if self.embedding_model.lower() == "openai":
                    openai_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
                else:
                    # Si especifica el modelo completo (ej: "text-embedding-3-large")
                    openai_model = self.embedding_model
                
                self.embedding_function = OpenAIEmbeddings(model=openai_model)
                logger.info(f"Usando OpenAI embeddings: {openai_model}")
            else:
                # Usar embeddings locales con sentence-transformers
                from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
                self.embedding_function = SentenceTransformerEmbeddings(
                    model_name=self.embedding_model
                )
                logger.info(f"Usando sentence-transformers: {self.embedding_model}")
            
            # Verificar si existe colección y resetear si es necesario
            if reset_collection and self._collection_exists():
                self._delete_collection()
            
            # Inicializar vector store
            if self._collection_exists() and not reset_collection:
                # Cargar colección existente
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Colección existente cargada: {self.collection_name}")
            else:
                # Crear nueva colección
                self.vectorstore = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Nueva colección creada: {self.collection_name}")
            
        except ImportError as e:
            logger.error(f"Error importando LangChain: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error inicializando vector store: {str(e)}")
            raise
    
    def _collection_exists(self) -> bool:
        """Verifica si la colección existe"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            collections = [col.name for col in client.list_collections()]
            return self.collection_name in collections
        except Exception:
            return False
    
    def _delete_collection(self):
        """Elimina la colección existente"""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            client.delete_collection(self.collection_name)
            logger.info(f"Colección eliminada: {self.collection_name}")
        except Exception as e:
            logger.warning(f"No se pudo eliminar la colección: {str(e)}")
    
    def add_documents(
        self, 
        documents: List[Any],
        batch_size: int = 100
    ) -> List[str]:
        """
        Agrega documentos al vector store
        
        Args:
            documents: Lista de documentos a agregar
            batch_size: Tamaño del lote para procesamiento
            
        Returns:
            Lista de IDs de documentos agregados
        """
        try:
            if not documents:
                logger.warning("No hay documentos para agregar")
                return []
            
            # Procesar en lotes para documentos grandes
            all_ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                ids = self.vectorstore.add_documents(batch)
                all_ids.extend(ids)
                logger.info(f"Lote {i//batch_size + 1} agregado: {len(batch)} documentos")
            
            logger.info(f"Total documentos agregados: {len(all_ids)}")
            return all_ids
            
        except Exception as e:
            logger.error(f"Error agregando documentos: {str(e)}")
            raise
    
    def _convert_filter_to_chroma_format(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte filtros simples al formato que ChromaDB espera
        
        Args:
            filter_dict: Diccionario simple de filtros {key: value}
            
        Returns:
            Filtro en formato ChromaDB
        """
        if not filter_dict or len(filter_dict) == 0:
            return None
        
        if len(filter_dict) == 1:
            # Un solo filtro, formato simple
            return filter_dict
        
        # Múltiples filtros, usar operador $and
        conditions = [{key: value} for key, value in filter_dict.items()]
        return {"$and": conditions}
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Búsqueda por similitud
        
        Args:
            query: Consulta de búsqueda
            k: Número de resultados a retornar
            filter_dict: Filtros de metadata
            
        Returns:
            Lista de documentos similares
        """
        try:
            if filter_dict:
                # Convertir filtros al formato ChromaDB
                chroma_filter = self._convert_filter_to_chroma_format(filter_dict)
                # Búsqueda con filtros
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k,
                    filter=chroma_filter
                )
            else:
                # Búsqueda sin filtros
                results = self.vectorstore.similarity_search(
                    query=query,
                    k=k
                )
            
            logger.info(f"Búsqueda completada: {len(results)} resultados para '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda por similitud: {str(e)}")
            raise
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """
        Búsqueda por similitud con scores
        
        Args:
            query: Consulta de búsqueda
            k: Número de resultados a retornar
            filter_dict: Filtros de metadata
            
        Returns:
            Lista de tuplas (documento, score)
        """
        try:
            if filter_dict:
                # Convertir filtros al formato ChromaDB
                chroma_filter = self._convert_filter_to_chroma_format(filter_dict)
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=chroma_filter
                )
            else:
                results = self.vectorstore.similarity_search_with_score(
                    query=query,
                    k=k
                )
            
            logger.info(f"Búsqueda con scores completada: {len(results)} resultados")
            return results
            
        except Exception as e:
            logger.error(f"Error en búsqueda con scores: {str(e)}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Obtiene información de la colección
        
        Returns:
            Diccionario con información de la colección
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection(self.collection_name)
            
            count = collection.count()
            
            return {
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'document_count': count,
                'embedding_model': self.embedding_model,
                'embedding_function': str(self.embedding_function)
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo información de la colección: {str(e)}")
            return {}
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Elimina documentos por IDs
        
        Args:
            ids: Lista de IDs a eliminar
            
        Returns:
            True si se eliminaron correctamente
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection(self.collection_name)
            
            collection.delete(ids=ids)
            logger.info(f"Documentos eliminados: {len(ids)}")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando documentos: {str(e)}")
            return False
    
    def search_by_metadata(
        self,
        filter_dict: Dict[str, Any],
        limit: int = 10
    ) -> List[Any]:
        """
        Búsqueda por metadata
        
        Args:
            filter_dict: Filtros de metadata
            limit: Límite de resultados
            
        Returns:
            Lista de documentos que coinciden con los filtros
        """
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_directory)
            collection = client.get_collection(self.collection_name)
            
            results = collection.get(
                where=filter_dict,
                limit=limit
            )
            
            # Convertir a formato de documentos LangChain
            documents = []
            for i, doc_id in enumerate(results['ids']):
                from langchain_core.documents import Document
                doc = Document(
                    page_content=results['documents'][i],
                    metadata=results['metadatas'][i] if results['metadatas'] else {}
                )
                documents.append(doc)
            
            logger.info(f"Búsqueda por metadata completada: {len(documents)} resultados")
            return documents
            
        except Exception as e:
            logger.error(f"Error en búsqueda por metadata: {str(e)}")
            raise


def create_vector_store(
    collection_name: str = None,
    persist_directory: str = None,
    embedding_model: str = None,
    reset_collection: bool = False
) -> VectorStore:
    """
    Función de conveniencia para crear un vector store con LangChain
    
    Args:
        collection_name: Nombre de la colección (por defecto de CHROMA_COLLECTION_NAME)
        persist_directory: Directorio de persistencia (por defecto de CHROMA_PERSIST_DIRECTORY)
        embedding_model: Modelo de embeddings (por defecto de EMBEDDING_MODEL)
        reset_collection: Si resetear la colección
        
    Returns:
        Instancia del vector store
    """
    return VectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        reset_collection=reset_collection
    )


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Crear vector store
    print("Creando vector store...")
    vector_store = create_vector_store(
        reset_collection=True
    )
    
    # Mostrar información de la colección
    info = vector_store.get_collection_info()
    print(f"Información de la colección: {info}")
    
    # Crear documentos de ejemplo
    from langchain_core.documents import Document
    
    sample_docs = [
        Document(
            page_content="La distribución normal es una distribución de probabilidad continua.",
            metadata={"materia": "Probabilidad y estadística", "tipo": "definición"}
        ),
        Document(
            page_content="Calcular la probabilidad P(X < 2) para X ~ N(0, 1).",
            metadata={"materia": "Probabilidad y estadística", "tipo": "ejercicio"}
        ),
        Document(
            page_content="Los algoritmos de machine learning incluyen regresión y clasificación.",
            metadata={"materia": "Sistemas de Inteligencia Artificial", "tipo": "concepto"}
        )
    ]
    
    # Agregar documentos
    print("Agregando documentos...")
    ids = vector_store.add_documents(sample_docs)
    print(f"Documentos agregados con IDs: {ids}")
    
    # Búsqueda por similitud
    print("\nBúsqueda por similitud:")
    results = vector_store.similarity_search("distribución normal", k=2)
    for i, doc in enumerate(results):
        print(f"Resultado {i+1}: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
    
    # Búsqueda con filtros
    print("\nBúsqueda con filtros (solo ejercicios):")
    filtered_results = vector_store.similarity_search(
        "probabilidad",
        k=2,
        filter_dict={"tipo": "ejercicio"}
    )
    for i, doc in enumerate(filtered_results):
        print(f"Resultado filtrado {i+1}: {doc.page_content}")
    
    # Búsqueda por metadata
    print("\nBúsqueda por metadata:")
    metadata_results = vector_store.search_by_metadata(
        {"materia": "Probabilidad y estadística"},
        limit=5
    )
    for i, doc in enumerate(metadata_results):
        print(f"Por metadata {i+1}: {doc.page_content}")
    
    # Información final
    final_info = vector_store.get_collection_info()
    print(f"\nInformación final: {final_info}")
