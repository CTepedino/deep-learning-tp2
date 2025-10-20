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
            from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
            
            # Configurar embeddings con validación y fallback
            try:
                self.embedding_function = SentenceTransformerEmbeddings(
                    model_name=self.embedding_model,
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                logger.warning(f"Error con modelo {self.embedding_model}, usando fallback: {str(e)}")
                # Usar modelo más simple como fallback
                self.embedding_function = SentenceTransformerEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    encode_kwargs={'normalize_embeddings': True}
                )
            
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
    
    def _clean_document_content(self, doc: Any) -> Any:
        """
        Limpia y valida el contenido de un documento preservando información de errores
        
        Args:
            doc: Documento a limpiar
            
        Returns:
            Documento limpio con información de errores preservada
        """
        try:
            # Guardar contenido original para debugging
            original_content = None
            if hasattr(doc, 'page_content'):
                original_content = doc.page_content
            
            # Limpiar contenido
            if hasattr(doc, 'page_content'):
                content = doc.page_content
                if content and isinstance(content, str):
                    # Limpieza conservadora - solo remover caracteres realmente problemáticos
                    import re
                    
                    # Solo remover caracteres de control que causan problemas específicos
                    # Mantener caracteres Unicode importantes
                    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
                    
                    # Normalizar espacios múltiples pero preservar estructura
                    content = re.sub(r'[ \t]+', ' ', content)
                    content = re.sub(r'\n\s*\n', '\n\n', content)  # Preservar párrafos
                    
                    # Limitar longitud pero preservar información
                    if len(content) > 10000:
                        content = content[:10000] + "\n\n[CONTENIDO TRUNCADO]"
                    
                    # Asegurar que no esté vacío
                    if not content.strip():
                        content = "Contenido no disponible"
                    
                    doc.page_content = content.strip()
                else:
                    doc.page_content = "Contenido no disponible"
            
            # Limpiar metadata preservando información
            if hasattr(doc, 'metadata') and doc.metadata:
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, str):
                        # Limpieza conservadora de metadata
                        import re
                        cleaned_value = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', str(value))
                        cleaned_value = re.sub(r'\s+', ' ', cleaned_value).strip()
                        cleaned_metadata[key] = cleaned_value[:500]  # Limitar longitud
                    else:
                        cleaned_metadata[key] = value
                
                # Agregar información de debugging si hubo problemas (formato compatible con ChromaDB)
                if original_content and len(original_content) != len(doc.page_content):
                    cleaned_metadata['_cleaning_info'] = f"original:{len(original_content)},cleaned:{len(doc.page_content)},was_cleaned:True"
                
                doc.metadata = cleaned_metadata
            
            return doc
        except Exception as e:
            logger.warning(f"Error limpiando documento: {str(e)}")
            # Preservar información del error en metadata
            if hasattr(doc, 'metadata'):
                doc.metadata['_cleaning_error'] = str(e)
                doc.metadata['_cleaning_failed'] = True
            if hasattr(doc, 'page_content'):
                if not doc.page_content or doc.page_content == "Contenido no disponible":
                    doc.page_content = f"Error procesando contenido: {str(e)}"
            return doc

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
            
            # Limpiar documentos antes de agregar
            cleaned_documents = []
            for doc in documents:
                cleaned_doc = self._clean_document_content(doc)
                cleaned_documents.append(cleaned_doc)
            
            # Procesar en lotes para documentos grandes
            all_ids = []
            for i in range(0, len(cleaned_documents), batch_size):
                batch = cleaned_documents[i:i + batch_size]
                try:
                    ids = self.vectorstore.add_documents(batch)
                    all_ids.extend(ids)
                    logger.info(f"Lote {i//batch_size + 1} agregado: {len(batch)} documentos")
                except Exception as batch_error:
                    logger.warning(f"Error en lote {i//batch_size + 1}: {str(batch_error)}")
                    # Guardar información del error en metadata
                    for doc in batch:
                        if hasattr(doc, 'metadata'):
                            doc.metadata['_batch_error'] = str(batch_error)
                            doc.metadata['_batch_number'] = i//batch_size + 1
                    
                    # Intentar agregar documentos uno por uno
                    for j, doc in enumerate(batch):
                        try:
                            single_id = self.vectorstore.add_documents([doc])
                            all_ids.extend(single_id)
                            logger.info(f"Documento {j+1} del lote {i//batch_size + 1} agregado exitosamente")
                        except Exception as doc_error:
                            logger.warning(f"Error agregando documento {j+1} del lote {i//batch_size + 1}: {str(doc_error)}")
                            # Guardar información detallada del error
                            if hasattr(doc, 'metadata'):
                                doc.metadata['_individual_error'] = str(doc_error)
                                doc.metadata['_document_position'] = j+1
                                doc.metadata['_failed_processing'] = True
                            continue
            
            # Generar estadísticas de errores
            self._log_processing_stats(cleaned_documents, all_ids)
            
            logger.info(f"Total documentos agregados: {len(all_ids)}")
            return all_ids
            
        except Exception as e:
            logger.error(f"Error agregando documentos: {str(e)}")
            raise
    
    def _log_processing_stats(self, documents: List[Any], successful_ids: List[str]) -> None:
        """
        Genera estadísticas de procesamiento y errores
        
        Args:
            documents: Lista de documentos procesados
            successful_ids: Lista de IDs exitosos
        """
        try:
            total_docs = len(documents)
            successful_docs = len(successful_ids)
            failed_docs = total_docs - successful_docs
            
            # Contar tipos de errores
            cleaning_errors = 0
            batch_errors = 0
            individual_errors = 0
            
            for doc in documents:
                if hasattr(doc, 'metadata') and doc.metadata:
                    if doc.metadata.get('_cleaning_failed'):
                        cleaning_errors += 1
                    if doc.metadata.get('_batch_error'):
                        batch_errors += 1
                    if doc.metadata.get('_individual_error'):
                        individual_errors += 1
            
            # Log estadísticas
            logger.info("=" * 50)
            logger.info("ESTADÍSTICAS DE PROCESAMIENTO")
            logger.info("=" * 50)
            logger.info(f"Total documentos procesados: {total_docs}")
            logger.info(f"Documentos agregados exitosamente: {successful_docs}")
            logger.info(f"Documentos fallidos: {failed_docs}")
            logger.info(f"Tasa de éxito: {(successful_docs/total_docs)*100:.1f}%")
            
            if cleaning_errors > 0:
                logger.warning(f"Errores de limpieza: {cleaning_errors}")
            if batch_errors > 0:
                logger.warning(f"Errores de lote: {batch_errors}")
            if individual_errors > 0:
                logger.warning(f"Errores individuales: {individual_errors}")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.warning(f"Error generando estadísticas: {str(e)}")
    
    def analyze_error_chunks(self) -> None:
        """
        Analiza y muestra información detallada de chunks con errores
        """
        try:
            # Obtener todos los documentos de la colección
            collection = self.vectorstore._collection
            all_docs = collection.get()
            
            if not all_docs['ids']:
                logger.info("No hay documentos en la colección")
                return
            
            # Analizar metadata de cada documento
            error_chunks = []
            for i, (doc_id, metadata, content) in enumerate(zip(all_docs['ids'], all_docs['metadatas'], all_docs['documents'])):
                has_error = False
                error_info = {}
                
                # Verificar diferentes tipos de errores
                for key, value in metadata.items():
                    if key.startswith('_') and ('error' in key or 'failed' in key):
                        has_error = True
                        error_info[key] = value
                
                if has_error:
                    error_chunks.append({
                        'doc_id': doc_id,
                        'metadata': metadata,
                        'content_preview': content[:200] + "..." if len(content) > 200 else content,
                        'error_info': error_info,
                        'position': i + 1
                    })
            
            # Mostrar resultados
            logger.info("=" * 80)
            logger.info("ANÁLISIS DE CHUNKS CON ERRORES")
            logger.info("=" * 80)
            logger.info(f"Total documentos en colección: {len(all_docs['ids'])}")
            logger.info(f"Documentos con errores: {len(error_chunks)}")
            logger.info(f"Tasa de errores: {(len(error_chunks)/len(all_docs['ids']))*100:.1f}%")
            logger.info("=" * 80)
            
            for i, chunk in enumerate(error_chunks, 1):
                logger.info(f"\n🔍 CHUNK CON ERROR #{i}")
                logger.info(f"   📄 ID: {chunk['doc_id']}")
                logger.info(f"   📍 Posición: {chunk['position']}")
                logger.info(f"   📁 Archivo: {chunk['metadata'].get('filename', 'N/A')}")
                logger.info(f"   📚 Materia: {chunk['metadata'].get('materia', 'N/A')}")
                logger.info(f"   📝 Tipo: {chunk['metadata'].get('tipo_documento', 'N/A')}")
                logger.info(f"   🔢 Chunk: {chunk['metadata'].get('chunk_index', 'N/A')}")
                
                logger.info(f"   ❌ ERRORES DETECTADOS:")
                for error_key, error_value in chunk['error_info'].items():
                    logger.info(f"      {error_key}: {error_value}")
                
                logger.info(f"   📖 Contenido: {chunk['content_preview']}")
                logger.info("-" * 60)
            
            # Resumen de tipos de errores
            error_types = {}
            for chunk in error_chunks:
                for error_key in chunk['error_info'].keys():
                    error_types[error_key] = error_types.get(error_key, 0) + 1
            
            logger.info("\n📊 RESUMEN DE TIPOS DE ERRORES:")
            for error_type, count in error_types.items():
                logger.info(f"   {error_type}: {count} documentos")
            
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Error analizando chunks con errores: {str(e)}")
    
    def create_from_documents(self, documents: List[Any]) -> None:
        """
        Crea vector store desde documentos usando el patrón de las clases
        (Chroma.from_documents)
        
        Args:
            documents: Lista de documentos
        """
        try:
            from langchain_chroma import Chroma
            
            if not documents:
                logger.warning("No hay documentos para crear vector store")
                return
            
            # Usar el patrón exacto de las clases
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_function,
                collection_name=self.collection_name,
                persist_directory=self.persist_directory
            )
            
            logger.info(f"Vector store creado con {len(documents)} documentos usando patrón de clases")
            
        except Exception as e:
            logger.error(f"Error creando vector store desde documentos: {str(e)}")
            raise
    
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
                # Convertir filtro a formato ChromaDB
                chroma_filter = self._convert_filter_to_chroma(filter_dict)
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
                # Convertir filtro a formato ChromaDB
                chroma_filter = self._convert_filter_to_chroma(filter_dict)
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
            
            # Obtener información de embeddings
            all_data = collection.get()
            embedding_count = len(all_data['embeddings']) if all_data['embeddings'] else count
            embedding_dimension = len(all_data['embeddings'][0]) if all_data['embeddings'] and all_data['embeddings'][0] else None
            
            return {
                'collection_name': self.collection_name,
                'persist_directory': self.persist_directory,
                'document_count': count,
                'embedding_count': embedding_count,
                'embedding_dimension': embedding_dimension,
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
    
    def _convert_filter_to_chroma(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte filtros a formato compatible con ChromaDB
        
        Args:
            filter_dict: Diccionario de filtros
            
        Returns:
            Filtro en formato ChromaDB
        """
        if not filter_dict:
            return None
        
        # Si hay múltiples condiciones, usar operador $and
        if len(filter_dict) > 1:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append({key: {"$eq": value}})
            return {"$and": conditions}
        else:
            # Una sola condición
            key, value = next(iter(filter_dict.items()))
            return {key: {"$eq": value}}


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
