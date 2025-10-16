"""
Data Loading Module for RAG System
Soporte para carga de documentos PDF, TXT y TEX con extracción de metadata
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    DirectoryLoader, 
    TextLoader, 
    PyPDFLoader,
    UnstructuredFileLoader
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Clase para cargar documentos de diferentes formatos con metadata enriquecida
    """
    
    def __init__(self, supported_extensions: List[str] = None):
        """
        Inicializa el cargador de documentos
        
        Args:
            supported_extensions: Lista de extensiones soportadas
        """
        self.supported_extensions = supported_extensions or ['.pdf', '.txt', '.tex']
        self.loaders = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.tex': TextLoader,  # TEX se trata como texto plano
        }
    
    def load_documents_from_directory(
        self, 
        directory_path: str, 
        metadata_extractor: Optional[callable] = None
    ) -> List[Document]:
        """
        Carga todos los documentos de un directorio
        
        Args:
            directory_path: Ruta del directorio
            metadata_extractor: Función para extraer metadata personalizada
            
        Returns:
            Lista de documentos con metadata
        """
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directorio no encontrado: {directory_path}")
            return documents
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    docs = self.load_single_document(file_path, metadata_extractor)
                    documents.extend(docs)
                    logger.info(f"Documento cargado: {file_path}")
                except Exception as e:
                    logger.error(f"Error cargando {file_path}: {str(e)}")
        
        logger.info(f"Total documentos cargados: {len(documents)}")
        return documents
    
    def load_single_document(
        self, 
        file_path: Path, 
        metadata_extractor: Optional[callable] = None
    ) -> List[Document]:
        """
        Carga un documento individual
        
        Args:
            file_path: Ruta del archivo
            metadata_extractor: Función para extraer metadata personalizada
            
        Returns:
            Lista de documentos (puede ser múltiple para PDFs)
        """
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.loaders:
            raise ValueError(f"Extensión no soportada: {file_extension}")
        
        # Cargar documento
        loader = self.loaders[file_extension](str(file_path))
        documents = loader.load()
        
        # Enriquecer metadata
        for doc in documents:
            doc.metadata.update(self._extract_basic_metadata(file_path))
            
            # Aplicar extractor personalizado si se proporciona
            if metadata_extractor:
                custom_metadata = metadata_extractor(doc.page_content, file_path)
                doc.metadata.update(custom_metadata)
        
        return documents
    
    def _extract_basic_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extrae metadata básica del archivo
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Diccionario con metadata básica
        """
        return {
            'source': str(file_path),
            'filename': file_path.name,
            'file_type': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
            'parent_dir': file_path.parent.name,
        }
    
    def extract_academic_metadata(self, content: str, file_path: Path) -> Dict[str, Any]:
        """
        Extrae metadata académica específica para el contexto ITBA
        
        Args:
            content: Contenido del documento
            file_path: Ruta del archivo
            
        Returns:
            Diccionario con metadata académica
        """
        metadata = {}
        
        # Detectar materia basada en el nombre del archivo o directorio
        filename_lower = file_path.name.lower()
        parent_dir = file_path.parent.name.lower()
        
        # Mapeo de materias (expandible)
        materia_keywords = {
            'probabilidad': 'Probabilidad y estadística',
            'estadistica': 'Probabilidad y estadística',
            'sia': 'Sistemas de Inteligencia Artificial',
            'inteligencia': 'Sistemas de Inteligencia Artificial',
            'ai': 'Sistemas de Inteligencia Artificial',
        }
        
        for keyword, materia in materia_keywords.items():
            if keyword in filename_lower or keyword in parent_dir:
                metadata['materia'] = materia
                break
        
        # Detectar tipo de documento
        if 'apunte' in filename_lower:
            metadata['tipo_documento'] = 'apunte'
        elif 'guia' in filename_lower:
            metadata['tipo_documento'] = 'guia'
        elif 'examen' in filename_lower or 'parcial' in filename_lower:
            metadata['tipo_documento'] = 'examen'
        elif 'ejercicio' in filename_lower:
            metadata['tipo_documento'] = 'ejercicios'
        else:
            metadata['tipo_documento'] = 'documento'
        
        # Extraer palabras clave del contenido (primeras 500 caracteres)
        preview = content[:500].lower()
        keywords = []
        
        # Palabras clave comunes en matemáticas
        math_keywords = ['distribución', 'normal', 'probabilidad', 'estadística', 
                        'regresión', 'correlación', 'hipótesis', 'test']
        
        for keyword in math_keywords:
            if keyword in preview:
                keywords.append(keyword)
        
        if keywords:
            metadata['palabras_clave'] = keywords
        
        return metadata


def load_documents(
    directory_path: str, 
    use_academic_metadata: bool = True
) -> List[Document]:
    """
    Función de conveniencia para cargar documentos
    
    Args:
        directory_path: Ruta del directorio
        use_academic_metadata: Si usar extractor de metadata académica
        
    Returns:
        Lista de documentos cargados
    """
    loader = DocumentLoader()
    
    metadata_extractor = None
    if use_academic_metadata:
        metadata_extractor = loader.extract_academic_metadata
    
    return loader.load_documents_from_directory(directory_path, metadata_extractor)


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Cargar documentos de ejemplo
    docs = load_documents("./data/raw")
    
    for doc in docs[:2]:  # Mostrar primeros 2 documentos
        print(f"Archivo: {doc.metadata.get('filename')}")
        print(f"Materia: {doc.metadata.get('materia', 'No detectada')}")
        print(f"Tipo: {doc.metadata.get('tipo_documento', 'No detectado')}")
        print(f"Contenido (primeros 200 chars): {doc.page_content[:200]}...")
        print("-" * 50)
