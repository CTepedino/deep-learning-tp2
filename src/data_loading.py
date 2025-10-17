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
        Extrae metadata académica desde la estructura jerárquica de carpetas
        
        Estructura esperada:
            docs/[MATERIA]/Unidad_[NN]_[TEMA]/[TIPO]/archivo.pdf
        
        Args:
            content: Contenido del documento
            file_path: Ruta del archivo
            
        Returns:
            Diccionario con metadata académica
        """
        metadata = {}
        
        # Obtener todas las partes de la ruta
        parts = file_path.parts
        
        # Intentar encontrar 'docs' en la ruta
        try:
            docs_index = parts.index('docs')
            path_after_docs = parts[docs_index + 1:]
        except ValueError:
            # Si no hay 'docs', intentar con 'data' o usar toda la ruta
            try:
                docs_index = parts.index('data')
                path_after_docs = parts[docs_index + 1:]
            except ValueError:
                # Usar los últimos niveles de la ruta
                path_after_docs = parts[-4:] if len(parts) >= 4 else parts
        
        # NIVEL 1: Materia (primera carpeta después de docs)
        if len(path_after_docs) >= 1:
            materia_raw = path_after_docs[0]
            # Convertir guiones bajos a espacios y capitalizar
            metadata['materia'] = materia_raw.replace('_', ' ')
        else:
            # Fallback: detectar materia por keywords
            metadata['materia'] = self._detect_materia_fallback(file_path)
        
        # NIVEL 2: Unidad/Tema (segunda carpeta)
        if len(path_after_docs) >= 2:
            unidad_raw = path_after_docs[1]
            
            # Extraer número de unidad (ej: "Unidad_01_Variables" -> 1)
            unit_number = self._extract_unit_number(unidad_raw)
            if unit_number is not None:
                metadata['unidad_numero'] = unit_number
            
            # Extraer tema de unidad (ej: "Unidad_01_Variables_Aleatorias" -> "Variables Aleatorias")
            unit_topic = self._extract_unit_topic(unidad_raw)
            metadata['unidad_tema'] = unit_topic
            metadata['unidad'] = unit_topic  # Alias para compatibilidad
        
        # NIVEL 3: Tipo de documento (tercera carpeta)
        if len(path_after_docs) >= 3:
            tipo_raw = path_after_docs[2]
            metadata['tipo_documento'] = tipo_raw
        else:
            # Fallback: detectar tipo desde el nombre del archivo
            metadata['tipo_documento'] = self._detect_tipo_documento_fallback(file_path)
        
        # Detectar nivel de dificultad sugerido del nombre de archivo
        filename_lower = file_path.name.lower()
        if any(word in filename_lower for word in ['basico', 'introductorio', 'intro']):
            metadata['nivel_sugerido'] = 'introductorio'
        elif 'avanzado' in filename_lower:
            metadata['nivel_sugerido'] = 'avanzado'
        elif any(word in filename_lower for word in ['intermedio', 'medio']):
            metadata['nivel_sugerido'] = 'intermedio'
        
        # Extraer palabras clave del contenido (primeras 500 caracteres)
        preview = content[:500].lower()
        keywords = []
        
        # Palabras clave comunes en matemáticas e IA
        keyword_list = [
            'distribución', 'normal', 'probabilidad', 'estadística',
            'regresión', 'correlación', 'hipótesis', 'test',
            'clustering', 'clasificación', 'red neuronal', 'machine learning',
            'algoritmo', 'optimización', 'gradiente'
        ]
        
        for keyword in keyword_list:
            if keyword in preview:
                keywords.append(keyword)
        
        if keywords:
            metadata['palabras_clave'] = keywords
        
        return metadata
    
    def _extract_unit_number(self, unit_name: str) -> Optional[int]:
        """
        Extrae el número de unidad de nombres como 'Unidad_01_Tema' -> 1
        
        Args:
            unit_name: Nombre de la carpeta de unidad
            
        Returns:
            Número de unidad o None
        """
        import re
        match = re.search(r'[Uu]nidad[_\s]?(\d+)', unit_name)
        if match:
            return int(match.group(1))
        return None
    
    def _extract_unit_topic(self, unit_name: str) -> str:
        """
        Extrae el tema de 'Unidad_01_Variables_Aleatorias' -> 'Variables Aleatorias'
        
        Args:
            unit_name: Nombre de la carpeta de unidad
            
        Returns:
            Tema de la unidad (sin el prefijo Unidad_XX)
        """
        import re
        # Remover "Unidad_XX_" del principio
        cleaned = re.sub(r'^[Uu]nidad[_\s]?\d+[_\s]?', '', unit_name)
        # Reemplazar guiones bajos por espacios
        return cleaned.replace('_', ' ')
    
    def _detect_materia_fallback(self, file_path: Path) -> str:
        """
        Detecta materia por keywords cuando no está en la estructura de carpetas
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Nombre de la materia detectada o 'No especificada'
        """
        filename_lower = file_path.name.lower()
        parent_dir = file_path.parent.name.lower()
        
        # Mapeo de materias (expandible)
        materia_keywords = {
            'probabilidad': 'Probabilidad y estadística',
            'estadistica': 'Probabilidad y estadística',
            'sia': 'Sistemas de Inteligencia Artificial',
            'inteligencia': 'Sistemas de Inteligencia Artificial',
            'ai': 'Sistemas de Inteligencia Artificial',
            'machine learning': 'Sistemas de Inteligencia Artificial',
        }
        
        for keyword, materia in materia_keywords.items():
            if keyword in filename_lower or keyword in parent_dir:
                return materia
        
        return 'No especificada'
    
    def _detect_tipo_documento_fallback(self, file_path: Path) -> str:
        """
        Detecta tipo de documento por el nombre cuando no está en la estructura
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Tipo de documento
        """
        filename_lower = file_path.name.lower()
        
        if 'apunte' in filename_lower:
            return 'apuntes'
        elif 'guia' in filename_lower:
            return 'guias'
        elif 'examen' in filename_lower:
            return 'examenes'
        elif 'parcial' in filename_lower:
            return 'parciales'
        elif 'final' in filename_lower:
            return 'finales'
        elif 'ejercicio' in filename_lower or 'practica' in filename_lower:
            return 'ejercicios'
        else:
            return 'documento'


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
