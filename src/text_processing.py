"""
Text Processing Module for RAG System
Fragmentación inteligente de documentos con metadata enriquecida
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)


class AcademicTextSplitter:
    """
    Text splitter especializado para documentos académicos del ITBA
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None,
        academic_separators: bool = True
    ):
        """
        Inicializa el text splitter académico
        
        Args:
            chunk_size: Tamaño máximo de cada chunk (por defecto de CHUNK_SIZE)
            chunk_overlap: Solapamiento entre chunks (por defecto de CHUNK_OVERLAP)
            separators: Separadores personalizados
            academic_separators: Si usar separadores específicos para documentos académicos
        """
        # Leer valores de variables de entorno si no se especifican
        self.chunk_size = chunk_size if chunk_size is not None else int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else int(os.getenv('CHUNK_OVERLAP', '200'))
        
        if separators is None:
            if academic_separators:
                # Separadores optimizados para documentos académicos
                self.separators = [
                    "\n\n",  # Párrafos
                    "\n",    # Líneas
                    ". ",    # Oraciones
                    " ",     # Palabras
                    ""       # Caracteres
                ]
            else:
                # Separadores estándar
                self.separators = ["\n\n", "\n", " ", ""]
        else:
            self.separators = separators
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=self.separators,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True
        )
    
    def split_documents(
        self, 
        documents: List[Document],
        preserve_metadata: bool = True,
        add_chunk_metadata: bool = True
    ) -> List[Document]:
        """
        Fragmenta documentos en chunks más pequeños
        
        Args:
            documents: Lista de documentos a fragmentar
            preserve_metadata: Si preservar metadata original
            add_chunk_metadata: Si agregar metadata específica del chunk
            
        Returns:
            Lista de chunks de documentos
        """
        chunks = []
        
        for doc in documents:
            try:
                # Fragmentar documento
                doc_chunks = self.text_splitter.split_documents([doc])
                
                # Enriquecer metadata de cada chunk
                for i, chunk in enumerate(doc_chunks):
                    if preserve_metadata:
                        # Preservar metadata original
                        chunk.metadata.update(doc.metadata)
                    
                    if add_chunk_metadata:
                        # Agregar metadata específica del chunk
                        chunk.metadata.update(self._extract_chunk_metadata(chunk, i))
                    
                    chunks.append(chunk)
                
                logger.info(f"Documento fragmentado en {len(doc_chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error fragmentando documento: {str(e)}")
                continue
        
        logger.info(f"Total chunks generados: {len(chunks)}")
        return chunks
    
    def process_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Procesa documentos y los convierte al formato esperado por el pipeline
        
        Args:
            documents: Lista de documentos a procesar
            
        Returns:
            Lista de diccionarios con contenido y metadata
        """
        chunks = self.split_documents(documents)
        
        # Convertir a formato esperado por el pipeline
        processed_docs = []
        for chunk in chunks:
            processed_docs.append({
                'content': chunk.page_content,
                'metadata': chunk.metadata
            })
        
        return processed_docs
    
    def _extract_chunk_metadata(self, chunk: Document, chunk_index: int) -> Dict[str, Any]:
        """
        Extrae metadata específica de un chunk
        
        Args:
            chunk: Chunk del documento
            chunk_index: Índice del chunk
            
        Returns:
            Diccionario con metadata del chunk
        """
        metadata = {
            'chunk_index': chunk_index,
            'chunk_length': len(chunk.page_content),
            'start_index': chunk.metadata.get('start_index', 0)
        }
        
        # Detectar tipo de contenido del chunk
        content_lower = chunk.page_content.lower()
        
        # Detectar si contiene fórmulas matemáticas
        if self._contains_math_formulas(content_lower):
            metadata['contains_math'] = True
        
        # Detectar si contiene definiciones
        if self._contains_definitions(content_lower):
            metadata['contains_definitions'] = True
        
        # Detectar si contiene ejercicios
        if self._contains_exercises(content_lower):
            metadata['contains_exercises'] = True
        
        # Detectar nivel de dificultad aproximado
        metadata['difficulty_hint'] = self._estimate_difficulty(content_lower)
        
        return metadata
    
    def _contains_math_formulas(self, content: str) -> bool:
        """Detecta si el contenido contiene fórmulas matemáticas"""
        math_patterns = [
            r'\$.*?\$',  # LaTeX inline
            r'\\[a-zA-Z]+',  # Comandos LaTeX
            r'[a-zA-Z]\s*=\s*[0-9]',  # Variables igual a números
            r'[0-9]+\s*[+\-*/]\s*[0-9]+',  # Operaciones aritméticas
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, content):
                return True
        return False
    
    def _contains_definitions(self, content: str) -> bool:
        """Detecta si el contenido contiene definiciones"""
        definition_keywords = [
            'definición', 'definir', 'se define', 'es aquel', 'es aquella',
            'llamamos', 'denominamos', 'entendemos por'
        ]
        
        return any(keyword in content for keyword in definition_keywords)
    
    def _contains_exercises(self, content: str) -> bool:
        """Detecta si el contenido contiene ejercicios"""
        exercise_keywords = [
            'ejercicio', 'problema', 'calcular', 'determinar', 'encontrar',
            'resolver', 'demostrar', 'probar', 'verificar'
        ]
        
        return any(keyword in content for keyword in exercise_keywords)
    
    def _estimate_difficulty(self, content: str) -> str:
        """
        Estima el nivel de dificultad del contenido
        
        Args:
            content: Contenido del chunk
            
        Returns:
            Nivel de dificultad estimado
        """
        # Palabras clave por nivel de dificultad
        beginner_keywords = [
            'básico', 'simple', 'elemental', 'introducción', 'concepto',
            'definición', 'ejemplo', 'caso simple'
        ]
        
        advanced_keywords = [
            'avanzado', 'complejo', 'sofisticado', 'teorema', 'demostración',
            'corolario', 'lema', 'proposición', 'análisis', 'síntesis'
        ]
        
        beginner_count = sum(1 for keyword in beginner_keywords if keyword in content)
        advanced_count = sum(1 for keyword in advanced_keywords if keyword in content)
        
        if advanced_count > beginner_count:
            return 'avanzado'
        elif beginner_count > 0:
            return 'introductorio'
        else:
            return 'intermedio'


def split_documents(
    documents: List[Document],
    chunk_size: int = None,
    chunk_overlap: int = None,
    academic_mode: bool = True
) -> List[Document]:
    """
    Función de conveniencia para fragmentar documentos
    
    Args:
        documents: Lista de documentos a fragmentar
        chunk_size: Tamaño máximo de cada chunk (usa variable de entorno si es None)
        chunk_overlap: Solapamiento entre chunks (usa variable de entorno si es None)
        academic_mode: Si usar modo académico especializado
        
    Returns:
        Lista de chunks de documentos
    """
    # Usar variables de entorno si no se especifican
    if chunk_size is None:
        chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
    
    if chunk_overlap is None:
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
    
    splitter = AcademicTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        academic_separators=academic_mode
    )
    
    return splitter.split_documents(documents)


# Alias para compatibilidad con rag_pipeline.py
TextProcessor = AcademicTextSplitter


def analyze_chunks(chunks: List[Document]) -> Dict[str, Any]:
    """
    Analiza las características de los chunks generados
    
    Args:
        chunks: Lista de chunks a analizar
        
    Returns:
        Diccionario con estadísticas de los chunks
    """
    if not chunks:
        return {}
    
    lengths = [len(chunk.page_content) for chunk in chunks]
    
    # Contar tipos de contenido
    math_chunks = sum(1 for chunk in chunks if chunk.metadata.get('contains_math', False))
    definition_chunks = sum(1 for chunk in chunks if chunk.metadata.get('contains_definitions', False))
    exercise_chunks = sum(1 for chunk in chunks if chunk.metadata.get('contains_exercises', False))
    
    # Contar niveles de dificultad
    difficulty_counts = {}
    for chunk in chunks:
        difficulty = chunk.metadata.get('difficulty_hint', 'intermedio')
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    return {
        'total_chunks': len(chunks),
        'avg_length': sum(lengths) / len(lengths),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'math_chunks': math_chunks,
        'definition_chunks': definition_chunks,
        'exercise_chunks': exercise_chunks,
        'difficulty_distribution': difficulty_counts
    }


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Crear documento de ejemplo
    sample_doc = Document(
        page_content="""
        Definición de Distribución Normal
        
        Una variable aleatoria X sigue una distribución normal con parámetros μ y σ² 
        si su función de densidad de probabilidad es:
        
        f(x) = (1/√(2πσ²)) * e^(-(x-μ)²/(2σ²))
        
        Ejercicio: Calcular la probabilidad P(X < 2) para X ~ N(0, 1).
        
        Solución: Usando la tabla de la distribución normal estándar...
        """,
        metadata={'source': 'ejemplo.pdf', 'materia': 'Probabilidad y estadística'}
    )
    
    # Fragmentar documento
    chunks = split_documents([sample_doc], chunk_size=200, chunk_overlap=50)
    
    # Analizar chunks
    analysis = analyze_chunks(chunks)
    
    print("Análisis de chunks:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    print("\nPrimer chunk:")
    print(f"Contenido: {chunks[0].page_content}")
    print(f"Metadata: {chunks[0].metadata}")
