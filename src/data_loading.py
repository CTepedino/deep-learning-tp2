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
            # Agregar metadata básica
            doc.metadata.update(self._extract_basic_metadata(file_path))
            
            # Aplicar extractor personalizado si se proporciona
            if metadata_extractor:
                custom_metadata = metadata_extractor(doc.page_content, file_path)
                doc.metadata.update(custom_metadata)
            
            # Limpiar metadata no deseada al final
            doc.metadata = self._clean_metadata(doc.metadata)
        
        return documents
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Limpia metadata eliminando campos no deseados del PDF
        Mantiene campos relevantes según el tipo de documento
        
        Args:
            metadata: Metadata original del documento
            
        Returns:
            Metadata limpia sin campos no deseados
        """
        # Campos a eliminar (metadata técnica del PDF no relevante para el sistema académico)
        fields_to_remove = {
            # Metadata técnica del PDF
            'creator',
            'author', 
            'title',
            'producer',
            'creationdate',
            'moddate',
            'keywords',
            'subject',
            'trapped',
            'ptex.fullbanner',
            'difficulty_hint'  # Campo generado automáticamente que no queremos
        }
        
        # Crear nueva metadata sin los campos no deseados
        cleaned_metadata = {}
        for key, value in metadata.items():
            if key not in fields_to_remove:
                cleaned_metadata[key] = value
        
        # Verificar que los campos importantes estén presentes según el tipo de documento
        tipo_doc = cleaned_metadata.get('tipo_documento', '')
        
        # Campos básicos que siempre deben estar
        basic_fields = ['materia', 'tipo_documento', 'filename', 'source', 'file_type']
        
        # Campos específicos según tipo de documento
        if tipo_doc in ['examenes', 'parciales', 'finales']:
            # Para exámenes: año, cuatrimestre, tipo_examen, tema
            exam_fields = ['año', 'cuatrimestre', 'tipo_examen', 'tema']
            expected_fields = basic_fields + exam_fields
        elif tipo_doc in ['apuntes', 'teoricas']:
            # Para teóricas: unidad_tema, unidad_numero
            theory_fields = ['unidad_tema', 'unidad_numero']
            expected_fields = basic_fields + theory_fields
        elif tipo_doc in ['guias', 'ejercicios']:
            # Para guías: puede tener unidad o no
            guide_fields = ['unidad_tema', 'unidad_numero']
            expected_fields = basic_fields + guide_fields
        else:
            # Para otros tipos
            expected_fields = basic_fields
        
        # Log de campos faltantes (para debugging)
        missing_fields = [field for field in expected_fields if field not in cleaned_metadata]
        if missing_fields:
            logger.debug(f"Campos faltantes para {tipo_doc}: {missing_fields}")
        
        return cleaned_metadata

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
        
        Estructuras soportadas:
            1. Estructura de 2 niveles (actual):
               docs/[MATERIA]/[TIPO]/archivo.pdf
               Ejemplo: docs/Sistemas de Inteligencia Artificial/Guias/archivo.pdf
               
            2. Estructura de 3 niveles (con unidades):
               docs/[MATERIA]/Unidad_[NN]_[TEMA]/[TIPO]/archivo.pdf
               Ejemplo: docs/Probabilidad_y_estadistica/Unidad_01_Variables/apuntes/teoria.pdf
        
        Args:
            content: Contenido del documento
            file_path: Ruta del archivo
            
        Returns:
            Diccionario con metadata académica extraída de la ruta y contenido
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
        
        # NIVEL 2: Puede ser Unidad O Tipo de documento
        if len(path_after_docs) >= 2:
            level2_raw = path_after_docs[1]
            
            # Detectar si es una Unidad (contiene "Unidad" o números) o un Tipo de documento
            if self._is_unit_folder(level2_raw):
                # Es una carpeta de unidad (estructura de 3 niveles)
                unit_number = self._extract_unit_number(level2_raw)
                if unit_number is not None:
                    metadata['unidad_numero'] = unit_number
                
                unit_topic = self._extract_unit_topic(level2_raw)
                metadata['unidad_tema'] = unit_topic
                
                # NIVEL 3: Tipo de documento (cuando hay 3 niveles)
                if len(path_after_docs) >= 3:
                    tipo_raw = path_after_docs[2]
                    metadata['tipo_documento'] = self._normalize_tipo_documento(tipo_raw)
                else:
                    # Fallback: detectar tipo desde el nombre del archivo
                    metadata['tipo_documento'] = self._detect_tipo_documento_fallback(file_path)
            else:
                # Es una carpeta de tipo de documento (estructura de 2 niveles)
                # Ejemplo: docs/Sistemas de Inteligencia Artificial/Guias/archivo.pdf
                metadata['tipo_documento'] = self._normalize_tipo_documento(level2_raw)
                
                # No hay unidad explícita, intentar detectar del nombre de archivo
                unit_from_file = self._extract_unit_from_filename(file_path.name)
                if unit_from_file:
                    # Extraer número de unidad (ej: "Unidad 18" -> 18)
                    import re
                    unit_match = re.search(r'Unidad (\d+)', unit_from_file)
                    if unit_match:
                        metadata['unidad_numero'] = int(unit_match.group(1))
                
                # Intentar extraer el tema de la unidad del nombre de archivo
                unit_topic_from_file = self._extract_unit_topic_from_filename(file_path.name)
                if unit_topic_from_file:
                    metadata['unidad_tema'] = unit_topic_from_file
        else:
            # Solo hay materia, detectar tipo del archivo
            metadata['tipo_documento'] = self._detect_tipo_documento_fallback(file_path)
        
        # Detectar nivel de dificultad sugerido del nombre de archivo
        filename_lower = file_path.name.lower()
        if any(word in filename_lower for word in ['basico', 'introductorio', 'intro']):
            metadata['nivel_sugerido'] = 'introductorio'
        elif 'avanzado' in filename_lower:
            metadata['nivel_sugerido'] = 'avanzado'
        elif any(word in filename_lower for word in ['intermedio', 'medio']):
            metadata['nivel_sugerido'] = 'intermedio'
        
        # Metadata especial para exámenes
        tipo_doc = metadata.get('tipo_documento', '')
        if tipo_doc in ['examenes', 'parciales', 'finales']:
            exam_metadata = self._extract_exam_metadata(file_path.name)
            metadata.update(exam_metadata)
            # Limpiar campos de unidad que no aplican a exámenes
            metadata.pop('unidad_numero', None)
            metadata.pop('unidad_tema', None)
        
        return metadata
    
    def _extract_exam_metadata(self, filename: str) -> Dict[str, Any]:
        """
        Extrae metadata específica de exámenes
        
        Detecta:
        - tipo_examen: primer_parcial, segundo_parcial, final, recuperatorio
        - año: 2023, 2022, etc.
        - cuatrimestre: 1 o 2
        - tema: A, B, 1, 2
        
        Ejemplos:
        - "2023_Q1_2P_A.pdf" → año: 2023, cuatrimestre: 1, tipo: segundo_parcial, tema: A
        - "Parcial1_2022_cuat2_tema1.pdf" → tipo: primer_parcial, año: 2022, cuatrimestre: 2, tema: 1
        - "recup2019_ks_cuat2_bis_p1.pdf" → tipo: recuperatorio, año: 2019, cuatrimestre: 2
        
        Args:
            filename: Nombre del archivo de examen
            
        Returns:
            Diccionario con metadata del examen
        """
        import re
        metadata = {}
        filename_lower = filename.lower()
        
        # Detectar año (4 dígitos entre 2000-2099)
        year_match = re.search(r'(20\d{2})', filename)
        if year_match:
            metadata['año'] = int(year_match.group(1))
        
        # Detectar cuatrimestre
        # Patrones: Q1, Q2, cuat1, cuat2, cuatrimestre1, etc.
        cuatrimestre_patterns = [
            (r'q[_\s]?([12])', 1),
            (r'cuat[rimestre]*[_\s]?([12])', 1),
            (r'c([12])', 1),
        ]
        for pattern, group in cuatrimestre_patterns:
            match = re.search(pattern, filename_lower)
            if match:
                metadata['cuatrimestre'] = int(match.group(group))
                break
        
        # Detectar tipo de examen
        if any(word in filename_lower for word in ['recup', 'recuperatorio', 'recuperacion']):
            metadata['tipo_examen'] = 'recuperatorio'
        elif any(word in filename_lower for word in ['final']):
            metadata['tipo_examen'] = 'final'
        elif 'parcial' in filename_lower or 'p' in filename_lower:
            # Detectar si es primer o segundo parcial
            # Patrones: 1P, 2P, P1, P2, parcial1, parcial2, 1er_parcial, 2do_parcial
            if re.search(r'(?:^|[_\s])([12])[_\s]?p(?:[_\s]|$)', filename_lower):
                match = re.search(r'(?:^|[_\s])([12])[_\s]?p(?:[_\s]|$)', filename_lower)
                parcial_num = int(match.group(1))
            elif re.search(r'p[_\s]?([12])(?:[_\s]|$)', filename_lower):
                match = re.search(r'p[_\s]?([12])(?:[_\s]|$)', filename_lower)
                parcial_num = int(match.group(1))
            elif re.search(r'parcial[_\s]?([12])', filename_lower):
                match = re.search(r'parcial[_\s]?([12])', filename_lower)
                parcial_num = int(match.group(1))
            elif re.search(r'([12])[erdo]*[_\s]?parcial', filename_lower):
                match = re.search(r'([12])[erdo]*[_\s]?parcial', filename_lower)
                parcial_num = int(match.group(1))
            else:
                parcial_num = 1  # Por defecto primer parcial
            
            if parcial_num == 1:
                metadata['tipo_examen'] = 'primer_parcial'
            elif parcial_num == 2:
                metadata['tipo_examen'] = 'segundo_parcial'
        
        # Detectar tema (A, B, C, D o 1, 2, 3, 4)
        # Patrones: tema_A, temaA, _A, tema1, tema_1
        tema_patterns = [
            r'tema[_\s]?([A-Da-d1-4])',
            r'[_\s]([A-D])(?:\.|$)',  # _A.pdf
            r'tema[_\s]?([1-4])',
        ]
        for pattern in tema_patterns:
            match = re.search(pattern, filename)
            if match:
                tema = match.group(1).upper()
                metadata['tema'] = tema
                break
        
        return metadata
    
    def _is_unit_folder(self, folder_name: str) -> bool:
        """
        Determina si una carpeta es una carpeta de unidad o de tipo de documento
        
        Args:
            folder_name: Nombre de la carpeta
            
        Returns:
            True si parece ser una carpeta de unidad
        """
        import re
        folder_lower = folder_name.lower()
        
        # Detectar patrones de unidad
        unit_patterns = [
            r'unidad[_\s]?\d+',  # Unidad_01, Unidad 1, unidad01
            r'tema[_\s]?\d+',     # Tema_01, Tema 1
            r'capitulo[_\s]?\d+', # Capitulo_01
            r'modulo[_\s]?\d+',   # Modulo_01
        ]
        
        for pattern in unit_patterns:
            if re.search(pattern, folder_lower):
                return True
        
        # Si empieza con solo números, probablemente es unidad
        if re.match(r'^\d+[_\s]', folder_name):
            return True
        
        return False
    
    def _normalize_tipo_documento(self, tipo_raw: str) -> str:
        """
        Normaliza el tipo de documento a un formato estándar
        
        Args:
            tipo_raw: Tipo raw desde la carpeta
            
        Returns:
            Tipo normalizado
        """
        tipo_lower = tipo_raw.lower()
        
        # Mapeo de variantes a nombres estándar
        tipo_mapping = {
            'apunte': 'apuntes',
            'apuntes': 'apuntes',
            'teorica': 'apuntes',
            'teoricas': 'apuntes',
            'teoria': 'apuntes',
            'ejercicio': 'ejercicios',
            'ejercicios': 'ejercicios',
            'practica': 'ejercicios',
            'practicas': 'ejercicios',
            'guia': 'guias',
            'guias': 'guias',
            'examen': 'examenes',
            'examenes': 'examenes',
            'parcial': 'parciales',
            'parciales': 'parciales',
            'final': 'finales',
            'finales': 'finales',
            'laboratorio': 'laboratorios',
            'laboratorios': 'laboratorios',
            'proyecto': 'proyectos',
            'proyectos': 'proyectos'
        }
        
        return tipo_mapping.get(tipo_lower, tipo_raw)
    
    def _extract_unit_from_filename(self, filename: str) -> Optional[str]:
        """
        Intenta extraer el número o tema de unidad del nombre de archivo
        
        Detecta patrones como:
        - "18 - Intervalos de Confianza.pdf" → "Unidad 18"
        - "Unidad 5 - Tema.pdf" → "Unidad 5"
        - "tema_3_clustering.pdf" → "Unidad 3"
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Número de unidad si se detecta, None si no
        """
        import re
        filename_lower = filename.lower()
        
        # Patrón 1: Número al inicio seguido de guion o espacio
        # Ejemplo: "18 - Intervalos.pdf", "5-Tema.pdf"
        match = re.match(r'^(\d+)\s*[-_\s]', filename)
        if match:
            return f"Unidad {match.group(1)}"
        
        # Patrón 2: Palabras clave con número
        # Ejemplo: "unidad 1", "tema 2", "capitulo 3"
        patterns = [
            r'unidad[_\s]?(\d+)',
            r'tema[_\s]?(\d+)',
            r'capitulo[_\s]?(\d+)',
            r'cap[_\s]?(\d+)',
            r'u(\d+)[_\s]',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, filename_lower)
            if match:
                return f"Unidad {match.group(1)}"
        
        return None
    
    def _extract_unit_topic_from_filename(self, filename: str) -> Optional[str]:
        """
        Extrae el tema de la unidad del nombre de archivo
        
        Ejemplo: "18 - Intervalos de Confianza para la Media.pdf" 
                 → "Intervalos de Confianza para la Media"
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            Tema extraído o None
        """
        import re
        
        # Quitar extensión
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Patrón: Número al inicio seguido de guion/espacio y texto
        # Ejemplo: "18 - Intervalos de Confianza.pdf"
        match = re.match(r'^\d+\s*[-_\s]+(.+)$', name_without_ext)
        if match:
            topic = match.group(1).strip()
            # Limpiar caracteres extra
            topic = re.sub(r'\s+', ' ', topic)  # Normalizar espacios
            return topic
        
        # Si no tiene número al inicio pero tiene "Unidad X -" o similar
        match = re.match(r'^(?:unidad|tema|capitulo)[_\s]?\d+\s*[-_\s]+(.+)$', 
                        name_without_ext, re.IGNORECASE)
        if match:
            topic = match.group(1).strip()
            topic = re.sub(r'\s+', ' ', topic)
            return topic
        
        return None
    
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
