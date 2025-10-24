"""
Query Utilities Module
Utilidades para construir consultas de búsqueda optimizadas
"""

from typing import Dict
import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Normaliza texto para búsqueda robusta:
    - Convierte a minúsculas
    - Remueve acentos y tildes
    - Remueve espacios y caracteres especiales
    - Solo deja letras y números
    
    Args:
        text: Texto a normalizar
        
    Returns:
        Texto normalizado
    """
    if not text:
        return ""
    
    # Convertir a minúsculas
    text = text.lower()
    
    # Remover acentos y tildes
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Remover espacios y caracteres especiales, solo dejar letras y números
    text = re.sub(r'[^a-z0-9]', '', text)
    
    return text


def prepare_search_query(query_params: Dict[str, str]) -> str:
    """
    Construye una consulta de búsqueda concatenando campos relevantes.
    
    Args:
        query_params: Diccionario con parámetros de consulta (materia, unidad, tipo_ejercicio, etc.)
        
    Returns:
        String con la consulta de búsqueda optimizada
    """
    materia = query_params.get("materia", "")
    unidad = query_params.get("unidad", "")
    tipo_ejercicio = query_params.get("tipo_ejercicio", "")
    consulta_libre = query_params.get("consulta_libre", "")
    tipo_consulta = query_params.get("tipo_consulta", "")
    tipo_examen = query_params.get("tipo_examen", "")

    query_parts = []

    # Campos básicos
    if materia:
        query_parts.append(materia)
    if unidad:
        query_parts.append(unidad)
    if tipo_ejercicio:
        query_parts.append(tipo_ejercicio)

    # Nuevos campos del formato JSON
    if consulta_libre:
        query_parts.append(consulta_libre)
    
    # Agregar contexto según tipo de consulta
    if tipo_consulta == "evaluacion" and tipo_examen:
        query_parts.append(tipo_examen)
        # Agregar palabras clave según tipo de examen
        if tipo_examen == "primer_parcial":
            query_parts.extend(["conceptos básicos", "fundamentos"])
        elif tipo_examen == "segundo_parcial":
            query_parts.extend(["aplicaciones", "problemas intermedios"])
        elif tipo_examen == "final":
            query_parts.extend(["síntesis", "integración", "aplicaciones avanzadas"])
        elif tipo_examen == "recuperatorio":
            query_parts.extend(["repaso", "consolidación"])

    # Palabras clave según tipo de ejercicio
    if tipo_ejercicio == "multiple_choice":
        query_parts.extend(["conceptos", "definiciones"])
    elif tipo_ejercicio == "practico":
        query_parts.extend(["ejercicios", "problemas", "cálculos"])
    elif tipo_ejercicio == "desarrollo":
        query_parts.extend(["teoría", "explicaciones"])

    return " ".join(query_parts)
