"""
Query Utilities Module
Utilidades para construir consultas de búsqueda optimizadas
"""

from typing import Dict


def prepare_search_query(query_params: Dict[str, str]) -> str:
    """
    Construye una consulta de búsqueda concatenando campos relevantes.
    
    Args:
        query_params: Diccionario con parámetros de consulta (materia, unidad, tipo_ejercicio)
        
    Returns:
        String con la consulta de búsqueda optimizada
    """
    materia = query_params.get("materia", "")
    unidad = query_params.get("unidad", "")
    tipo_ejercicio = query_params.get("tipo_ejercicio", "")

    query_parts = []

    if materia:
        query_parts.append(materia)
    if unidad:
        query_parts.append(unidad)
    if tipo_ejercicio:
        query_parts.append(tipo_ejercicio)

    if tipo_ejercicio == "multiple_choice":
        query_parts.extend(["conceptos", "definiciones"])
    elif tipo_ejercicio == "practico":
        query_parts.extend(["ejercicios", "problemas", "cálculos"])
    elif tipo_ejercicio == "desarrollo":
        query_parts.extend(["teoría", "explicaciones"])

    return " ".join(query_parts)
