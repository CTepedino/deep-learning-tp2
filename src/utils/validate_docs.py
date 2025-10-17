"""
Script de validación de estructura de documentos
Verifica que la estructura de carpetas siga las convenciones recomendadas
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, List, Set
from collections import defaultdict


def validate_docs_structure(docs_path: str) -> Dict[str, Any]:
    """
    Valida la estructura de carpetas de documentos académicos
    
    Args:
        docs_path: Ruta al directorio docs
        
    Returns:
        Diccionario con resultados de validación
    """
    docs_dir = Path(docs_path)
    
    if not docs_dir.exists():
        return {
            'valid': False,
            'issues': [f"Directorio no encontrado: {docs_path}"],
            'stats': {}
        }
    
    issues = []
    stats = {
        'materias': set(),
        'unidades': set(),
        'tipos_documento': set(),
        'archivos_totales': 0,
        'archivos_por_materia': defaultdict(int),
        'archivos_por_tipo': defaultdict(int),
        'structure_levels': defaultdict(int)
    }
    
    # Recorrer todos los archivos
    for file_path in docs_dir.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            stats['archivos_totales'] += 1
            
            # Analizar la estructura
            relative_path = file_path.relative_to(docs_dir)
            parts = relative_path.parts
            
            # Validar niveles
            if len(parts) >= 1:
                materia = parts[0]
                stats['materias'].add(materia)
                stats['archivos_por_materia'][materia] += 1
                stats['structure_levels'][1] += 1
                
                # Validar nombre de materia
                if ' ' in materia:
                    issues.append(f"❌ Materia con espacios: {materia} (usa guiones bajos)")
            
            if len(parts) >= 2:
                unidad = parts[1]
                stats['unidades'].add(unidad)
                stats['structure_levels'][2] += 1
                
                # Validar formato de unidad
                if not _is_valid_unit_name(unidad):
                    issues.append(f"⚠️  Formato de unidad no estándar: {unidad}")
            
            if len(parts) >= 3:
                tipo = parts[2]
                stats['tipos_documento'].add(tipo)
                stats['archivos_por_tipo'][tipo] += 1
                stats['structure_levels'][3] += 1
                
                # Validar tipo de documento
                if not _is_valid_document_type(tipo):
                    issues.append(f"⚠️  Tipo de documento no estándar: {tipo}")
            
            if len(parts) == 1:
                issues.append(f"⚠️  Archivo sin estructura jerárquica: {file_path.name}")
            
            # Validar extensión
            if file_path.suffix.lower() not in ['.pdf', '.txt', '.tex', '.md']:
                issues.append(f"⚠️  Extensión inusual: {file_path.name}")
    
    # Convertir sets a listas para serialización
    stats['materias'] = sorted(list(stats['materias']))
    stats['unidades'] = sorted(list(stats['unidades']))
    stats['tipos_documento'] = sorted(list(stats['tipos_documento']))
    stats['archivos_por_materia'] = dict(stats['archivos_por_materia'])
    stats['archivos_por_tipo'] = dict(stats['archivos_por_tipo'])
    stats['structure_levels'] = dict(stats['structure_levels'])
    
    # Calcular nivel de cobertura
    total_files = stats['archivos_totales']
    files_with_3_levels = stats['structure_levels'].get(3, 0)
    
    if total_files > 0:
        coverage = (files_with_3_levels / total_files) * 100
        stats['coverage_3_levels'] = f"{coverage:.1f}%"
    
    # Determinar si es válido
    valid = len([i for i in issues if i.startswith('❌')]) == 0
    
    return {
        'valid': valid,
        'issues': issues,
        'stats': stats
    }


def _is_valid_unit_name(unit_name: str) -> bool:
    """
    Verifica si el nombre de unidad sigue el formato recomendado
    
    Formatos válidos:
    - Unidad_01_Tema
    - Unidad01_Tema
    - unidad_01_tema
    """
    pattern = r'^[Uu]nidad[_\s]?\d+[_\s]?.+'
    return bool(re.match(pattern, unit_name))


def _is_valid_document_type(doc_type: str) -> bool:
    """
    Verifica si el tipo de documento es estándar
    """
    standard_types = [
        'apuntes', 'ejercicios', 'guias', 'examenes',
        'parciales', 'finales', 'practicas', 'teoria',
        'laboratorio', 'proyectos'
    ]
    return doc_type.lower() in standard_types


def print_validation_report(result: Dict[str, Any]):
    """
    Imprime un reporte de validación formateado
    
    Args:
        result: Resultado de validate_docs_structure()
    """
    print("=" * 70)
    print("📋 REPORTE DE VALIDACIÓN DE ESTRUCTURA DE DOCUMENTOS")
    print("=" * 70)
    
    # Estado general
    if result['valid']:
        print("✅ Estado: VÁLIDO")
    else:
        print("❌ Estado: REQUIERE ATENCIÓN")
    
    print()
    
    # Estadísticas
    stats = result['stats']
    print("📊 ESTADÍSTICAS GENERALES")
    print("-" * 70)
    print(f"Total de archivos: {stats.get('archivos_totales', 0)}")
    print(f"Materias encontradas: {len(stats.get('materias', []))}")
    print(f"Unidades encontradas: {len(stats.get('unidades', []))}")
    print(f"Tipos de documento: {len(stats.get('tipos_documento', []))}")
    
    if 'coverage_3_levels' in stats:
        print(f"Archivos con estructura completa (3 niveles): {stats['coverage_3_levels']}")
    
    print()
    
    # Materias
    if stats.get('materias'):
        print("📚 MATERIAS DETECTADAS")
        print("-" * 70)
        for materia in stats['materias']:
            count = stats['archivos_por_materia'].get(materia, 0)
            print(f"  • {materia}: {count} archivo(s)")
        print()
    
    # Tipos de documento
    if stats.get('tipos_documento'):
        print("📄 TIPOS DE DOCUMENTO")
        print("-" * 70)
        for tipo in stats['tipos_documento']:
            count = stats['archivos_por_tipo'].get(tipo, 0)
            icon = "✅" if _is_valid_document_type(tipo) else "⚠️"
            print(f"  {icon} {tipo}: {count} archivo(s)")
        print()
    
    # Issues
    issues = result.get('issues', [])
    if issues:
        print("⚠️  ADVERTENCIAS Y PROBLEMAS")
        print("-" * 70)
        
        # Separar por tipo
        errors = [i for i in issues if i.startswith('❌')]
        warnings = [i for i in issues if i.startswith('⚠️')]
        
        if errors:
            print("\n🔴 ERRORES CRÍTICOS:")
            for issue in errors:
                print(f"  {issue}")
        
        if warnings:
            print("\n🟡 ADVERTENCIAS:")
            for issue in warnings[:10]:  # Limitar a 10 para no saturar
                print(f"  {issue}")
            
            if len(warnings) > 10:
                print(f"  ... y {len(warnings) - 10} advertencias más")
        
        print()
    else:
        print("✅ No se encontraron problemas")
        print()
    
    # Recomendaciones
    if not result['valid'] or issues:
        print("💡 RECOMENDACIONES")
        print("-" * 70)
        
        if any('espacios' in i for i in issues):
            print("  • Renombra carpetas con espacios usando guiones bajos")
        
        if any('sin estructura' in i for i in issues):
            print("  • Organiza archivos sueltos en la estructura jerárquica")
        
        if any('no estándar' in i for i in issues):
            print("  • Revisa el README.md en docs/ para convenciones de nombres")
        
        print()
    
    print("=" * 70)


def generate_structure_template(base_path: str):
    """
    Genera una estructura de carpetas de ejemplo
    
    Args:
        base_path: Ruta base donde crear la estructura
    """
    base = Path(base_path)
    
    # Estructura de ejemplo
    structure = {
        'Probabilidad_y_estadistica': {
            'Unidad_01_Variables_Aleatorias': ['apuntes', 'ejercicios', 'examenes'],
            'Unidad_02_Distribucion_Normal': ['apuntes', 'ejercicios', 'guias'],
            'Unidad_03_Regresion_Lineal': ['apuntes', 'ejercicios']
        },
        'SIA': {
            'Unidad_01_Clustering': ['apuntes', 'ejercicios', 'practicas'],
            'Unidad_02_Redes_Neuronales': ['apuntes', 'ejercicios'],
            'Unidad_03_Machine_Learning_Supervisado': ['apuntes', 'ejercicios']
        }
    }
    
    # Crear carpetas
    for materia, unidades in structure.items():
        for unidad, tipos in unidades.items():
            for tipo in tipos:
                folder_path = base / materia / unidad / tipo
                folder_path.mkdir(parents=True, exist_ok=True)
                
                # Crear un archivo .gitkeep para mantener la carpeta en git
                gitkeep = folder_path / '.gitkeep'
                gitkeep.touch()
    
    print(f"✅ Estructura de ejemplo creada en: {base_path}")
    print("\nEstructura creada:")
    for materia in structure.keys():
        print(f"  • {materia}/")
        for unidad in structure[materia].keys():
            print(f"    • {unidad}/")


if __name__ == "__main__":
    import sys
    
    # Usar argumento de línea de comandos o default
    docs_path = sys.argv[1] if len(sys.argv) > 1 else "./docs"
    
    print(f"Validando estructura en: {docs_path}\n")
    
    # Validar
    result = validate_docs_structure(docs_path)
    
    # Imprimir reporte
    print_validation_report(result)
    
    # Sugerir crear estructura si no existe
    if result['stats'].get('archivos_totales', 0) == 0:
        print("\n💡 ¿Quieres crear una estructura de ejemplo?")
        print("   Ejecuta: python -c \"from src.utils.validate_docs import generate_structure_template; generate_structure_template('./docs')\"")

