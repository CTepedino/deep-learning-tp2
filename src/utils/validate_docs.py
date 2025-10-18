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
    Valida la estructura de carpetas de documentos académicos y extrae metadata
    
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
        'structure_levels': defaultdict(int),
        # Estadísticas de exámenes
        'examenes': {
            'por_tipo': defaultdict(int),
            'por_año': defaultdict(int),
            'por_cuatrimestre': defaultdict(int),
            'con_tema': 0
        },
        # Estadísticas de unidades
        'unidades_detectadas': defaultdict(int),
        'archivos_con_unidad': 0
    }
    
    # Importar DocumentLoader para extraer metadata real
    try:
        import sys
        sys.path.insert(0, str(docs_dir.parent))
        from src.data_loading import DocumentLoader
        loader = DocumentLoader()
        use_metadata_extraction = True
    except:
        loader = None
        use_metadata_extraction = False
    
    # Recorrer todos los archivos
    for file_path in docs_dir.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('.'):
            stats['archivos_totales'] += 1
            
            # Analizar la estructura
            relative_path = file_path.relative_to(docs_dir)
            parts = relative_path.parts
            
            # Extraer metadata real si está disponible
            metadata = {}
            if use_metadata_extraction:
                try:
                    metadata = loader.extract_academic_metadata('', file_path)
                except:
                    pass
            
            # Validar niveles
            if len(parts) >= 1:
                materia = parts[0]
                stats['materias'].add(materia)
                stats['archivos_por_materia'][materia] += 1
                stats['structure_levels'][1] += 1
            
            if len(parts) >= 2:
                nivel2 = parts[1]
                stats['structure_levels'][2] += 1
                
                # Determinar si es unidad o tipo
                tipo_doc = metadata.get('tipo_documento', nivel2.lower())
                stats['tipos_documento'].add(tipo_doc)
                stats['archivos_por_tipo'][tipo_doc] += 1
                
                # Si es examen, extraer estadísticas específicas
                if tipo_doc in ['examenes', 'parciales', 'finales']:
                    if 'tipo_examen' in metadata:
                        stats['examenes']['por_tipo'][metadata['tipo_examen']] += 1
                    if 'año' in metadata:
                        stats['examenes']['por_año'][metadata['año']] += 1
                    if 'cuatrimestre' in metadata:
                        stats['examenes']['por_cuatrimestre'][metadata['cuatrimestre']] += 1
                    if 'tema' in metadata:
                        stats['examenes']['con_tema'] += 1
                
                # Si tiene unidad, contar
                if 'unidad_numero' in metadata:
                    stats['archivos_con_unidad'] += 1
                    stats['unidades_detectadas'][metadata['unidad_numero']] += 1
            
            if len(parts) >= 3:
                stats['structure_levels'][3] += 1
            
            if len(parts) == 1:
                issues.append(f"⚠️  Archivo sin estructura jerárquica: {file_path.name}")
            
            # Validar extensión
            if file_path.suffix.lower() not in ['.pdf', '.txt', '.tex', '.md']:
                issues.append(f"⚠️  Extensión inusual: {file_path.name}")
    
    # Convertir sets a listas y defaultdicts a dicts para serialización
    stats['materias'] = sorted(list(stats['materias']))
    stats['tipos_documento'] = sorted(list(stats['tipos_documento']))
    stats['archivos_por_materia'] = dict(stats['archivos_por_materia'])
    stats['archivos_por_tipo'] = dict(stats['archivos_por_tipo'])
    stats['structure_levels'] = dict(stats['structure_levels'])
    stats['unidades_detectadas'] = dict(sorted(stats['unidades_detectadas'].items()))
    
    # Convertir estadísticas de exámenes
    stats['examenes']['por_tipo'] = dict(stats['examenes']['por_tipo'])
    stats['examenes']['por_año'] = dict(sorted(stats['examenes']['por_año'].items(), reverse=True))
    stats['examenes']['por_cuatrimestre'] = dict(stats['examenes']['por_cuatrimestre'])
    
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
    
    # Estadísticas de exámenes
    examenes_data = stats.get('examenes', {})
    total_examenes = sum(stats['archivos_por_tipo'].get(t, 0) for t in ['examenes', 'parciales', 'finales'])
    
    if total_examenes > 0:
        print("📝 ESTADÍSTICAS DE EXÁMENES")
        print("-" * 70)
        print(f"Total de exámenes: {total_examenes}")
        
        # Por tipo
        if examenes_data.get('por_tipo'):
            print("\n  Por tipo de examen:")
            for tipo_examen, count in sorted(examenes_data['por_tipo'].items()):
                tipo_label = tipo_examen.replace('_', ' ').title()
                print(f"    • {tipo_label}: {count}")
        
        # Por año
        if examenes_data.get('por_año'):
            print("\n  Por año:")
            for año, count in examenes_data['por_año'].items():
                print(f"    • {año}: {count} examen(es)")
        
        # Por cuatrimestre
        if examenes_data.get('por_cuatrimestre'):
            print("\n  Por cuatrimestre:")
            for cuat, count in sorted(examenes_data['por_cuatrimestre'].items()):
                print(f"    • Cuatrimestre {cuat}: {count} examen(es)")
        
        # Con tema
        if examenes_data.get('con_tema', 0) > 0:
            print(f"\n  Exámenes con tema (A/B/etc): {examenes_data['con_tema']}")
        
        print()
    
    # Estadísticas de unidades
    unidades_detectadas = stats.get('unidades_detectadas', {})
    archivos_con_unidad = stats.get('archivos_con_unidad', 0)
    
    if archivos_con_unidad > 0:
        print("📚 ESTADÍSTICAS DE UNIDADES")
        print("-" * 70)
        print(f"Archivos con unidad detectada: {archivos_con_unidad}")
        
        if unidades_detectadas:
            print("\n  Unidades encontradas:")
            # Mostrar top 10 unidades más comunes
            sorted_units = sorted(unidades_detectadas.items(), key=lambda x: x[1], reverse=True)[:10]
            for unidad_num, count in sorted_units:
                print(f"    • Unidad {unidad_num}: {count} archivo(s)")
            
            if len(unidades_detectadas) > 10:
                print(f"    ... y {len(unidades_detectadas) - 10} unidades más")
        
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

