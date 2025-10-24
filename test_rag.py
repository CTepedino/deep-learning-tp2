#!/usr/bin/env python3
"""
Script simple para probar el sistema RAG y generar ejercicios
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Agregar el directorio raíz al path para que funcionen las importaciones
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import create_rag_pipeline
from src.export_utils import export_exercises

def main():
    print("\n" + "=" * 70)
    print("SISTEMA RAG - GENERADOR DE EJERCICIOS ACADEMICOS")
    print("=" * 70)
    print("Demo interactiva del pipeline completo")
    print("Este script muestra cada paso del proceso RAG:")
    print("  1. Configuracion y variables de entorno")
    print("  2. Inicializacion del pipeline")
    print("  3. Estado de la base de datos vectorial")
    print("  4. Generacion de ejercicios con contexto")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("PASO 1: CONFIGURACION")
    print("=" * 70)
    print("\nCargando variables de entorno desde .env...")
    load_dotenv()
    
    env_file = Path('.env')
    if not env_file.exists():
        print("Archivo .env no encontrado. Usando env.example")
        load_dotenv('env.example')
    
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here':
        print("ERROR: Necesitas configurar tu OPENAI_API_KEY en el archivo .env")
        print("   1. Copia env.example a .env: cp env.example .env")
        print("   2. Edita .env y agrega tu API key de OpenAI")
        return
    
    print("Variables de entorno cargadas:")
    print(f"   - OPENAI_API_KEY: {'*' * 20}{os.getenv('OPENAI_API_KEY')[-4:]}")
    print(f"   - EMBEDDING_MODEL: {os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')}")
    print(f"   - LLM_MODEL: {os.getenv('LLM_MODEL', 'gpt-4o-mini')}")
    print(f"   - CHUNK_SIZE: {os.getenv('CHUNK_SIZE', '1000')}")
    
    print("\n" + "=" * 70)
    print("PASO 2: INICIALIZACION DEL PIPELINE")
    print("=" * 70)
    print("\nCreando componentes del sistema RAG...")
    
    rag_pipeline = create_rag_pipeline(
        reset_collection=False
    )
    
    system_info = rag_pipeline.get_system_info()
    embedding_model = system_info['vector_store']['embedding_model']
    doc_count = system_info['vector_store']['document_count']
    
    print(f"\nPipeline inicializado con exito")
    print(f"\nComponentes creados:")
    print(f"   1. Vector Store:")
    print(f"      - Tipo: {system_info['vector_store']['type']}")
    print(f"      - Coleccion: {system_info['vector_store']['collection_name']}")
    print(f"      - Embeddings: {embedding_model}")
    print(f"   2. Retriever:")
    print(f"      - k (docs a recuperar): {system_info['retriever']['k']}")
    print(f"      - Score threshold: {system_info['retriever']['score_threshold']}")
    print(f"   3. Generator:")
    print(f"      - Modelo LLM: {system_info['generator']['model']}")
    print(f"   4. Text Processor:")
    print(f"      - Chunk size: {system_info['text_processing']['chunk_size']}")
    print(f"      - Chunk overlap: {system_info['text_processing']['chunk_overlap']}")
    
    # 3. Verificar si hay documentos cargados
    print("\n" + "=" * 70)
    print(" PASO 3: ESTADO DE LA BASE DE DATOS")
    print("=" * 70)
    
    print(f"\n Documentos en el sistema: {doc_count}")
    
    print("\n Cargando todos los materiales...")
    print("   (Esto puede tomar unos minutos la primera vez)")
    
    # Cargar todos los materiales desde docs
    result = rag_pipeline.load_materials(
        data_directory="./docs",
        skip_on_error=True  # Continuar si hay errores
    )
    
    # Manejar diferentes estados
    status = result.get('status')
    
    if status == 'error' and not result.get('can_continue'):
        print(f" Error crítico: {result.get('message')}")
        return
    elif status == 'partial_error':
        print(f"\n  Error parcial al cargar nuevos documentos:")
        print(f"   {result.get('message')}")
        print(f"\n Continuando con {result.get('existing_documents', 0)} documentos existentes")
    elif status == 'warning':
        print(f"\n  {result.get('message')}")
        print(f"   Usando documentos existentes para generar ejercicios")
    elif status == 'success':
        print(f"\n Carga completada:")
        print(f"   - Documentos procesados: {result['documents_loaded']}")
        print(f"   - Chunks generados: {result['chunks_created']}")
        print(f"   - IDs asignados: {result['documents_added']}")
    
    # 3.1 Mostrar ejemplos de chunks
    print("\n Mostrando ejemplos de chunks almacenados...")
    sample_results = rag_pipeline.search_materials(
        query="probabilidad variables aleatorias",
        k=3
    )
    
    if sample_results:
        for i, doc in enumerate(sample_results[:2], 1):
            print(f"\n Chunk de ejemplo {i}:")
            content_preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            print(f"   Contenido: {content_preview}")
            print(f"   Metadata:")
            for key, value in doc['metadata'].items():
                if key not in ['page', 'start_index']:  # Omitir metadata menos relevante
                    print(f"      - {key}: {value}")
    
    # 3.2 Analizar metadata
    print("\n Análisis de metadata...")
    all_docs = rag_pipeline.search_materials(query="", k=100)
    
    if all_docs:
        materias = set()
        tipos_doc = set()
        difficulty_levels = set()
        
        for doc in all_docs:
            metadata = doc['metadata']
            if 'materia' in metadata:
                materias.add(metadata['materia'])
            if 'tipo_documento' in metadata:
                tipos_doc.add(metadata['tipo_documento'])
            if 'difficulty_hint' in metadata:
                difficulty_levels.add(metadata['difficulty_hint'])
        
        print(f"   - Materias detectadas: {', '.join(materias) if materias else 'N/A'}")
        print(f"   - Tipos de documento: {', '.join(tipos_doc) if tipos_doc else 'N/A'}")
        print(f"   - Niveles de dificultad: {', '.join(difficulty_levels) if difficulty_levels else 'N/A'}")
    
    # 4. Generar ejercicio de prueba
    print("\n" + "=" * 70)
    print(" PASO 4: GENERACIÓN DE EJERCICIOS")
    print("=" * 70)
    
    query_params = {
        "materia": "Probabilidad y estadística",
        "unidad": "Variables Aleatorias",
        "cantidad": 1,
        "nivel_dificultad": "intermedio",
        "tipo_ejercicio": "multiple_choice",
        "formato": "txt"  # Puede ser: 'txt', 'pdf', 'tex'
    }
    
    print(f"\n Parámetros de generación:")
    for key, value in query_params.items():
        print(f"   - {key}: {value}")
    
    # 4.1 Primero mostrar qué documentos se recuperarían
    print("\n Recuperando contexto relevante...")
    from src.query_utils import prepare_search_query
    search_query = prepare_search_query(query_params)
    print(f"   Query de búsqueda: '{search_query}'")
    
    context_preview = rag_pipeline.search_materials(
        query=search_query,
        k=5
    )
    
    if context_preview:
        print(f"\n Se recuperaron {len(context_preview)} chunks de contexto:")
        for i, doc in enumerate(context_preview[:3], 1):
            content_snippet = doc['content'][:100].replace('\n', ' ') + "..."
            source = doc['metadata'].get('source', 'desconocido')
            print(f"   {i}. [{source}] {content_snippet}")
    
    # 4.2 Generar ejercicio
    print("\n Generando ejercicio con LLM...")
    result = rag_pipeline.generate_exercises(
        query_params=query_params,
        k_retrieval=5
    )
    
    # Verificar si hubo error
    if result.get('status') == 'error':
        print(f" Error: {result.get('message')}")
        return
    
    # 5. Mostrar resultados
    print("\n" + "=" * 70)
    print(" EJERCICIO GENERADO")
    print("=" * 70)
    
    if 'ejercicios' not in result or not result['ejercicios']:
        print(" No se generaron ejercicios")
        return
    
    for i, exercise in enumerate(result['ejercicios'], 1):
        print(f"\n┌{'─' * 68}┐")
        print(f"│  EJERCICIO {i}")
        print(f"└{'─' * 68}┘")
        print(f"\n{exercise['pregunta']}")
        print(f"\nOpciones:")
        opciones_letras = ['A', 'B', 'C', 'D']
        for j, opcion in enumerate(exercise['opciones']):
            print(f"  {opciones_letras[j]}) {opcion}")
        
        print(f"\n Respuesta correcta: {exercise['respuesta_correcta']}")
        
        if 'pista' in exercise and exercise['pista']:
            print(f"\n Pista: {exercise['pista']}")
        
        if 'solucion' in exercise and exercise['solucion']:
            print(f"\n Solución: {exercise['solucion']}")
    
    # Metadata del proceso
    print("\n" + "=" * 70)
    print(" METADATA DEL PROCESO DE GENERACIÓN")
    print("=" * 70)
    
    metadata = result.get('metadata', {})
    context_info = result.get('context_info', {})
    
    print(f"\n Parámetros usados:")
    print(f"   - Materia: {metadata.get('materia', 'N/A')}")
    print(f"   - Unidad: {metadata.get('unidad', 'N/A')}")
    print(f"   - Tipo: {metadata.get('tipo_ejercicio', 'N/A')}")
    print(f"   - Dificultad: {metadata.get('nivel_dificultad', 'N/A')}")
    
    print(f"\n Proceso de retrieval:")
    print(f"   - Query de búsqueda: '{context_info.get('search_query', 'N/A')}'")
    print(f"   - Chunks recuperados: {metadata.get('chunks_recuperados', 0)}")
    print(f"   - Filtros aplicados: {context_info.get('filters_applied', 'Ninguno')}")
    
    print(f"\n Generación:")
    print(f"   - Modelo LLM: {metadata.get('modelo_usado', 'N/A')}")
    
    if 'fuentes' in metadata and metadata['fuentes']:
        print(f"\n Fuentes utilizadas:")
        unique_sources = list(set(metadata['fuentes']))[:5]
        for source in unique_sources:
            source_name = source.split('/')[-1] if '/' in source else source
            print(f"   - {source_name}")
        if len(unique_sources) > 5:
            print(f"   ... y {len(unique_sources) - 5} más")
    
    print("\n" + "=" * 70)
    print(" ¡DEMO COMPLETADA EXITOSAMENTE!")
    print("=" * 70)
    
    # Exportar ejercicios
    print("\n" + "=" * 70)
    print(" EXPORTACIÓN DE EJERCICIOS")
    print("=" * 70)
    
    print("\n Exportando ejercicios a archivos...")
    
    # Obtener formato del query_params o usar txt por defecto
    formato_exportacion = query_params.get('formato', 'txt')
    print(f"   Formato seleccionado: {formato_exportacion.upper()}")
    
    try:
        from src.export_utils import ExerciseExporter
        
        exporter = ExerciseExporter(output_directory="./output")
        
        # Exportar TODAS las versiones (completo, ejercicio, pistas, soluciones)
        archivos = exporter.export_all_versions(
            result=result,
            format=formato_exportacion
        )
        
        print(f"\n Se generaron {len(archivos) - 1} archivos en formato {formato_exportacion.upper()}:\n")
        print(f"   1.  Completo (todo):        {archivos['completo']}")
        print(f"   2.  Ejercicio (sin ayuda):  {archivos['ejercicio']}")
        print(f"   3.  Pistas:                 {archivos['pistas']}")
        print(f"   4.  Soluciones:             {archivos['soluciones']}")
        
        print(f"\n Carpeta de esta sesión: {archivos['session_folder']}")
        print(f"   Los archivos están organizados por timestamp para no pisarse")
        
    except ImportError as e:
        if 'reportlab' in str(e) and formato_exportacion == 'pdf':
            print(f"     Para exportar PDF necesitas: pip install reportlab")
        else:
            print(f"    Error de importación: {str(e)}")
    except Exception as e:
        print(f"    Error durante exportación: {str(e)}")
    
    # Opciones adicionales
    print("\n" + "=" * 70)
    print(" PRÓXIMOS PASOS")
    print("=" * 70)
    print("\n Para generar más ejercicios:")
    print("   - Modificar los query_params en este script")
    print("   - Probar diferentes tipos: 'desarrollo', 'practico', 'teorico'")
    print("   - Cambiar la materia a 'Sistemas de Inteligencia Artificial'")
    print("   - Ajustar la dificultad: 'basico', 'intermedio', 'avanzado'")
    
    print("\n Formatos de exportación disponibles:")
    print("   - 'txt': Texto plano (fácil de editar)")
    print("   - 'tex': LaTeX (para documentos profesionales)")
    print("   - 'pdf': PDF (listo para imprimir)")
    print("   Cambia en query_params: 'formato': 'txt'|'pdf'|'tex'")
    
    print("\n Estructura de archivos generados:")
    print("   Cada ejecución crea una carpeta con timestamp en ./output/")
    print("   Dentro encontrarás 4 archivos:")
    print("   1. _completo: Todo el ejercicio (para docentes)")
    print("   2. _ejercicio: Solo preguntas (para estudiantes)")
    print("   3. _pistas: Solo las pistas (ayuda intermedia)")
    print("   4. _soluciones: Solo las soluciones (corrección)")
    
    print("\n Consejos:")
    print("   - Usa .txt para edición rápida")
    print("   - Usa .tex para documentos académicos formales")
    print("   - Usa .pdf para distribución directa (requiere reportlab)")

if __name__ == "__main__":
    main()
