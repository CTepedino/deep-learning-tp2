#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prueba para el sistema RAG usando configuración JSON
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configurar codificación UTF-8 para Windows
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Agregar el directorio src al path
sys.path.append('src')

# Configurar logging con codificación UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_test_config(config_file: str = "test_config.json") -> Dict[str, Any]:
    """
    Carga la configuración de pruebas desde un archivo JSON
    
    Args:
        config_file: Ruta al archivo de configuración
        
    Returns:
        Diccionario con la configuración
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo de configuración {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: JSON inválido en {config_file}: {e}")
        sys.exit(1)

def run_single_test(rag_pipeline, test_config: Dict[str, Any], config_index: int) -> bool:
    """
    Ejecuta una sola prueba de configuración
    
    Args:
        rag_pipeline: Instancia del pipeline RAG
        test_config: Configuración de la prueba
        config_index: Índice de la configuración
        
    Returns:
        True si la prueba fue exitosa, False en caso contrario
    """
    print(f"\n{'='*80}")
    print(f"🧪 PRUEBA {config_index + 1}: {test_config['name']}")
    print(f"📝 {test_config['description']}")
    print(f"{'='*80}")
    
    try:
        # Obtener parámetros de la configuración
        query_params = test_config['query_params']
        search_query = test_config.get('search_query', '')
        k_context = test_config.get('k_context', 5)
        
        print(f"\n🎯 Parámetros de consulta:")
        for key, value in query_params.items():
            print(f"   - {key}: {value}")
        
        print(f"\n🔍 Consulta de búsqueda: '{search_query}'")
        print(f"📚 Chunks de contexto: {k_context}")
        
        # Generar ejercicio
        print(f"\n🔄 Generando ejercicio...")
        result = rag_pipeline.generate_exercises(
            query_params=query_params,
            k_retrieval=k_context,
            use_filters=True  # Habilitar filtros ahora que la metadata está correcta
        )
        
        if result.get('status') == 'error':
            print(f"   ❌ Error: {result.get('message', 'Error desconocido')}")
            return False
        
        # Mostrar contexto si está habilitado
        if 'context_info' in result:
            context_info = result['context_info']
            print(f"\n📚 Contexto utilizado:")
            print(f"   - Documentos recuperados: {context_info.get('documents_retrieved', 'N/A')}")
            print(f"   - Consulta de búsqueda: {context_info.get('search_query', 'N/A')}")
            print(f"   - Filtros aplicados: {context_info.get('filters_applied', 'N/A')}")
        
        # Mostrar ejercicio generado
        if 'ejercicios' in result and result['ejercicios']:
            print(f"\n✅ Ejercicio generado exitosamente!")
            ejercicio = result['ejercicios'][0]
            tipo_ejercicio = query_params.get('tipo_ejercicio', 'unknown')
            
            print(f"\n📝 EJERCICIO DE {tipo_ejercicio.upper()} GENERADO:")
            print("   " + "="*60)
            
            # Mostrar campos según el tipo de ejercicio
            if tipo_ejercicio == 'desarrollo':
                print(f"   📋 Enunciado: {ejercicio.get('enunciado', 'N/A')}")
                print(f"   🎯 Objetivos: {ejercicio.get('objetivos', 'N/A')}")
                print(f"   📝 Instrucciones: {ejercicio.get('instrucciones', 'N/A')}")
                print(f"   💡 Criterios de evaluación: {ejercicio.get('criterios_evaluacion', 'N/A')}")
                print(f"   🔧 Solución sugerida: {ejercicio.get('solucion_sugerida', 'N/A')}")
                print(f"   📚 Referencias: {ejercicio.get('referencias', 'N/A')}")
            elif tipo_ejercicio == 'multiple_choice':
                print(f"   📋 Pregunta: {ejercicio.get('pregunta', 'N/A')}")
                print(f"   🔤 Opciones:")
                for i, opcion in enumerate(ejercicio.get('opciones', []), 1):
                    print(f"      {chr(64+i)}. {opcion}")
                print(f"   ✅ Respuesta correcta: {ejercicio.get('respuesta_correcta', 'N/A')}")
                print(f"   💡 Pista: {ejercicio.get('pista', 'N/A')}")
                print(f"   🔧 Solución: {ejercicio.get('solucion', 'N/A')}")
            elif tipo_ejercicio == 'practico':
                print(f"   📋 Pregunta: {ejercicio.get('pregunta', 'N/A')}")
                print(f"   📊 Datos: {ejercicio.get('datos', 'N/A')}")
                print(f"   💡 Pista: {ejercicio.get('pista', 'N/A')}")
                print(f"   🔧 Solución: {ejercicio.get('solucion', 'N/A')}")
            elif tipo_ejercicio == 'teorico':
                print(f"   📋 Pregunta: {ejercicio.get('pregunta', 'N/A')}")
                print(f"   🔑 Conceptos clave: {ejercicio.get('conceptos_clave', 'N/A')}")
                print(f"   💡 Pista: {ejercicio.get('pista', 'N/A')}")
                print(f"   🔧 Solución: {ejercicio.get('solucion', 'N/A')}")
            else:
                # Mostrar todos los campos disponibles
                for key, value in ejercicio.items():
                    print(f"   {key}: {value}")
            
            print("   " + "="*60)
            
            # Mostrar metadata si está habilitado
            if 'metadata' in result:
                print(f"\n📊 Metadata del ejercicio:")
                metadata = result['metadata']
                print(f"   - Modelo usado: {metadata.get('modelo_usado', 'N/A')}")
                print(f"   - Chunks recuperados: {metadata.get('chunks_recuperados', 'N/A')}")
                print(f"   - Fuentes: {len(metadata.get('fuentes', []))} documentos")
                print(f"   - Materia: {metadata.get('materia', 'N/A')}")
                print(f"   - Unidad: {metadata.get('unidad', 'N/A')}")
                print(f"   - Dificultad: {metadata.get('nivel_dificultad', 'N/A')}")
        else:
            print(f"   ❌ No se generaron ejercicios")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error en la prueba: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🚀 INICIANDO PRUEBAS DEL SISTEMA RAG")
    print("=" * 80)
    
    # Cargar configuración
    print("📋 Cargando configuración de pruebas...")
    config = load_test_config()
    test_configs = config.get('test_configs', [])
    default_config = config.get('default_config', {})
    
    if not test_configs:
        print("❌ Error: No se encontraron configuraciones de prueba")
        return
    
    print(f"✅ Configuración cargada: {len(test_configs)} pruebas configuradas")
    
    # Mostrar pruebas disponibles
    print(f"\n📋 Pruebas disponibles:")
    for i, test_config in enumerate(test_configs, 1):
        print(f"   {i}. {test_config['name']} ({test_config['query_params']['tipo_ejercicio']})")
    
    # Verificar argumentos de línea de comandos
    test_indices = []
    if len(sys.argv) > 1:
        try:
            if sys.argv[1] == "all":
                test_indices = list(range(len(test_configs)))
            else:
                test_indices = [int(arg) - 1 for arg in sys.argv[1:]]
                # Validar índices
                invalid_indices = [i for i in test_indices if i < 0 or i >= len(test_configs)]
                if invalid_indices:
                    print(f"❌ Error: Índices inválidos: {[i+1 for i in invalid_indices]}")
                    print(f"   Índices válidos: 1-{len(test_configs)}")
                    return
        except ValueError:
            print("❌ Error: Los argumentos deben ser números o 'all'")
            print("Uso: python test_rag.py [all|1|2|3|...]")
            return
    else:
        # Por defecto, ejecutar solo la primera prueba
        test_indices = [0]
    
    print(f"🎯 Ejecutando pruebas: {[i+1 for i in test_indices]}")
    
    # Inicializar pipeline RAG
    print(f"\n🔧 Inicializando pipeline RAG...")
    try:
        from src.rag_pipeline import RAGPipeline
        
        # Forzar recarga de documentos con metadata correcta
        rag_pipeline = RAGPipeline(reset_collection=True)
        print("✅ Pipeline RAG inicializado correctamente")
        
        # Verificar si hay documentos cargados
        collection_info = rag_pipeline.vector_store.get_collection_info()
        doc_count = collection_info.get('document_count', 0)
        
        if doc_count == 0:
            print("📚 No hay documentos cargados. Cargando documentos...")
            result = rag_pipeline.load_materials(
                data_directory="docs",
                file_extensions=[".pdf", ".txt", ".md"]
            )
            
            if result.get('status') == 'success':
                print(f"✅ Documentos cargados: {result.get('documents_loaded', 0)}")
                print(f"✅ Chunks creados: {result.get('chunks_created', 0)}")
            else:
                print(f"❌ Error cargando documentos: {result.get('message', 'Error desconocido')}")
                return
        else:
            print(f"✅ Colección existente con {doc_count} documentos")
        
    except Exception as e:
        print(f"❌ Error inicializando pipeline: {str(e)}")
        return
    
    # Ejecutar pruebas
    print(f"\n🧪 INICIANDO PRUEBAS")
    print("=" * 80)
    
    successful_tests = 0
    total_tests = len(test_indices)
    
    for i, test_index in enumerate(test_indices):
        test_config = test_configs[test_index]
        success = run_single_test(rag_pipeline, test_config, test_index)
        if success:
            successful_tests += 1
    
    # Resumen final
    print(f"\n{'='*80}")
    print(f"📊 RESUMEN DE PRUEBAS")
    print(f"{'='*80}")
    print(f"✅ Pruebas exitosas: {successful_tests}/{total_tests}")
    print(f"❌ Pruebas fallidas: {total_tests - successful_tests}/{total_tests}")
    print(f"📈 Tasa de éxito: {(successful_tests/total_tests)*100:.1f}%")
    
    if successful_tests == total_tests:
        print(f"\n🎉 ¡Todas las pruebas pasaron exitosamente!")
    else:
        print(f"\n⚠️ Algunas pruebas fallaron. Revisa los logs para más detalles.")
    
    print(f"\n💡 Uso del script:")
    print(f"   python test_rag.py          # Ejecutar primera prueba")
    print(f"   python test_rag.py all      # Ejecutar todas las pruebas")
    print(f"   python test_rag.py 1 3 5    # Ejecutar pruebas específicas")

if __name__ == "__main__":
    main()