#!/usr/bin/env python3
"""
Generador de ejercicios desde archivo de configuración JSON
Uso: python generate_from_config.py config.json
"""

import json
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import create_rag_pipeline
from src.export_utils import ExerciseExporter

def main():
    if len(sys.argv) != 2:
        print("Uso: python generate_from_config.py <archivo_config.json>")
        print("Ejemplo: python generate_from_config.py config_example.json")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    try:
        # Cargar configuración
        with open(config_file, 'r', encoding='utf-8') as f:
            params = json.load(f)
        
        # Validar campos requeridos
        campos_requeridos = ['materia', 'cantidad', 'nivel_dificultad', 'tipo_ejercicio', 'formato']
        campos_faltantes = [campo for campo in campos_requeridos if campo not in params]
        
        if campos_faltantes:
            print(f"❌ Error: Faltan campos requeridos: {', '.join(campos_faltantes)}")
            print("💡 El archivo debe contener al menos: materia, cantidad, nivel_dificultad, tipo_ejercicio, formato")
            sys.exit(1)
        
        # Validar tipo de consulta si existe
        tipo_consulta = params.get('tipo_consulta', '')
        if tipo_consulta == 'evaluacion' and 'tipo_examen' not in params:
            print("❌ Error: Para tipo 'evaluacion' se requiere el campo 'tipo_examen'")
            sys.exit(1)
        elif tipo_consulta == 'unidad' and 'unidad' not in params:
            print("❌ Error: Para tipo 'unidad' se requiere el campo 'unidad'")
            sys.exit(1)
        
        print("🎓 GENERADOR DESDE CONFIGURACIÓN")
        print("=" * 40)
        print(f"📄 Archivo: {config_file}")
        print(f"📚 Materia: {params.get('materia', 'N/A')}")
        
        # Mostrar información según tipo de consulta
        tipo_consulta = params.get('tipo_consulta', '')
        if tipo_consulta == 'evaluacion':
            print(f"📚 Tipo: Evaluación")
            print(f"📝 Examen: {params.get('tipo_examen', 'N/A')}")
        elif tipo_consulta == 'unidad':
            print(f"📚 Tipo: Unidad específica")
            print(f"📖 Unidad: {params.get('unidad', 'N/A')}")
        else:
            # Formato legacy
            print(f"📖 Unidad: {params.get('unidad', 'N/A')}")
        
        # Mostrar consulta libre si existe
        consulta_libre = params.get('consulta_libre', '')
        if consulta_libre:
            print(f"💭 Consulta específica: {consulta_libre}")
        
        print(f"🔢 Cantidad: {params.get('cantidad', 1)}")
        print(f"⚡ Dificultad: {params.get('nivel_dificultad', 'intermedio')}")
        print(f"📝 Tipo: {params.get('tipo_ejercicio', 'multiple_choice')}")
        print(f"📄 Formato: {params.get('formato', 'txt')}")
        print()
        
        # Crear pipeline RAG
        print("📦 Cargando sistema RAG...")
        rag = create_rag_pipeline()
        
        # Mostrar información de debug
        print("🔍 Parámetros de consulta:")
        for key, value in params.items():
            print(f"   {key}: {value}")
        print()
        
        # Generar ejercicios
        print(f"🎲 Generando {params.get('cantidad', 1)} ejercicio(s)...")
        result = rag.generate_exercises(query_params=params)
        
        if result.get('ejercicios') and len(result.get('ejercicios', [])) > 0:
            print("✅ ¡Ejercicios generados exitosamente!")
            
            # Exportar archivos
            formato = params.get('formato', 'txt')
            print(f"💾 Exportando a formato {formato.upper()}...")
            exporter = ExerciseExporter(output_directory="./output")
            archivos = exporter.export_all_versions(
                result=result,
                format=formato
            )
            
            print(f"\n✅ Archivos generados en: {archivos['session_folder']}")
            print("   📄 Completo (todo)")
            print("   📝 Ejercicio (solo preguntas)")
            print("   💡 Pistas")
            print("   ✅ Soluciones")
            
        else:
            print(f"❌ Error: {result.get('message', 'Error desconocido')}")
            print(f"🔍 Debug - Resultado completo: {result}")
            
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {config_file}")
        print("💡 Crea un archivo de configuración basado en config_example.json")
    except json.JSONDecodeError as e:
        print(f"❌ Error en el archivo JSON: {e}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()

