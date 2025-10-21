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
        
        print("🎓 GENERADOR DESDE CONFIGURACIÓN")
        print("=" * 40)
        print(f"📄 Archivo: {config_file}")
        print(f"📚 Materia: {params.get('materia', 'N/A')}")
        print(f"📖 Unidad: {params.get('unidad', 'N/A')}")
        print(f"🔢 Cantidad: {params.get('cantidad', 1)}")
        print(f"⚡ Dificultad: {params.get('nivel_dificultad', 'intermedio')}")
        print(f"📝 Tipo: {params.get('tipo_ejercicio', 'multiple_choice')}")
        print(f"📄 Formato: {params.get('formato', 'txt')}")
        print()
        
        # Crear pipeline RAG
        print("📦 Cargando sistema RAG...")
        rag = create_rag_pipeline()
        
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

