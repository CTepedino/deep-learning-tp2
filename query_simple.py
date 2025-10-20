#!/usr/bin/env python3
"""
Uso Súper Simple del Sistema RAG
Solo escribe tu consulta en texto libre
"""

from src.rag_pipeline import create_rag_pipeline

print("🚀 Inicializando sistema...")
pipeline = create_rag_pipeline()

print("\n📝 Escribe tu consulta en texto libre:")
print("\nEjemplos:")
print("  'Genera 3 ejercicios de opción múltiple sobre distribución normal'")
print("  'Necesito 5 ejercicios prácticos de clustering'")
print("  'Dame preguntas teóricas sobre variables aleatorias'\n")

# Recibir consulta del usuario
user_query = input("Tu consulta: ")

if not user_query:
    print("❌ No ingresaste nada")
    exit(1)

# Procesar la consulta (el sistema interpreta automáticamente el texto)
print("\n⏳ Procesando...")
resultado = pipeline.query(user_query)

# Mostrar resultados
if resultado.get('status') == 'error':
    print(f"❌ Error: {resultado['message']}")
else:
    ejercicios = resultado.get('ejercicios', [])
    print(f"\n✅ Generados: {len(ejercicios)} ejercicio(s)\n")
    
    for i, ej in enumerate(ejercicios, 1):
        print("=" * 70)
        print(f"EJERCICIO {i}")
        print("=" * 70)
        print(f"\n{ej.get('pregunta', '')}\n")
        
        if 'opciones' in ej:
            for letra, opcion in zip(['A', 'B', 'C', 'D'], ej.get('opciones', [])):
                print(f"  {letra}) {opcion}")
            print(f"\n✅ Respuesta: {ej.get('respuesta_correcta', '')}")
        
        print(f"\n💡 {ej.get('pista', '')}")
        print(f"\n📖 {ej.get('solucion', '')}\n")

