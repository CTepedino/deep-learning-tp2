#!/usr/bin/env python3
"""
Sistema de Evaluación RAG - Basado en RAGAS
Sistema que usa exactamente la misma metodología del notebook clase8_RAGAS.ipynb
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Cargar variables de entorno
_ = load_dotenv(find_dotenv())

# Importar RAGAS como en el notebook
from datasets import Dataset
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate

from src.rag_pipeline import create_rag_pipeline

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_banner():
    """Imprime el banner del sistema"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎓 SISTEMA DE EVALUACIÓN RAG                        ║
    ║                                                              ║
    ║        Evaluación completa con métricas personalizadas       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)


def evaluate_rag_system(
    dataset_name: str = "evaluation_dataset.json",
    max_samples: int = 3,
    output_dir: str = "./data"
):
    """
    Sistema de evaluación RAG que prueba la GENERACIÓN DE EJERCICIOS
    (no respuestas a preguntas, que es lo que realmente hace el sistema)
    
    Args:
        dataset_name: Nombre del dataset de evaluación
        max_samples: Número máximo de muestras a evaluar
        output_dir: Directorio de salida
    """
    print("\n" + "="*80)
    print("🔬 SISTEMA DE EVALUACIÓN RAG CON RAGAS")
    print("="*80)
    
    try:
        # 1. Inicializar pipeline RAG
        print("\n⏳ Inicializando pipeline RAG...")
        rag_pipeline = create_rag_pipeline(reset_collection=False)
        
        system_info = rag_pipeline.get_system_info()
        doc_count = system_info['vector_store']['document_count']
        
        print(f"✅ Pipeline inicializado")
        print(f"   - Chunks en DB: {doc_count}")
        print(f"   - Modelo LLM: {system_info['generator']['model']}")
        print(f"   - Embeddings: {system_info['vector_store']['embedding_model']}")
        
        if doc_count == 0:
            print("\n⚠️  Base de datos vacía. Por favor ejecuta el sistema RAG primero para cargar materiales.")
            return False
        
        # 2. Cargar dataset de evaluación
        print("\n⏳ Cargando dataset de evaluación...")
        dataset_path = Path(output_dir) / dataset_name
        
        if not dataset_path.exists():
            print(f"❌ Dataset no encontrado: {dataset_path}")
            return False
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        evaluation_samples = data.get('samples', data)
        
        # Limitar muestras
        if len(evaluation_samples) > max_samples:
            evaluation_samples = evaluation_samples[:max_samples]
        
        print(f"✅ Dataset cargado: {len(evaluation_samples)} muestras")
        
        # 3. Procesar muestras - GENERAR EJERCICIOS (no respuestas)
        print(f"\n🚀 Procesando {len(evaluation_samples)} muestras...")
        print("   📝 Evaluando GENERACIÓN DE EJERCICIOS (no respuestas a preguntas)")
        
        exercise_requests = []
        generated_exercises = []
        contexts = []
        
        for i, sample in enumerate(evaluation_samples, 1):
            # Usar parámetros directamente del dataset (formato correcto)
            query_params = {
                'materia': sample.get('materia', 'Probabilidad y estadística'),
                'unidad': sample.get('unidad', 'tema general'),
                'tipo_ejercicio': sample.get('tipo_ejercicio', 'multiple_choice'),
                'nivel_dificultad': sample.get('nivel_dificultad', 'intermedio'),
                'cantidad': sample.get('cantidad', 1)
            }
            
            descripcion = sample.get('descripcion', 'Ejercicio de evaluación')
            print(f"\n[{i}/{len(evaluation_samples)}] {descripcion}")
            print(f"   📚 Materia: {query_params['materia']}")
            print(f"   📖 Unidad: {query_params['unidad']}")
            print(f"   📝 Tipo: {query_params['tipo_ejercicio']}")
            print(f"   ⚡ Dificultad: {query_params['nivel_dificultad']}")
            print(f"   🔍 Filtros: ACTIVADOS")
            
            try:
                # Debug: mostrar qué filtros se van a aplicar
                print(f"   🔍 Filtros que se aplicarán:")
                print(f"      - materia: '{query_params.get('materia')}'")
                print(f"      - nivel_dificultad: '{query_params.get('nivel_dificultad')}'")
                print(f"      - tipo_ejercicio: '{query_params.get('tipo_ejercicio')}'")
                
                # GENERAR EJERCICIO usando el pipeline RAG
                result = rag_pipeline.generate_exercises(
                    query_params=query_params,
                    k_retrieval=3,
                    use_filters=True
                )
                
                # Verificar si se generaron ejercicios (sin importar el status)
                if result.get('ejercicios') and len(result.get('ejercicios', [])) > 0:
                    exercises = result['ejercicios']
                    
                    # Obtener contextos reales (no inventar)
                    context_docs = result.get('context_docs', [])
                    
                    # Solo usar contextos reales si existen
                    if context_docs and len(context_docs) > 0:
                        retrieved_contexts = [doc['content'] for doc in context_docs]
                    else:
                        # No inventar contextos - usar lista vacía
                        retrieved_contexts = []
                    
                    print(f"   📚 Contextos: {len(retrieved_contexts)}")
                    print(f"   📝 Ejercicio: {exercises[0].get('pregunta', 'Sin pregunta')[:60]}...")
                    
                    # Almacenar datos para evaluación
                    exercise_requests.append({
                        'materia': query_params['materia'],
                        'unidad': query_params['unidad'],
                        'tipo_ejercicio': query_params['tipo_ejercicio'],
                        'nivel_dificultad': query_params['nivel_dificultad'],
                        'descripcion': descripcion
                    })
                    generated_exercises.append(exercises[0])
                    contexts.append(retrieved_contexts)
                    
                    print(f"   [OK] Ejercicio generado")
                else:
                    error_msg = result.get('message', 'No se generaron ejercicios')
                    print(f"   [ERROR] {error_msg}")
                    
                    # Agregar valores por defecto
                    exercise_requests.append(query_params)
                    generated_exercises.append({
                        'pregunta': f'Error: {error_msg}',
                        'pista': 'Error en generación',
                        'solucion': 'Error en generación'
                    })
                    contexts.append([])  # Lista vacía, no contexto inventado
                
            except Exception as e:
                print(f"   [WARN] Error procesando muestra {i}: {str(e)}")
                # Agregar valores por defecto
                exercise_requests.append(query_params)
                generated_exercises.append({
                    'pregunta': f'Error al generar ejercicio: {str(e)}',
                    'pista': 'Error en generación',
                    'solucion': 'Error en generación'
                })
                contexts.append([])  # Lista vacía, no contexto inventado
        
        # 4. Crear dataset para evaluación de ejercicios
        print(f"\n📊 Creando dataset para evaluación de ejercicios...")
        
        # Convertir ejercicios generados en formato compatible con RAGAS
        questions = []
        answers = []
        
        for i, (request, exercise) in enumerate(zip(exercise_requests, generated_exercises)):
            # Crear pregunta basada en el ejercicio generado
            question = f"Ejercicio sobre {request['materia']} - {request['unidad']} ({request['tipo_ejercicio']}, {request['nivel_dificultad']})"
            questions.append(question)
            
            # Crear respuesta basada en el ejercicio generado (usar 'pregunta' en lugar de 'enunciado')
            answer = f"Pregunta: {exercise.get('pregunta', exercise.get('enunciado', 'N/A'))}\n"
            if exercise.get('opciones'):
                answer += f"Opciones: {', '.join(exercise['opciones'])}\n"
            if exercise.get('respuesta_correcta'):
                answer += f"Respuesta correcta: {exercise['respuesta_correcta']}\n"
            if exercise.get('pista'):
                answer += f"Pista: {exercise['pista']}\n"
            if exercise.get('solucion'):
                answer += f"Solución: {exercise['solucion']}"
            answers.append(answer)
        
        data_samples = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
        }
        
        dataset = Dataset.from_dict(data_samples)
        print(f"✅ Dataset creado: {len(dataset)} muestras")
        
        # 5. Evaluar con RAGAS - MÉTRICAS ESPECÍFICAS PARA EJERCICIOS
        print(f"\n🚀 Ejecutando evaluación RAGAS...")
        print(f"   - Muestras: {len(questions)}")
        print(f"   - Métricas: faithfulness, answer_relevancy")
        
        try:
            # Usar RAGAS para evaluar la calidad de los ejercicios generados
            score = evaluate(dataset, metrics=[faithfulness, answer_relevancy], raise_exceptions=False)
            results_df = score.to_pandas()
            
            print("\n" + "="*80)
            print("✅ EVALUACIÓN COMPLETADA")
            print("="*80)
            
            # Calcular estadísticas
            if not results_df.empty:
                faithfulness_mean = results_df['faithfulness'].mean()
                answer_relevancy_mean = results_df['answer_relevancy'].mean()
                
                print(f"\n📈 MÉTRICAS PROMEDIO:")
                print(f"   - Faithfulness: {faithfulness_mean:.4f}")
                print(f"   - Answer Relevancy: {answer_relevancy_mean:.4f}")
                print(f"   - Score General: {(faithfulness_mean + answer_relevancy_mean) / 2:.4f}")
                
                print(f"\n📝 ANÁLISIS DE EJERCICIOS:")
                print(f"   - Ejercicios generados: {len(generated_exercises)}")
                print(f"   - Ejercicios exitosos: {sum(1 for ex in generated_exercises if not ex.get('pregunta', '').startswith('Error'))}")
                
                # Debug: mostrar contextos reales
                print(f"\n🔍 DEBUG - Contextos recuperados:")
                for i, ctx in enumerate(contexts):
                    print(f"   [{i+1}] Contextos: {len(ctx)}")
                    if ctx and len(ctx) > 0:
                        print(f"       Primer contexto: {ctx[0][:100]}...")
                    else:
                        print(f"       Sin contextos reales")
            
            # Guardar resultados
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = Path(output_dir) / f"exercise_evaluation_{timestamp}.json"
            
            evaluation_data = {
                'timestamp': timestamp,
                'method': 'RAGAS_EXERCISE_EVALUATION',
                'metrics': ['faithfulness', 'answer_relevancy'],
                'num_samples': len(questions),
                'exercise_requests': exercise_requests,
                'generated_exercises': generated_exercises,
                'results_df': results_df.to_dict('records'),
                'summary': {
                    'faithfulness_mean': float(faithfulness_mean) if not results_df.empty else 0.0,
                    'answer_relevancy_mean': float(answer_relevancy_mean) if not results_df.empty else 0.0,
                    'overall_mean': float((faithfulness_mean + answer_relevancy_mean) / 2) if not results_df.empty else 0.0,
                    'exercises_generated': len(generated_exercises),
                    'exercises_successful': sum(1 for ex in generated_exercises if ex.get('enunciado') != 'Error al generar ejercicio')
                }
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Resultados guardados en: {results_file}")
            
            return True
            
        except Exception as e:
            print(f"\n⚠️  Error en evaluación RAGAS: {str(e)}")
            print("💡 Esto puede deberse a problemas de API o configuración")
            
            # Crear resultado por defecto
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = Path(output_dir) / f"ragas_evaluation_error_{timestamp}.json"
            
            evaluation_data = {
                'timestamp': timestamp,
                'method': 'RAGAS',
                'status': 'error',
                'error': str(e),
                'num_samples': len(questions),
                'data_samples': data_samples
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Datos guardados en: {results_file}")
            return False
        
    except Exception as e:
        logger.error(f"Error durante evaluación: {str(e)}")
        print(f"\n❌ ERROR CRÍTICO: {str(e)}")
        return False




def main():
    """Función principal"""
    print_banner()
    
    # Cargar variables de entorno
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here':
        print("\n❌ ERROR: Necesitas configurar tu OPENAI_API_KEY en el archivo .env")
        return 1
    
    # Ejecutar evaluación
    success = evaluate_rag_system(
        dataset_name="evaluation_dataset.json",
        max_samples=3,
        output_dir="./data"
    )
    
    if success:
        print("\n🎉 ¡Evaluación completada exitosamente!")
    else:
        print("\n❌ Error en la evaluación")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
