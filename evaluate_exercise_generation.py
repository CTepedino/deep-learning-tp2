#!/usr/bin/env python3
"""
Script para evaluar generación de ejercicios con RAGAS
Uso: python evaluate_exercise_generation.py [opciones]

Este script evalúa la calidad de los ejercicios generados por el RAG
comparándolos con ejercicios de referencia (ground truth).
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import logging

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import create_rag_pipeline
from src.evaluation import create_evaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_exercise_requests(file_path: str) -> List[Dict[str, Any]]:
    """
    Carga solicitudes de ejercicios desde archivo JSON
    
    Formato esperado:
    [
        {
            "materia": "Probabilidad y estadística",
            "tema": "Variables aleatorias",
            "tipo_ejercicio": "multiple_choice",
            "nivel_dificultad": "intermedio",
            "ground_truth_exercise": "Descripción del ejercicio ideal..." (opcional)
        },
        ...
    ]
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_results(results: Dict[str, Any], output_path: str):
    """Guarda resultados en archivo JSON"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def convert_to_serializable(obj):
        """Convierte objetos no serializables a formatos JSON"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return convert_to_serializable(obj.to_dict('records'))
        else:
            return obj
    
    with open(output_path, 'w', encoding='utf-8') as f:
        results_serializable = convert_to_serializable(results)
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Resultados guardados en: {output_path}")


def generate_exercises_with_rag(
    rag_pipeline,
    exercise_requests: List[Dict[str, Any]]
) -> Dict[str, List]:
    """
    Genera ejercicios usando el pipeline RAG
    
    Args:
        rag_pipeline: Pipeline RAG inicializado
        exercise_requests: Lista de solicitudes de ejercicios
        
    Returns:
        Diccionario con questions, answers, contexts, metadata
    """
    logger.info(f"Generando {len(exercise_requests)} ejercicios...")
    
    questions = []
    answers = []
    contexts = []
    metadata_list = []
    
    for i, request in enumerate(exercise_requests, 1):
        materia = request.get('materia', 'Sin especificar')
        tema = request.get('tema', 'Sin especificar')
        tipo = request.get('tipo_ejercicio', 'multiple_choice')
        nivel = request.get('nivel_dificultad', 'intermedio')
        
        logger.info(f"Generando ejercicio {i}/{len(exercise_requests)}: {materia} - {tema}")
        
        try:
            # Construir query params para el RAG
            query_params = {
                'materia': materia,
                'unidad': tema,  # El RAG usa 'unidad' para el tema
                'tipo_ejercicio': tipo,
                'nivel_dificultad': nivel,
                'cantidad': 1
            }
            
            # Generar ejercicio usando el RAG
            result = rag_pipeline.generate_exercises(
                query_params=query_params,
                k_retrieval=5,
                use_filters=True
            )
            
            if result.get('status') == 'error':
                logger.warning(f"Error generando ejercicio: {result.get('message')}")
                # Agregar placeholder
                questions.append(f"Generar ejercicio de {materia} sobre {tema}")
                answers.append("Error: No se pudo generar el ejercicio.")
                contexts.append(["Error en recuperación de contexto"])
                metadata_list.append(request)
                continue
            
            # Extraer el ejercicio generado
            ejercicios = result.get('ejercicios', [])
            if not ejercicios:
                logger.warning(f"No se generaron ejercicios para: {materia} - {tema}")
                questions.append(f"Generar ejercicio de {materia} sobre {tema}")
                answers.append("No se generó ningún ejercicio.")
                contexts.append(["Sin contexto"])
                metadata_list.append(request)
                continue
            
            ejercicio = ejercicios[0]
            
            # Formatear como pregunta/respuesta para RAGAS
            # La "pregunta" es la solicitud del ejercicio
            question = f"Generar un ejercicio de {tipo} de nivel {nivel} sobre {tema} en la materia {materia}"
            questions.append(question)
            
            # La "respuesta" es el ejercicio completo generado
            answer_parts = []
            answer_parts.append(f"**{ejercicio.get('titulo', 'Ejercicio')}**\n")
            answer_parts.append(ejercicio.get('enunciado', ''))
            
            if tipo == 'multiple_choice' and 'opciones' in ejercicio:
                answer_parts.append("\n\nOpciones:")
                for opt in ejercicio['opciones']:
                    answer_parts.append(f"  {opt}")
                if 'respuesta_correcta' in ejercicio:
                    answer_parts.append(f"\n\nRespuesta correcta: {ejercicio['respuesta_correcta']}")
            
            if 'solucion' in ejercicio:
                answer_parts.append(f"\n\nSolución:\n{ejercicio['solucion']}")
            
            answer = "\n".join(answer_parts)
            answers.append(answer)
            
            # Contextos recuperados
            result_metadata = result.get('metadata', {})
            fuentes = result_metadata.get('fuentes', [])
            
            # Recuperar los chunks reales que se usaron
            retrieved_contexts = []
            if fuentes:
                # Intentar recuperar contexto usando el retriever
                try:
                    docs = rag_pipeline.retriever.retrieve(
                        query=f"{materia} {tema}",
                        k=5,
                        filter_dict={'materia': materia}
                    )
                    retrieved_contexts = [doc.page_content for doc in docs]
                except Exception as e:
                    logger.warning(f"No se pudieron recuperar contextos: {str(e)}")
                    retrieved_contexts = [f"Fuente: {f}" for f in fuentes[:3]]
            
            if not retrieved_contexts:
                retrieved_contexts = ["Contexto basado en conocimiento del modelo"]
            
            contexts.append(retrieved_contexts)
            
            # Guardar metadata
            metadata_list.append({
                **request,
                'ejercicio_generado': ejercicio,
                'fuentes_usadas': fuentes[:5] if fuentes else []
            })
            
            logger.info(f"  ✓ Ejercicio generado exitosamente")
            
        except Exception as e:
            logger.error(f"Error procesando solicitud: {str(e)}")
            import traceback
            traceback.print_exc()
            
            questions.append(f"Generar ejercicio de {materia} sobre {tema}")
            answers.append("Error generando ejercicio.")
            contexts.append(["Error"])
            metadata_list.append(request)
    
    return {
        'questions': questions,
        'answers': answers,
        'contexts': contexts,
        'metadata': metadata_list
    }


def evaluate_exercise_generation(
    exercise_requests: List[Dict[str, Any]],
    output_dir: str = "./evaluation_results"
) -> Dict[str, Any]:
    """
    Evalúa el sistema de generación de ejercicios
    
    Args:
        exercise_requests: Lista de solicitudes de ejercicios
        output_dir: Directorio para guardar resultados
        
    Returns:
        Resultados de evaluación
    """
    print("\n" + "=" * 80)
    print("🚀 EVALUACIÓN DE GENERACIÓN DE EJERCICIOS CON RAGAS")
    print("=" * 80)
    
    # 1. Inicializar RAG pipeline
    print("\n📦 Inicializando pipeline RAG...")
    rag = create_rag_pipeline()
    
    # Verificar que hay datos
    info = rag.vector_store.get_collection_info()
    print(f"   ✓ Colección: {info['collection_name']}")
    print(f"   ✓ Documentos en DB: {info['document_count']}")
    print(f"   ✓ Modelo LLM: {rag.generator_model_name}")
    print(f"   ✓ Embeddings: {info['embedding_model']}")
    
    if info['document_count'] == 0:
        print("\n❌ ERROR: No hay documentos en la base de datos.")
        print("   Ejecuta: python initialize_chroma.py")
        sys.exit(1)
    
    # 2. Generar ejercicios
    print(f"\n📝 Generando {len(exercise_requests)} ejercicios...")
    test_data = generate_exercises_with_rag(
        rag_pipeline=rag,
        exercise_requests=exercise_requests
    )
    
    # 3. Mostrar resumen
    print("\n📊 Datos generados:")
    print(f"   • Solicitudes: {len(test_data['questions'])}")
    print(f"   • Ejercicios: {len(test_data['answers'])}")
    print(f"   • Sets de contextos: {len(test_data['contexts'])}")
    
    # Mostrar ejemplo
    print("\n📝 Ejemplo de ejercicio generado:")
    print(f"   Solicitud: {exercise_requests[0].get('materia')} - {exercise_requests[0].get('tema')}")
    print(f"   Tipo: {exercise_requests[0].get('tipo_ejercicio')}")
    print(f"   Ejercicio (primeros 150 chars): {test_data['answers'][0][:150]}...")
    
    # 4. Preparar ground truths si existen
    ground_truths = []
    has_ground_truth = False
    
    for req in exercise_requests:
        if 'ground_truth_exercise' in req and req['ground_truth_exercise']:
            ground_truths.append(req['ground_truth_exercise'])
            has_ground_truth = True
        else:
            ground_truths.append(None)
    
    # Solo usar ground_truths si todos tienen
    if has_ground_truth and not all(ground_truths):
        print("\n⚠️  Algunas solicitudes no tienen ground_truth, se omitirá evaluación completa")
        ground_truths = None
        has_ground_truth = False
    
    if has_ground_truth:
        print(f"\n✓ Ground truths disponibles - evaluación completa con 4 métricas")
    else:
        print(f"\n⚠️  Sin ground truths - evaluación básica con 2 métricas")
    
    # 5. Evaluar con RAGAS
    print("\n🎯 Evaluando con RAGAS...")
    evaluator = create_evaluator()
    
    available_metrics = evaluator.get_available_metrics(has_ground_truth=has_ground_truth)
    print(f"   Métricas a evaluar: {', '.join(available_metrics)}")
    
    results = evaluator.evaluate_rag(
        questions=test_data['questions'],
        answers=test_data['answers'],
        contexts=test_data['contexts'],
        ground_truths=ground_truths if has_ground_truth else None
    )
    
    if results['status'] != 'success':
        print(f"\n❌ ERROR en evaluación: {results.get('message')}")
        return results
    
    # 6. Mostrar resultados
    print("\n" + "=" * 80)
    print("📈 RESULTADOS DE EVALUACIÓN")
    print("=" * 80)
    
    stats = results['stats']
    
    print(f"\n📝 Ejercicios evaluados: {stats['num_samples']}")
    print(f"🎯 Ground truth usado: {'Sí' if stats['has_ground_truth'] else 'No'}")
    
    print("\n📊 MÉTRICAS OBTENIDAS:")
    print("-" * 80)
    
    for metric_name, scores in stats['metrics_scores'].items():
        print(f"\n🔹 {metric_name.upper().replace('_', ' ')}:")
        print(f"   • Promedio: {scores['mean']:.4f}")
        print(f"   • Desv. Estándar: {scores['std']:.4f}")
        print(f"   • Rango: [{scores['min']:.4f}, {scores['max']:.4f}]")
        
        # Interpretación
        mean_score = scores['mean']
        if mean_score >= 0.8:
            print(f"   • Evaluación: ✅ EXCELENTE")
        elif mean_score >= 0.6:
            print(f"   • Evaluación: ✓ BUENO")
        elif mean_score >= 0.4:
            print(f"   • Evaluación: ⚠️  REGULAR")
        else:
            print(f"   • Evaluación: ❌ NECESITA MEJORA")
    
    # 7. Guardar resultados
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results_file = Path(output_dir) / "exercise_generation_results.json"
    save_results(results, results_file)
    
    report_file = Path(output_dir) / "exercise_generation_report.txt"
    report = evaluator.generate_report(results, output_file=str(report_file))
    
    # Guardar ejercicios generados
    exercises_file = Path(output_dir) / "generated_exercises.json"
    with open(exercises_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("💾 ARCHIVOS GENERADOS")
    print("=" * 80)
    print(f"   • Resultados JSON: {results_file}")
    print(f"   • Reporte de texto: {report_file}")
    print(f"   • Ejercicios generados: {exercises_file}")
    
    # 8. Interpretación
    print("\n" + "=" * 80)
    print("💡 INTERPRETACIÓN PARA GENERACIÓN DE EJERCICIOS")
    print("=" * 80)
    
    print("""
🎯 Faithfulness (Fidelidad):
   Mide si los ejercicios generados se basan en el contenido real de tus documentos.
   • Alto (>0.8): Los ejercicios están bien fundamentados en el material
   • Bajo (<0.6): Los ejercicios inventan información no presente en los docs
   
🎯 Answer Relevance (Relevancia):
   Mide si los ejercicios generados son relevantes a la materia/tema solicitado.
   • Alto (>0.8): Los ejercicios son muy pertinentes al tema
   • Bajo (<0.6): Los ejercicios se desvían del tema solicitado
""")
    
    if has_ground_truth:
        print("""🎯 Context Precision (Precisión de Contexto):
   Mide si los chunks recuperados son realmente útiles para generar el ejercicio.
   • Alto (>0.8): El retrieval trae contenido muy relevante
   • Bajo (<0.6): El retrieval trae mucho "ruido"
   
🎯 Context Recall (Exhaustividad):
   Mide si se recuperó toda la información necesaria para el ejercicio.
   • Alto (>0.8): Se recuperó todo el contexto necesario
   • Bajo (<0.6): Falta información importante
""")
    
    print("=" * 80)
    print("✅ EVALUACIÓN COMPLETADA")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evalúa la generación de ejercicios del RAG con RAGAS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluación con archivo de solicitudes
  python evaluate_exercise_generation.py --requests test_exercise_generation.json
  
  # Especificar directorio de salida
  python evaluate_exercise_generation.py --requests test_exercise_generation.json --output-dir ./mis_resultados

Formato del archivo de solicitudes (JSON):
[
    {
        "materia": "Probabilidad y estadística",
        "tema": "Variables aleatorias",
        "tipo_ejercicio": "multiple_choice",
        "nivel_dificultad": "intermedio",
        "ground_truth_exercise": "Descripción ideal del ejercicio..." (opcional)
    },
    ...
]
        """
    )
    
    parser.add_argument(
        '--requests',
        type=str,
        required=True,
        help='Archivo JSON con solicitudes de ejercicios'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Directorio para guardar resultados (default: ./evaluation_results)'
    )
    
    args = parser.parse_args()
    
    # Cargar solicitudes
    print(f"📖 Cargando solicitudes desde: {args.requests}")
    exercise_requests = load_exercise_requests(args.requests)
    print(f"   ✓ {len(exercise_requests)} solicitudes cargadas")
    
    # Evaluar
    try:
        results = evaluate_exercise_generation(
            exercise_requests=exercise_requests,
            output_dir=args.output_dir
        )
        
        if results.get('status') == 'success':
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n❌ Evaluación cancelada por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error inesperado: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

