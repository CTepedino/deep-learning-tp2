#!/usr/bin/env python3
"""
Script para evaluar sistema RAG con RAGAS
Uso: python evaluate_with_ragas.py [opciones]

Métricas evaluadas:
- Faithfulness: ¿La respuesta es fiel al contexto?
- Answer Relevance: ¿La respuesta es relevante a la pregunta?
- Context Precision: ¿Los contextos recuperados son precisos?
- Context Recall: ¿Se recuperó todo el contexto necesario?
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


def load_test_questions(file_path: str) -> List[Dict[str, Any]]:
    """
    Carga preguntas de prueba desde archivo JSON
    
    Formato esperado:
    [
        {
            "question": "¿Qué es una variable aleatoria?",
            "ground_truth": "Una variable aleatoria es una función..." (opcional)
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
        elif hasattr(obj, 'to_dict'):  # Para DataFrames u objetos similares
            return convert_to_serializable(obj.to_dict('records'))
        else:
            return obj

    with open(output_path, 'w', encoding='utf-8') as f:
        # Convertir DataFrame a dict si existe
        if 'dataframe' in results:
            results_copy = results.copy()
            results_copy['dataframe'] = results['dataframe'].to_dict('records')
            # Convertir cualquier numpy array a lista
            results_serializable = convert_to_serializable(results_copy)
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)
        else:
            results_serializable = convert_to_serializable(results)
            json.dump(results_serializable, f, ensure_ascii=False, indent=2)

    logger.info(f"Resultados guardados en: {output_path}")


def generate_test_data_with_rag(
    rag_pipeline,
    questions: List[str],
    materia: str = None,
    k: int = 5
) -> Dict[str, List]:
    """
    Genera respuestas y contextos usando el pipeline RAG
    
    Args:
        rag_pipeline: Pipeline RAG inicializado
        questions: Lista de preguntas
        materia: Materia para filtrar (opcional)
        k: Número de chunks a recuperar
        
    Returns:
        Diccionario con questions, answers, contexts
    """
    logger.info(f"Generando respuestas para {len(questions)} preguntas...")
    
    answers = []
    contexts = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"Procesando pregunta {i}/{len(questions)}: {question[:50]}...")
        
        try:
            # Construir filtro
            filter_dict = {"materia": materia} if materia else None
            
            # Recuperar contextos
            retrieved_docs = rag_pipeline.retriever.retrieve(
                query=question,
                k=k,
                filter_dict=filter_dict
            )
            
            if not retrieved_docs:
                logger.warning(f"No se encontraron contextos para: {question[:50]}...")
                contexts.append(["No se encontró contexto relevante"])
                answers.append("No se pudo generar una respuesta por falta de contexto.")
                continue
            
            # Extraer contenido de los documentos
            context_list = [doc.page_content for doc in retrieved_docs]
            contexts.append(context_list)
            
            # Generar respuesta usando el LLM
            from langchain_openai import ChatOpenAI
            from langchain.prompts import ChatPromptTemplate
            
            llm = ChatOpenAI(model=rag_pipeline.generator_model_name, temperature=0.3)
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "Eres un asistente académico experto. Responde la pregunta basándote ÚNICAMENTE en el contexto proporcionado. Si el contexto no contiene información suficiente, indícalo claramente."),
                ("user", """Contexto:
{context}

Pregunta: {question}

Respuesta:""")
            ])
            
            # Formatear contexto
            context_text = "\n\n".join([f"[Fuente {i+1}]: {ctx}" for i, ctx in enumerate(context_list)])
            
            # Generar respuesta
            chain = prompt_template | llm
            response = chain.invoke({
                "context": context_text,
                "question": question
            })
            
            answer = response.content
            answers.append(answer)
            
            logger.info(f"  ✓ Respuesta generada ({len(answer)} caracteres)")
            
        except Exception as e:
            logger.error(f"Error procesando pregunta: {str(e)}")
            contexts.append(["Error recuperando contexto"])
            answers.append("Error generando respuesta.")
    
    return {
        'questions': questions,
        'answers': answers,
        'contexts': contexts
    }


def evaluate_rag_system(
    questions: List[str],
    ground_truths: List[str] = None,
    materia: str = None,
    k_retrieval: int = 5,
    output_dir: str = "./evaluation_results"
) -> Dict[str, Any]:
    """
    Evalúa el sistema RAG completo
    
    Args:
        questions: Lista de preguntas
        ground_truths: Lista de respuestas correctas (opcional pero recomendado)
        materia: Materia para filtrar
        k_retrieval: Número de chunks a recuperar
        output_dir: Directorio para guardar resultados
        
    Returns:
        Resultados de evaluación
    """
    print("\n" + "=" * 80)
    print("🚀 EVALUACIÓN DE SISTEMA RAG CON RAGAS")
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
    
    # 2. Generar respuestas y recuperar contextos
    print(f"\n🔍 Generando respuestas para {len(questions)} preguntas...")
    if materia:
        print(f"   Filtro: materia = {materia}")
    print(f"   Chunks por pregunta: {k_retrieval}")
    
    test_data = generate_test_data_with_rag(
        rag_pipeline=rag,
        questions=questions,
        materia=materia,
        k=k_retrieval
    )
    
    # 3. Mostrar resumen de datos generados
    print("\n📊 Datos generados:")
    print(f"   • Preguntas: {len(test_data['questions'])}")
    print(f"   • Respuestas: {len(test_data['answers'])}")
    print(f"   • Sets de contextos: {len(test_data['contexts'])}")
    if ground_truths:
        print(f"   • Ground truths: {len(ground_truths)}")
    
    # Mostrar ejemplo
    print("\n📝 Ejemplo de pregunta procesada:")
    print(f"   Pregunta: {test_data['questions'][0]}")
    print(f"   Respuesta: {test_data['answers'][0][:100]}...")
    print(f"   Contextos recuperados: {len(test_data['contexts'][0])}")
    
    # 4. Evaluar con RAGAS
    print("\n🎯 Evaluando con RAGAS...")
    evaluator = create_evaluator()
    
    # Determinar métricas disponibles
    has_gt = ground_truths is not None and len(ground_truths) > 0
    available_metrics = evaluator.get_available_metrics(has_ground_truth=has_gt)
    print(f"   Métricas a evaluar: {', '.join(available_metrics)}")
    
    if not has_gt:
        print("\n   ⚠️  ADVERTENCIA: Sin ground_truths, solo se evaluarán:")
        print("      • Faithfulness (Fidelidad)")
        print("      • Answer Relevance (Relevancia)")
        print("   Para evaluación completa, proporciona ground_truths.")
    
    results = evaluator.evaluate_rag(
        questions=test_data['questions'],
        answers=test_data['answers'],
        contexts=test_data['contexts'],
        ground_truths=ground_truths
    )
    
    if results['status'] != 'success':
        print(f"\n❌ ERROR en evaluación: {results.get('message')}")
        return results
    
    # 5. Mostrar resultados
    print("\n" + "=" * 80)
    print("📈 RESULTADOS DE EVALUACIÓN")
    print("=" * 80)
    
    stats = results['stats']
    
    print(f"\n📝 Muestras evaluadas: {stats['num_samples']}")
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
    
    # 6. Guardar resultados
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Guardar resultados JSON
    results_file = Path(output_dir) / "ragas_evaluation_results.json"
    save_results(results, results_file)
    
    # Guardar reporte de texto
    report_file = Path(output_dir) / "ragas_evaluation_report.txt"
    report = evaluator.generate_report(results, output_file=str(report_file))
    
    # Guardar datos de prueba
    test_data_file = Path(output_dir) / "test_data.json"
    with open(test_data_file, 'w', encoding='utf-8') as f:
        test_data_copy = test_data.copy()
        if ground_truths:
            test_data_copy['ground_truths'] = ground_truths
        json.dump(test_data_copy, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("💾 ARCHIVOS GENERADOS")
    print("=" * 80)
    print(f"   • Resultados JSON: {results_file}")
    print(f"   • Reporte de texto: {report_file}")
    print(f"   • Datos de prueba: {test_data_file}")
    
    # 7. Mostrar interpretación
    print("\n" + "=" * 80)
    print("💡 INTERPRETACIÓN DE MÉTRICAS")
    print("=" * 80)
    
    print("""
🎯 Faithfulness (Fidelidad): {faithfulness}
   Mide si la respuesta se basa en hechos del contexto.
   • 1.0 = Perfectamente fiel al contexto
   • 0.0 = Información no basada en el contexto
   
🎯 Answer Relevance (Relevancia de Respuesta): {answer_relevancy}
   Mide cuán relevante es la respuesta a la pregunta.
   • 1.0 = Perfectamente relevante
   • 0.0 = Completamente irrelevante
""".format(
        faithfulness="✅ Evaluada" if 'faithfulness' in stats['metrics_scores'] else "❌ No evaluada",
        answer_relevancy="✅ Evaluada" if 'answer_relevancy' in stats['metrics_scores'] else "❌ No evaluada"
    ))
    
    if has_gt:
        print("""🎯 Context Precision (Precisión de Contexto): {context_precision}
   Mide si los contextos recuperados son precisos.
   • 1.0 = Todos los contextos son relevantes
   • 0.0 = Ningún contexto es relevante
   
🎯 Context Recall (Exhaustividad de Contexto): {context_recall}
   Mide si se recuperó todo el contexto necesario.
   • 1.0 = Todo el contexto necesario fue recuperado
   • 0.0 = No se recuperó contexto necesario
""".format(
            context_precision="✅ Evaluada" if 'context_precision' in stats['metrics_scores'] else "❌ No evaluada",
            context_recall="✅ Evaluada" if 'context_recall' in stats['metrics_scores'] else "❌ No evaluada"
        ))
    
    print("=" * 80)
    print("✅ EVALUACIÓN COMPLETADA")
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evalúa el sistema RAG con RAGAS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluación básica con preguntas predefinidas (sin ground_truth)
  python evaluate_with_ragas.py
  
  # Evaluación con archivo de preguntas personalizado
  python evaluate_with_ragas.py --questions test_questions.json
  
  # Evaluación completa con ground_truths (archivo JSON)
  python evaluate_with_ragas.py --questions test_questions.json --with-ground-truth
  
  # Evaluar solo una materia específica
  python evaluate_with_ragas.py --materia "Probabilidad y estadística"
  
  # Especificar número de chunks a recuperar
  python evaluate_with_ragas.py -k 10

Formato del archivo de preguntas (JSON):
[
    {
        "question": "¿Qué es una variable aleatoria?",
        "ground_truth": "Una variable aleatoria es..." (opcional)
    },
    ...
]
        """
    )
    
    parser.add_argument(
        '--questions',
        type=str,
        help='Archivo JSON con preguntas de prueba'
    )
    
    parser.add_argument(
        '--with-ground-truth',
        action='store_true',
        help='Usar ground_truths del archivo (si están disponibles)'
    )
    
    parser.add_argument(
        '--materia',
        type=str,
        help='Filtrar por materia específica'
    )
    
    parser.add_argument(
        '-k', '--k-retrieval',
        type=int,
        default=5,
        help='Número de chunks a recuperar por pregunta (default: 5)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./evaluation_results',
        help='Directorio para guardar resultados (default: ./evaluation_results)'
    )
    
    args = parser.parse_args()
    
    # Cargar preguntas
    if args.questions:
        print(f"📖 Cargando preguntas desde: {args.questions}")
        test_questions_data = load_test_questions(args.questions)
        questions = [item['question'] for item in test_questions_data]
        
        # Extraer ground_truths si existen y se solicitan
        ground_truths = None
        if args.with_ground_truth:
            ground_truths = [item.get('ground_truth') for item in test_questions_data]
            # Verificar que todas tienen ground_truth
            if any(gt is None for gt in ground_truths):
                print("⚠️  Algunas preguntas no tienen ground_truth, se omitirá evaluación completa")
                ground_truths = None
            else:
                print(f"   ✓ {len(ground_truths)} ground_truths cargados")
    else:
        # Preguntas de ejemplo predefinidas
        print("📖 Usando preguntas de ejemplo predefinidas")
        questions = [
            "¿Qué es una variable aleatoria?",
            "¿Cuál es la definición de esperanza matemática?",
            "¿Qué propiedades tiene la varianza?",
            "¿Qué es una distribución normal?",
            "¿Cómo se calcula la probabilidad condicional?"
        ]
        ground_truths = None
    
    print(f"   ✓ {len(questions)} preguntas cargadas")
    
    # Evaluar
    try:
        results = evaluate_rag_system(
            questions=questions,
            ground_truths=ground_truths,
            materia=args.materia,
            k_retrieval=args.k_retrieval,
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
