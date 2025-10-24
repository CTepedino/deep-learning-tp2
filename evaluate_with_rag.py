#!/usr/bin/env python3
"""
Script alternativo para evaluar tu sistema RAG usando tu pipeline y RAGAS
Uso: python evaluate_with_rag.py --input test_questions_with_ground_truth_4.json

- Genera respuestas con tu pipeline RAG
- Evalúa con RAGAS (faithfulness, answer_relevancy, context_precision, context_recall)
"""
import argparse
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='Evalúa sistema RAG usando pipeline y RAGAS')
    parser.add_argument('--input', type=str, required=True, help='Archivo JSON con queries y ground_truth')
    parser.add_argument('--output', type=str, default='./evaluation_results/rag_pipeline_results.json', help='Archivo para guardar resultados')
    args = parser.parse_args()

    logger.info(f"Cargando datos de test desde: {args.input}")
    test_data = load_test_data(args.input)

    # Importar pipeline
    from src.rag_pipeline import create_rag_pipeline
    pipeline = create_rag_pipeline()

    questions, answers, contexts, ground_truths = [], [], [], []
    for item in test_data:
        query = item['question'] if isinstance(item['question'], dict) else None
        if not query:
            # Si la pregunta es string, intentar parsear los campos
            logger.warning("El campo 'question' no es un dict. Se omite este caso.")
            continue
        ground_truth = item.get('ground_truth', None)
        # Generar respuesta y contextos con el pipeline
        result = pipeline.generate_exercises(query)
        # Extraer respuesta y contextos
        answer = result.get('ejercicio', '') if 'ejercicio' in result else str(result)
        context = result.get('contextos', []) if 'contextos' in result else []
        questions.append(json.dumps(query, ensure_ascii=False))
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(ground_truth)

    # Importar RAGAS y métricas
    from datasets import Dataset
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from ragas import evaluate

    samples = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths
    }
    dataset = Dataset.from_dict(samples)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    logger.info(f"Evaluando {len(questions)} muestras con RAGAS...")
    result = evaluate(dataset, metrics=metrics, raise_exceptions=False)
    df = result.to_pandas()

    # Guardar resultados
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_json(args.output, orient='records', force_ascii=False, indent=2)
    logger.info(f"Resultados guardados en: {args.output}")

    # Mostrar resumen
    print("\n=== RESULTADOS DE EVALUACIÓN RAG + RAGAS ===\n")
    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
        if metric in df.columns:
            vals = df[metric].dropna()
            if len(vals) > 0:
                print(f"{metric}: Promedio={vals.mean():.4f}, Min={vals.min():.4f}, Max={vals.max():.4f}")
    print("\nEvaluación completada.\n")

if __name__ == "__main__":
    main()

