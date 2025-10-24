#!/usr/bin/env python3
"""
Script para evaluar sistema RAG de generación de ejercicios usando LLM as a Judge
Uso: python evaluate_with_llm_judge.py --input test_questions.json

En lugar de comparar con ground_truth, evalúa la calidad del ejercicio generado
según criterios relevantes para generación de ejercicios académicos.
"""
import argparse
import json
from pathlib import Path
import logging
from typing import Dict, Any, List
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_test_data(file_path: str) -> List[Dict]:
    """Carga el archivo de test con queries"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_query(question) -> Dict[str, Any]:
    """
    Parsea la query a formato dict si es string con formato especial
    """
    if isinstance(question, dict):
        return question
    
    # Si es string con formato "Materia: ... | Tipo: ..."
    if isinstance(question, str) and '|' in question:
        query_dict = {}
        parts = question.split('|')
        for part in parts:
            if ':' in part:
                key, value = part.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip()
                query_dict[key] = value
        return query_dict
    
    # Si es un string simple, usar como consulta_libre
    return {"consulta_libre": question}


def create_judge_prompt(query: Dict[str, Any], generated_exercise: str, contexts: List[str]) -> str:
    """
    Crea el prompt para que el LLM evalúe el ejercicio generado usando las mismas métricas de RAGAS
    """
    # Formatear la query de forma legible
    query_text = "\n".join([f"  - {k}: {v}" for k, v in query.items()])
    
    # Formatear contextos (limitar longitud)
    contexts_text = "\n".join([f"  [{i+1}] {ctx[:300]}..." for i, ctx in enumerate(contexts[:5])])

    prompt = f"""Eres un evaluador experto de sistemas RAG. Tu tarea es evaluar un ejercicio generado por un sistema RAG usando las mismas métricas que RAGAS (Retrieval-Augmented Generation Assessment).

**SOLICITUD ORIGINAL:**
{query_text}

**CONTEXTOS RECUPERADOS DEL RAG:**
{contexts_text}

**EJERCICIO GENERADO:**
{generated_exercise}

---

**MÉTRICAS A EVALUAR (escala 0.0 a 1.0, siguiendo criterios de RAGAS):**

1. **FAITHFULNESS (Fidelidad al Contexto)** [0.0 - 1.0]:
   - ¿El ejercicio generado se basa ÚNICAMENTE en información presente en los contextos recuperados?
   - ¿Hay afirmaciones, conceptos o datos que NO estén respaldados por los contextos?
   - 1.0 = Todas las afirmaciones están respaldadas por los contextos
   - 0.0 = Contiene información inventada o no respaldada

2. **ANSWER RELEVANCE (Relevancia de la Respuesta)** [0.0 - 1.0]:
   - ¿El ejercicio generado responde directamente a lo solicitado en la query?
   - ¿El tema, materia, tipo y nivel de dificultad coinciden con lo solicitado?
   - 1.0 = Perfectamente relevante a la solicitud
   - 0.0 = Completamente irrelevante

3. **CONTEXT PRECISION (Precisión del Contexto)** [0.0 - 1.0]:
   - ¿Los contextos recuperados son relevantes y útiles para generar el ejercicio?
   - ¿Cuántos de los contextos recuperados realmente contribuyen al ejercicio?
   - 1.0 = Todos los contextos son relevantes y útiles
   - 0.0 = Ningún contexto es relevante

4. **CONTEXT RECALL (Exhaustividad del Contexto)** [0.0 - 1.0]:
   - ¿Los contextos recuperados contienen toda la información necesaria para un buen ejercicio?
   - ¿Se nota que falta información importante del dominio?
   - 1.0 = Los contextos tienen toda la información necesaria
   - 0.0 = Los contextos no tienen información suficiente

---

**FORMATO DE RESPUESTA REQUERIDO (JSON):**
```json
{{
  "faithfulness": {{
    "score": <0.0-1.0>,
    "reasoning": "<explicación de por qué esta puntuación>"
  }},
  "answer_relevance": {{
    "score": <0.0-1.0>,
    "reasoning": "<explicación de por qué esta puntuación>"
  }},
  "context_precision": {{
    "score": <0.0-1.0>,
    "reasoning": "<explicación de por qué esta puntuación>"
  }},
  "context_recall": {{
    "score": <0.0-1.0>,
    "reasoning": "<explicación de por qué esta puntuación>"
  }},
  "overall_score": <promedio de las 4 métricas>,
  "summary": "<resumen breve de la evaluación general>"
}}
```

**IMPORTANTE:**
- Usa escala decimal de 0.0 a 1.0 (no 0-10)
- Sé estricto con faithfulness: cualquier información no respaldada reduce el score
- Para context_precision, considera cuántos contextos son realmente útiles vs recuperados
- Para context_recall, evalúa si los contextos tienen suficiente profundidad temática

Responde SOLO con el JSON, sin texto adicional."""

    return prompt


def evaluate_with_llm(query: Dict[str, Any], generated_exercise: str, contexts: List[str], llm_client) -> Dict[str, Any]:
    """
    Evalúa el ejercicio generado usando un LLM como juez
    """
    prompt = create_judge_prompt(query, generated_exercise, contexts)
    
    try:
        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "Eres un evaluador experto de contenido académico. Respondes siempre en formato JSON válido."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Baja temperatura para evaluaciones más consistentes
            max_tokens=1000
        )
        
        result_text = response.choices[0].message.content.strip()
        
        # Extraer JSON si está en un bloque de código
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()
        
        evaluation = json.loads(result_text)
        return {
            "status": "success",
            "evaluation": evaluation
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Error al parsear respuesta del LLM: {e}")
        logger.error(f"Respuesta: {result_text}")
        return {
            "status": "error",
            "message": f"Error al parsear JSON: {str(e)}",
            "raw_response": result_text
        }
    except Exception as e:
        logger.error(f"Error al llamar al LLM: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


def calculate_aggregate_stats(results: List[Dict]) -> Dict[str, Any]:
    """
    Calcula estadísticas agregadas de todas las evaluaciones usando métricas RAGAS
    """
    metrics = ['faithfulness', 'answer_relevance', 'context_precision', 'context_recall']

    stats = {
        'num_evaluaciones': len(results),
        'num_exitosas': sum(1 for r in results if r.get('status') == 'success'),
        'num_errores': sum(1 for r in results if r.get('status') == 'error'),
        'metrics': {}
    }
    
    for metric in metrics:
        scores = []
        for result in results:
            if result.get('status') == 'success' and 'evaluation' in result:
                eval_data = result['evaluation']
                if metric in eval_data and 'score' in eval_data[metric]:
                    scores.append(eval_data[metric]['score'])

        if scores:
            stats['metrics'][metric] = {
                'mean': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores),
                'std': (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores)) ** 0.5,
                'num_samples': len(scores)
            }
    
    # Calcular promedio total (overall_score)
    overall_scores = []
    for result in results:
        if result.get('status') == 'success' and 'evaluation' in result:
            overall_score = result['evaluation'].get('overall_score')
            if overall_score is not None:
                overall_scores.append(overall_score)

    if overall_scores:
        stats['overall_score'] = {
            'mean': sum(overall_scores) / len(overall_scores),
            'min': min(overall_scores),
            'max': max(overall_scores),
            'std': (sum((x - sum(overall_scores)/len(overall_scores))**2 for x in overall_scores) / len(overall_scores)) ** 0.5
        }
    
    return stats


def generate_report(results: List[Dict], stats: Dict[str, Any]) -> str:
    """
    Genera un reporte legible de la evaluación usando métricas RAGAS
    """
    lines = [
        "=" * 80,
        "🎓 REPORTE DE EVALUACIÓN: LLM AS A JUDGE (Métricas RAGAS)",
        "=" * 80,
        "",
        f"📊 Total de evaluaciones: {stats['num_evaluaciones']}",
        f"✅ Exitosas: {stats['num_exitosas']}",
        f"❌ Errores: {stats['num_errores']}",
        "",
        "=" * 80,
        "📈 MÉTRICAS RAGAS (escala 0.0-1.0)",
        "=" * 80,
        ""
    ]
    
    # Métricas RAGAS en orden
    metric_names = {
        'faithfulness': '🎯 Faithfulness (Fidelidad al Contexto)',
        'answer_relevance': '🎯 Answer Relevance (Relevancia de Respuesta)',
        'context_precision': '🎯 Context Precision (Precisión del Contexto)',
        'context_recall': '🎯 Context Recall (Exhaustividad del Contexto)'
    }
    
    for metric, name in metric_names.items():
        if metric in stats['metrics']:
            data = stats['metrics'][metric]
            lines.append(f"{name}:")
            lines.append(f"   Promedio: {data['mean']:.4f}")
            lines.append(f"   Desv. Estándar: {data['std']:.4f}")
            lines.append(f"   Rango: [{data['min']:.4f} - {data['max']:.4f}]")

            # Interpretación
            mean_score = data['mean']
            if mean_score >= 0.8:
                lines.append(f"   Evaluación: ✅ EXCELENTE")
            elif mean_score >= 0.6:
                lines.append(f"   Evaluación: ✓ BUENO")
            elif mean_score >= 0.4:
                lines.append(f"   Evaluación: ⚠️  REGULAR")
            else:
                lines.append(f"   Evaluación: ❌ NECESITA MEJORA")
            lines.append("")
    
    if 'overall_score' in stats:
        lines.extend([
            "=" * 80,
            "🏆 PUNTUACIÓN GENERAL",
            "=" * 80,
            "",
            f"   Promedio: {stats['overall_score']['mean']:.4f}",
            f"   Desv. Estándar: {stats['overall_score']['std']:.4f}",
            f"   Rango: [{stats['overall_score']['min']:.4f} - {stats['overall_score']['max']:.4f}]",
            ""
        ])
    
    # Detalles individuales
    lines.extend([
        "=" * 80,
        "📝 EVALUACIONES INDIVIDUALES",
        "=" * 80,
        ""
    ])
    
    for i, result in enumerate(results, 1):
        lines.append(f"--- Ejercicio {i} ---")
        
        if result.get('status') == 'error':
            lines.append(f"❌ Error: {result.get('message', 'Desconocido')}")
        else:
            evaluation = result.get('evaluation', {})
            overall = evaluation.get('overall_score', 'N/A')
            lines.append(f"Puntuación General: {overall:.4f}" if isinstance(overall, (int, float)) else f"Puntuación General: {overall}")

            # Mostrar cada métrica
            for metric, name in metric_names.items():
                if metric in evaluation:
                    score = evaluation[metric].get('score', 'N/A')
                    lines.append(f"  • {metric}: {score:.4f}" if isinstance(score, (int, float)) else f"  • {metric}: {score}")

            summary = evaluation.get('summary', '')
            if summary:
                lines.append(f"Resumen: {summary}")

            # Mostrar métrica con menor puntuación
            min_metric = None
            min_score = 2.0
            for metric in metric_names.keys():
                if metric in evaluation and 'score' in evaluation[metric]:
                    score = evaluation[metric]['score']
                    if score < min_score:
                        min_score = score
                        min_metric = (metric, evaluation[metric])

            if min_metric and min_score < 0.6:
                lines.append(f"⚠️  Punto débil: {metric_names[min_metric[0]]} ({min_score:.4f})")
                lines.append(f"   {min_metric[1].get('reasoning', '')}")

        lines.append("")
    
    lines.extend([
        "=" * 80,
        "💡 INTERPRETACIÓN DE MÉTRICAS RAGAS",
        "=" * 80,
        "",
        "🎯 Faithfulness (Fidelidad al Contexto):",
        "   Mide si el ejercicio se basa únicamente en información de los contextos.",
        "   1.0 = Perfectamente fiel | 0.0 = Información no respaldada",
        "",
        "🎯 Answer Relevance (Relevancia de Respuesta):",
        "   Mide si el ejercicio responde a lo solicitado en la query.",
        "   1.0 = Perfectamente relevante | 0.0 = Completamente irrelevante",
        "",
        "🎯 Context Precision (Precisión del Contexto):",
        "   Mide si los contextos recuperados son relevantes y útiles.",
        "   1.0 = Todos relevantes | 0.0 = Ninguno relevante",
        "",
        "🎯 Context Recall (Exhaustividad del Contexto):",
        "   Mide si los contextos contienen toda la información necesaria.",
        "   1.0 = Información completa | 0.0 = Información insuficiente",
        "",
        "=" * 80,
        "📊 ESCALA DE INTERPRETACIÓN:",
        "=" * 80,
        "",
        "  0.8 - 1.0:  ✅ Excelente",
        "  0.6 - 0.8:  ✓  Bueno",
        "  0.4 - 0.6:  ⚠️  Regular - Necesita mejora",
        "  0.0 - 0.4:  ❌ Deficiente - Requiere atención",
        "",
        "=" * 80
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Evalúa generación de ejercicios usando LLM as a Judge')
    parser.add_argument('--input', type=str, required=True, help='Archivo JSON con queries')
    parser.add_argument('--output-dir', type=str, default='./evaluation_results', help='Directorio para guardar resultados')
    parser.add_argument('--skip-generation', action='store_true', help='Si ya tienes ejercicios generados, solo evaluar')
    args = parser.parse_args()

    logger.info(f"Cargando datos de test desde: {args.input}")
    test_data = load_test_data(args.input)

    # Inicializar cliente LLM
    from openai import OpenAI
    llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Inicializar pipeline RAG si es necesario
    if not args.skip_generation:
        from src.rag_pipeline import create_rag_pipeline
        logger.info("Inicializando pipeline RAG...")
        pipeline = create_rag_pipeline()
    
    results = []
    
    for i, item in enumerate(test_data, 1):
        logger.info(f"Procesando ejercicio {i}/{len(test_data)}...")
        
        # Parsear query
        query = parse_query(item['question'])
        
        if not args.skip_generation:
            # Generar ejercicio con el pipeline
            logger.info("Generando ejercicio con RAG...")
            result = pipeline.generate_exercises(query)
            
            if result.get('status') == 'error':
                logger.error(f"Error generando ejercicio: {result.get('message')}")
                results.append({
                    'query': query,
                    'status': 'error',
                    'message': f"Error en generación: {result.get('message')}"
                })
                continue
            
            # Extraer ejercicio
            generated_exercise = result.get('ejercicios', [{}])[0].get('contenido', '') if 'ejercicios' in result else result.get('ejercicio', str(result))

            # Recuperar contextos directamente del retriever
            logger.info("Recuperando contextos del retriever...")
            from src.query_utils import prepare_search_query
            search_query = prepare_search_query(query)

            # Construir filtros si es necesario
            filter_dict = None
            if query.get("materia"):
                from src.text_processing import normalize_text
                materia_normalizada = normalize_text(query["materia"])
                filter_dict = {"materia": materia_normalizada}

            # Recuperar documentos
            context_docs = pipeline.retriever.retrieve(
                query=search_query,
                k=5,
                filter_dict=filter_dict
            )

            # Extraer textos de los documentos
            contexts = [doc.page_content for doc in context_docs] if context_docs else []
            logger.info(f"Recuperados {len(contexts)} contextos")
        else:
            # Usar ejercicio pre-generado si existe
            generated_exercise = item.get('answer', '')
            contexts = item.get('contexts', [])
        
        # Evaluar con LLM Judge
        logger.info("Evaluando con LLM Judge...")
        evaluation_result = evaluate_with_llm(query, generated_exercise, contexts, llm_client)
        
        results.append({
            'query': query,
            'generated_exercise': generated_exercise,
            'contexts': contexts,
            **evaluation_result
        })
    
    # Calcular estadísticas agregadas
    stats = calculate_aggregate_stats(results)
    
    # Guardar resultados
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar JSON con resultados detallados
    results_file = output_dir / f"llm_judge_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'stats': stats,
            'results': results
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Resultados detallados guardados en: {results_file}")
    
    # Generar y guardar reporte
    report = generate_report(results, stats)
    report_file = output_dir / f"llm_judge_report_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Reporte guardado en: {report_file}")
    
    # Mostrar reporte en consola
    print("\n" + report)
    
    logger.info("✅ Evaluación completada")


if __name__ == "__main__":
    main()
