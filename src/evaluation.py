"""
Evaluation Module for RAG System
Sistema de evaluación con métricas RAGAS
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    Evaluador de sistemas RAG usando RAGAS
    
    Métricas implementadas:
    - Faithfulness: ¿La respuesta es fiel al contexto recuperado?
    - Answer Relevancy: ¿La respuesta es relevante a la pregunta?
    - Context Precision: ¿Los contextos recuperados son precisos?
    - Context Recall: ¿Se recuperó todo el contexto necesario?
    """
    
    def __init__(self):
        """Inicializa el evaluador con métricas RAGAS"""
        try:
            from datasets import Dataset
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            from ragas import evaluate
            
            self.Dataset = Dataset
            self.evaluate = evaluate
            self.metrics = {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'context_precision': context_precision,
                'context_recall': context_recall
            }
            
            logger.info("RAGEvaluator inicializado con métricas RAGAS")
            
        except ImportError as e:
            logger.error(f"Error importando RAGAS: {str(e)}")
            logger.error("Instala ragas con: pip install ragas datasets")
            raise
    
    def evaluate_rag(
        self,
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evalúa un sistema RAG con RAGAS
        
        Args:
            questions: Lista de preguntas
            answers: Lista de respuestas generadas por el sistema
            contexts: Lista de listas de contextos recuperados para cada pregunta
            ground_truths: Lista de respuestas correctas (opcional, para context_recall)
            metrics: Lista de nombres de métricas a usar (None = todas)
            
        Returns:
            Diccionario con resultados de evaluación
        """
        # Validar longitudes
        if not (len(questions) == len(answers) == len(contexts)):
            raise ValueError("questions, answers y contexts deben tener la misma longitud")
        
        # Preparar datos para RAGAS
        data_samples = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
        }
        
        # Agregar ground_truths si están disponibles (necesario para context_recall)
        if ground_truths:
            if len(ground_truths) != len(questions):
                raise ValueError("ground_truths debe tener la misma longitud que questions")
            data_samples['ground_truth'] = ground_truths
        
        # Crear dataset
        dataset = self.Dataset.from_dict(data_samples)
        
        # Seleccionar métricas
        if metrics:
            selected_metrics = [self.metrics[m] for m in metrics if m in self.metrics]
        else:
            # Si no hay ground_truths, omitir context_recall
            if ground_truths:
                selected_metrics = list(self.metrics.values())
            else:
                selected_metrics = [
                    self.metrics['faithfulness'],
                    self.metrics['answer_relevancy'],
                    self.metrics['context_precision']
                ]
        
        logger.info(f"Evaluando {len(questions)} preguntas con {len(selected_metrics)} métricas")
        
        # Evaluar
        try:
            result = self.evaluate(
                dataset,
                metrics=selected_metrics,
                raise_exceptions=False
            )
            
            # Convertir a pandas para mejor visualización
            df_results = result.to_pandas()
            
            # Calcular estadísticas agregadas
            stats = {
                'num_samples': len(questions),
                'metrics_scores': {},
                'detailed_results': df_results.to_dict('records')
            }
            
            # Calcular promedios de cada métrica
            for metric_name in self.metrics.keys():
                if metric_name in df_results.columns:
                    scores = df_results[metric_name].dropna()
                    if len(scores) > 0:
                        stats['metrics_scores'][metric_name] = {
                            'mean': float(scores.mean()),
                            'std': float(scores.std()),
                            'min': float(scores.min()),
                            'max': float(scores.max())
                        }
            
            logger.info("Evaluación completada exitosamente")
            return {
                'status': 'success',
                'stats': stats,
                'dataframe': df_results
            }
            
        except Exception as e:
            logger.error(f"Error durante evaluación: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def generate_report(
        self,
        evaluation_results: Dict[str, Any],
        output_file: Optional[str] = None
    ) -> str:
        """
        Genera un reporte legible de la evaluación
        
        Args:
            evaluation_results: Resultados de evaluate_rag()
            output_file: Archivo donde guardar el reporte (opcional)
            
        Returns:
            String con el reporte
        """
        if evaluation_results.get('status') != 'success':
            return f"Error en evaluación: {evaluation_results.get('message', 'Desconocido')}"
        
        stats = evaluation_results['stats']
        
        # Construir reporte
        report_lines = [
            "=" * 70,
            "REPORTE DE EVALUACION RAG",
            "=" * 70,
            "",
            f" Muestras evaluadas: {stats['num_samples']}",
            "",
            " MÉTRICAS AGREGADAS:",
            "-" * 70,
        ]
        
        # Agregar cada métrica
        for metric_name, scores in stats['metrics_scores'].items():
            report_lines.extend([
                f"\n {metric_name.upper().replace('_', ' ')}:",
                f"   - Promedio: {scores['mean']:.4f}",
                f"   - Desv. Estándar: {scores['std']:.4f}",
                f"   - Mínimo: {scores['min']:.4f}",
                f"   - Máximo: {scores['max']:.4f}"
            ])
        
        # Interpretación de métricas
        report_lines.extend([
            "",
            "=" * 70,
            " INTERPRETACIÓN DE MÉTRICAS:",
            "=" * 70,
            "",
            " Faithfulness (Fidelidad):",
            "   Mide si la respuesta se basa en los hechos del contexto.",
            "   1.0 = perfectamente fiel, 0.0 = no fiel",
            "",
            " Answer Relevancy (Relevancia de Respuesta):",
            "   Mide cuán relevante es la respuesta a la pregunta.",
            "   1.0 = perfectamente relevante, 0.0 = irrelevante",
            "",
            " Context Precision (Precisión de Contexto):",
            "   Mide si los contextos recuperados son precisos.",
            "   1.0 = todos los contextos son relevantes, 0.0 = ninguno relevante",
            "",
            " Context Recall (Exhaustividad de Contexto):",
            "   Mide si se recuperó todo el contexto necesario.",
            "   1.0 = todo recuperado, 0.0 = nada recuperado",
            "   (Requiere ground_truth)",
            "",
            "=" * 70
        ])
        
        report = "\n".join(report_lines)
        
        # Guardar si se especifica archivo
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Reporte guardado en: {output_file}")
        
        return report


def create_evaluator() -> RAGEvaluator:
    """
    Función de conveniencia para crear un evaluador
    
    Returns:
        Instancia de RAGEvaluator
    """
    return RAGEvaluator()


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Datos de ejemplo
    questions = [
        "¿Qué es una variable aleatoria?",
        "¿Cuál es la fórmula de la varianza?"
    ]
    
    answers = [
        "Una variable aleatoria es una función que asigna valores numéricos a los resultados de un experimento aleatorio.",
        "La varianza es V(X) = E[(X - μ)²]"
    ]
    
    contexts = [
        ["Una variable aleatoria es una función que mapea resultados de experimentos a números reales."],
        ["La varianza se calcula como V(X) = E[(X - E[X])²] = E[X²] - (E[X])²"]
    ]
    
    # Crear evaluador
    evaluator = create_evaluator()
    
    # Evaluar
    results = evaluator.evaluate_rag(
        questions=questions,
        answers=answers,
        contexts=contexts
    )
    
    # Generar reporte
    if results['status'] == 'success':
        report = evaluator.generate_report(results)
        print(report)

