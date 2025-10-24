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
    - Context Precision: ¿Los contextos recuperados son precisos? (requiere ground_truth)
    - Context Recall: ¿Se recuperó todo el contexto necesario? (requiere ground_truth)
    """
    
    def __init__(self):
        """Inicializa el evaluador con métricas RAGAS"""
        try:
            from datasets import Dataset
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_entity_recall
            )
            from ragas import evaluate
            
            self.Dataset = Dataset
            self.evaluate = evaluate
            
            # Métricas que NO requieren ground_truth
            self.metrics_without_gt = {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
            }
            
            # Métricas que SÍ requieren ground_truth
            self.metrics_with_gt = {
                'context_precision': context_precision,
                'context_recall': context_recall,
                'context_entity_recall': context_entity_recall
            }
            
            # Todas las métricas
            self.all_metrics = {**self.metrics_without_gt, **self.metrics_with_gt}
            
            logger.info("RAGEvaluator inicializado con métricas RAGAS")
            logger.info(f"Métricas sin ground_truth: {list(self.metrics_without_gt.keys())}")
            logger.info(f"Métricas con ground_truth: {list(self.metrics_with_gt.keys())}")
            
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
            ground_truths: Lista de respuestas correctas (opcional, para métricas avanzadas)
            metrics: Lista de nombres de métricas a usar (None = todas disponibles)
            
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
        
        # Verificar si tenemos ground_truths
        has_ground_truth = ground_truths is not None and len(ground_truths) > 0
        
        if has_ground_truth:
            if len(ground_truths) != len(questions):
                raise ValueError("ground_truths debe tener la misma longitud que questions")
            data_samples['ground_truth'] = ground_truths
            logger.info("Ground truths proporcionados - se usarán todas las métricas")
        else:
            logger.warning("Sin ground truths - solo se usarán métricas básicas (faithfulness, answer_relevancy)")
        
        # Crear dataset
        dataset = self.Dataset.from_dict(data_samples)
        
        # Seleccionar métricas según disponibilidad de ground_truth
        if metrics:
            # Usuario especificó métricas específicas
            selected_metrics = []
            for m in metrics:
                if m in self.all_metrics:
                    # Verificar si la métrica requiere ground_truth
                    if m in self.metrics_with_gt and not has_ground_truth:
                        logger.warning(f"Métrica '{m}' requiere ground_truth pero no está disponible - se omite")
                    else:
                        selected_metrics.append(self.all_metrics[m])
                else:
                    logger.warning(f"Métrica '{m}' no reconocida - se omite")
        else:
            # Usar todas las métricas disponibles según ground_truth
            if has_ground_truth:
                selected_metrics = list(self.all_metrics.values())
                logger.info("Usando todas las métricas (con ground_truth)")
            else:
                selected_metrics = list(self.metrics_without_gt.values())
                logger.info("Usando solo métricas básicas (sin ground_truth)")
        
        if not selected_metrics:
            raise ValueError("No hay métricas válidas para evaluar")
        
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
                'has_ground_truth': has_ground_truth,
                'metrics_scores': {},
                'detailed_results': df_results.to_dict('records')
            }
            
            # Calcular promedios de cada métrica
            for metric_name in self.all_metrics.keys():
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
            import traceback
            traceback.print_exc()
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
            "📊 REPORTE DE EVALUACIÓN RAG",
            "=" * 70,
            "",
            f"📝 Muestras evaluadas: {stats['num_samples']}",
            f"🎯 Ground truth disponible: {'Sí' if stats.get('has_ground_truth') else 'No'}",
            "",
            "📈 MÉTRICAS AGREGADAS:",
            "-" * 70,
        ]
        
        # Agregar cada métrica
        for metric_name, scores in stats['metrics_scores'].items():
            report_lines.extend([
                f"\n🔹 {metric_name.upper().replace('_', ' ')}:",
                f"   - Promedio: {scores['mean']:.4f}",
                f"   - Desv. Estándar: {scores['std']:.4f}",
                f"   - Mínimo: {scores['min']:.4f}",
                f"   - Máximo: {scores['max']:.4f}"
            ])
        
        # Interpretación de métricas
        report_lines.extend([
            "",
            "=" * 70,
            "💡 INTERPRETACIÓN DE MÉTRICAS:",
            "=" * 70,
            "",
            "🎯 Faithfulness (Fidelidad):",
            "   Mide si la respuesta se basa en los hechos del contexto.",
            "   1.0 = perfectamente fiel, 0.0 = no fiel",
            "   ✅ NO requiere ground_truth",
            "",
            "🎯 Answer Relevancy (Relevancia de Respuesta):",
            "   Mide cuán relevante es la respuesta a la pregunta.",
            "   1.0 = perfectamente relevante, 0.0 = irrelevante",
            "   ✅ NO requiere ground_truth",
            "",
            "🎯 Context Precision (Precisión de Contexto):",
            "   Mide si los contextos recuperados son precisos.",
            "   1.0 = todos los contextos son relevantes, 0.0 = ninguno relevante",
            "   ⚠️  REQUIERE ground_truth",
            "",
            "🎯 Context Recall (Exhaustividad de Contexto):",
            "   Mide si se recuperó todo el contexto necesario.",
            "   1.0 = todo recuperado, 0.0 = nada recuperado",
            "   ⚠️  REQUIERE ground_truth",
            "",
            "=" * 70
        ])
        
        # Agregar nota sobre ground_truth si no está disponible
        if not stats.get('has_ground_truth'):
            report_lines.extend([
                "",
                "⚠️  NOTA: Algunas métricas avanzadas no están disponibles porque",
                "   no se proporcionaron ground_truths (respuestas correctas).",
                "   Para evaluación completa, proporciona ground_truths al evaluar.",
                ""
            ])
        
        report = "\n".join(report_lines)
        
        # Guardar si se especifica archivo
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Reporte guardado en: {output_file}")
        
        return report
    
    def get_available_metrics(self, has_ground_truth: bool = False) -> List[str]:
        """
        Retorna lista de métricas disponibles según disponibilidad de ground_truth
        
        Args:
            has_ground_truth: Si hay ground_truth disponible
            
        Returns:
            Lista de nombres de métricas disponibles
        """
        if has_ground_truth:
            return list(self.all_metrics.keys())
        else:
            return list(self.metrics_without_gt.keys())


def create_evaluator() -> RAGEvaluator:
    """
    Función de conveniencia para crear un evaluador
    
    Returns:
        Instancia de RAGEvaluator
    """
    return RAGEvaluator()

