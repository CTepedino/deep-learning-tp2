#!/usr/bin/env python3
"""
Script para evaluar el sistema RAG con métricas RAGAS
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Agregar el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import create_rag_pipeline
from src.evaluation import create_evaluator


def load_evaluation_dataset(dataset_path: str = "./data/evaluation_dataset.json"):
    """Carga el dataset de evaluación"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['samples']


def run_evaluation(
    rag_pipeline,
    evaluation_samples,
    max_samples: int = None,
    save_results: bool = True
):
    """
    Ejecuta la evaluación completa del sistema RAG
    
    Args:
        rag_pipeline: Pipeline RAG inicializado
        evaluation_samples: Lista de muestras de evaluación
        max_samples: Número máximo de muestras a evaluar (None = todas)
        save_results: Si guardar resultados en archivo
        
    Returns:
        Diccionario con resultados de evaluación
    """
    print("\n" + "="*70)
    print("🔬 EVALUACIÓN DEL SISTEMA RAG")
    print("="*70)
    
    # Limitar número de muestras si se especifica
    if max_samples:
        evaluation_samples = evaluation_samples[:max_samples]
    
    print(f"\n📊 Muestras a evaluar: {len(evaluation_samples)}")
    
    # Listas para almacenar datos
    questions = []
    ground_truths = []
    generated_answers = []
    retrieved_contexts = []
    metadata_list = []
    
    print("\n⏳ Generando respuestas para cada pregunta...")
    print("-"*70)
    
    # Para cada muestra del dataset
    for i, sample in enumerate(evaluation_samples, 1):
        question = sample['question']
        ground_truth = sample['ground_truth']
        
        print(f"\n[{i}/{len(evaluation_samples)}] {question[:60]}...")
        
        # Preparar parámetros para la generación
        query_params = {
            'materia': sample.get('materia', 'Probabilidad y estadística'),
            'tipo_ejercicio': sample.get('tipo_ejercicio', 'teorico'),
            'nivel_dificultad': sample.get('nivel_dificultad', 'intermedio'),
            'cantidad': 1,
            'pregunta_especifica': question
        }
        
        try:
            # Primero recuperar contexto
            from src.query_utils import prepare_search_query
            search_query = prepare_search_query(query_params)
            
            context_docs = rag_pipeline.search_materials(
                query=search_query,
                k=5
            )
            
            # Extraer contextos
            contexts = [doc['content'] for doc in context_docs]
            
            print(f"   ✅ Recuperados {len(contexts)} chunks de contexto")
            
            # Generar respuesta usando el generador directamente
            # En lugar de generar ejercicios, hacemos una pregunta directa
            context_text = "\n\n".join(contexts)
            
            # Usar el generador para responder la pregunta
            prompt = f"""Basándote en el siguiente contexto, responde la pregunta de forma clara y concisa.

Contexto:
{context_text}

Pregunta: {question}

Respuesta:"""
            
            # Llamar al LLM
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model=os.getenv('LLM_MODEL', 'gpt-4o-mini'),
                temperature=0.3
            )
            
            response = llm.invoke(prompt)
            answer = response.content.strip()
            
            print(f"   ✅ Respuesta generada: {answer[:80]}...")
            
            # Almacenar resultados
            questions.append(question)
            ground_truths.append(ground_truth)
            generated_answers.append(answer)
            retrieved_contexts.append(contexts)
            metadata_list.append(sample)
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            # Agregar valores por defecto para no romper la evaluación
            questions.append(question)
            ground_truths.append(ground_truth)
            generated_answers.append("Error al generar respuesta")
            retrieved_contexts.append(["No se pudo recuperar contexto"])
            metadata_list.append(sample)
    
    print("\n" + "="*70)
    print("📈 CALCULANDO MÉTRICAS RAGAS")
    print("="*70)
    
    # Crear evaluador
    evaluator = create_evaluator()
    
    # Evaluar con RAGAS
    results = evaluator.evaluate_rag(
        questions=questions,
        answers=generated_answers,
        contexts=retrieved_contexts,
        ground_truths=ground_truths
    )
    
    if results['status'] != 'success':
        print(f"\n❌ Error en evaluación: {results.get('message')}")
        return results
    
    # Generar reporte
    print("\n" + "="*70)
    print("📄 GENERANDO REPORTE")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if save_results:
        # Guardar reporte de texto
        report_file = f"./data/evaluation_report_{timestamp}.txt"
        report = evaluator.generate_report(results, output_file=report_file)
        print(report)
        print(f"\n💾 Reporte guardado en: {report_file}")
        
        # Guardar resultados detallados en CSV
        csv_file = f"./data/evaluation_results_{timestamp}.csv"
        df = results['dataframe']
        
        # Agregar metadata al dataframe
        for i, metadata in enumerate(metadata_list):
            for key, value in metadata.items():
                if key not in ['question', 'ground_truth']:
                    df.at[i, f'meta_{key}'] = value
        
        df.to_csv(csv_file, index=False)
        print(f"💾 Resultados detallados guardados en: {csv_file}")
        
        # Guardar datos crudos en JSON
        json_file = f"./data/evaluation_data_{timestamp}.json"
        evaluation_data = {
            'timestamp': timestamp,
            'num_samples': len(questions),
            'samples': [
                {
                    'question': q,
                    'ground_truth': gt,
                    'generated_answer': ga,
                    'contexts': ctx,
                    'metadata': meta
                }
                for q, gt, ga, ctx, meta in zip(
                    questions, ground_truths, generated_answers,
                    retrieved_contexts, metadata_list
                )
            ],
            'metrics': results['stats']['metrics_scores']
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Datos de evaluación guardados en: {json_file}")
    else:
        report = evaluator.generate_report(results)
        print(report)
    
    return results


def main():
    print("\n" + "="*70)
    print("🎓 EVALUACIÓN DEL SISTEMA RAG - GENERADOR DE EJERCICIOS")
    print("="*70)
    
    # Cargar variables de entorno
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here':
        print("\n❌ ERROR: Necesitas configurar tu OPENAI_API_KEY en el archivo .env")
        print("   1. Copia env.example a .env: cp env.example .env")
        print("   2. Edita .env y agrega tu API key de OpenAI")
        return
    
    # Inicializar pipeline RAG
    print("\n⏳ Inicializando pipeline RAG...")
    rag_pipeline = create_rag_pipeline(reset_collection=False)
    
    system_info = rag_pipeline.get_system_info()
    doc_count = system_info['vector_store']['document_count']
    
    print(f"✅ Pipeline inicializado")
    print(f"   - Chunks en DB: {doc_count}")
    print(f"   - Modelo LLM: {system_info['generator']['model']}")
    print(f"   - Embeddings: {system_info['vector_store']['embedding_model']}")
    
    if doc_count == 0:
        print("\n⚠️  Base de datos vacía. Por favor ejecuta test_rag.py primero para cargar materiales.")
        return
    
    # Cargar dataset de evaluación
    print("\n⏳ Cargando dataset de evaluación...")
    try:
        evaluation_samples = load_evaluation_dataset()
        print(f"✅ Dataset cargado: {len(evaluation_samples)} muestras")
    except FileNotFoundError:
        print("❌ ERROR: No se encontró el archivo data/evaluation_dataset.json")
        return
    
    # Preguntar cuántas muestras evaluar
    print("\n" + "="*70)
    print("📝 CONFIGURACIÓN DE EVALUACIÓN")
    print("="*70)
    print(f"\nMuestras disponibles: {len(evaluation_samples)}")
    print("Opciones:")
    print("  1. Evaluar todas las muestras (puede tomar varios minutos)")
    print("  2. Evaluar solo 5 muestras (rápido, para prueba)")
    print("  3. Evaluar solo 10 muestras")
    
    # Por defecto evaluar 10 para no gastar muchos tokens
    max_samples = 10
    print(f"\n▶️  Evaluando {max_samples} muestras...")
    
    # Ejecutar evaluación
    results = run_evaluation(
        rag_pipeline=rag_pipeline,
        evaluation_samples=evaluation_samples,
        max_samples=max_samples,
        save_results=True
    )
    
    if results['status'] == 'success':
        print("\n" + "="*70)
        print("✅ EVALUACIÓN COMPLETADA EXITOSAMENTE")
        print("="*70)
        
        stats = results['stats']
        print(f"\n📊 Resumen de Métricas:")
        for metric_name, scores in stats['metrics_scores'].items():
            print(f"   - {metric_name}: {scores['mean']:.4f} (±{scores['std']:.4f})")
        
        print("\n💡 Archivos generados en ./data/:")
        print("   - evaluation_report_*.txt (reporte legible)")
        print("   - evaluation_results_*.csv (resultados detallados)")
        print("   - evaluation_data_*.json (datos completos)")
    else:
        print("\n❌ Error en la evaluación")


if __name__ == "__main__":
    main()

