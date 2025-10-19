"""
Generator Module for RAG System
Generación de ejercicios usando OpenAI API con prompt engineering
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

logger = logging.getLogger(__name__)


class ExerciseGenerator:
    """
    Generador de ejercicios académicos usando LLM
    """
    
    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ):
        """
        Inicializa el generador de ejercicios
        
        Args:
            model_name: Nombre del modelo de OpenAI (por defecto de LLM_MODEL)
            temperature: Temperatura para la generación
            max_tokens: Máximo número de tokens
        """
        # Leer valores de variables de entorno si no se especifican
        self.model_name = model_name or os.getenv('LLM_MODEL', 'gpt-4o-mini')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        
        self._initialize_openai_client()
        self._setup_prompts()
    
    def _initialize_openai_client(self):
        """Inicializa el cliente de OpenAI
        
        El modo de capacidad se configura mediante la variable de entorno OPENAI_MODE
        (valores válidos: flex, standard, priority). Por defecto usa 'standard'. 
        """
        try:
            from openai import OpenAI
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY no encontrada en variables de entorno")
            
            # Leer y validar el modo de capacidad
            mode = os.getenv('OPENAI_MODE', 'standard').lower()
            valid_modes = {"flex", "standard", "priority"}
            
            if mode not in valid_modes:
                logger.warning(f"Modo OPENAI_MODE '{mode}' no válido. Usando 'standard' por defecto.")
                mode = 'standard'
            
            # Configurar headers para el modo de capacidad
            self.extra_headers = {"x-openai-pricing-tier": mode}
            
            self.client = OpenAI(api_key=api_key)
            logger.info(f"Cliente OpenAI inicializado con modelo: {self.model_name} y modo: {mode}")
            
        except ImportError:
            logger.error("openai no está instalado. Instalar con: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Error inicializando cliente OpenAI: {str(e)}")
            raise
    
    def _setup_prompts(self):
        """Configura los templates de prompts"""
        self.system_prompt = """Eres un experto generador de ejercicios académicos para la carrera de Ingeniería Informática del ITBA. 
Tu tarea es crear ejercicios de alta calidad basados en el material académico proporcionado.

REGLAS IMPORTANTES:
1. Los ejercicios deben estar estrictamente dentro del plan de estudios del ITBA
2. Deben ser apropiados para el nivel de dificultad especificado
3. Deben basarse únicamente en el contexto proporcionado
4. La respuesta debe ser en formato JSON válido
5. Incluye soluciones detalladas y pistas útiles
6. Asegúrate de que los ejercicios sean originales y no copias directas del material

TIPOS DE EJERCICIOS SOPORTADOS:
- multiple_choice: Preguntas de opción múltiple (A, B, C, D)
- desarrollo: Preguntas de desarrollo teórico
- practico: Problemas prácticos con cálculos
- teorico: Preguntas conceptuales teóricas"""

        self.exercise_templates = {
            "multiple_choice": {
                "template": """Genera {cantidad} ejercicio(s) de opción múltiple sobre {tema} en {materia}.

Contexto académico:
{contexto}

Nivel de dificultad: {nivel_dificultad}

Formato de respuesta (JSON):
{{
  "ejercicios": [
    {{
      "pregunta": "Pregunta clara y específica",
      "opciones": ["Opción A", "Opción B", "Opción C", "Opción D"],
      "respuesta_correcta": "C",
      "pista": "Pista útil para resolver el ejercicio",
      "solucion": "Explicación detallada de por qué la respuesta es correcta"
    }}
  ]
}}""",
                "validation": self._validate_multiple_choice
            },
            
            "desarrollo": {
                "template": """Genera {cantidad} ejercicio(s) de desarrollo sobre {tema} en {materia}.

Contexto académico:
{contexto}

Nivel de dificultad: {nivel_dificultad}

Formato de respuesta (JSON):
{{
  "ejercicios": [
    {{
      "pregunta": "Pregunta que requiere desarrollo teórico",
      "pista": "Pista sobre cómo abordar el problema",
      "solucion": "Solución detallada paso a paso",
      "puntos_clave": ["Punto clave 1", "Punto clave 2", "Punto clave 3"]
    }}
  ]
}}""",
                "validation": self._validate_desarrollo
            },
            
            "practico": {
                "template": """Genera {cantidad} ejercicio(s) práctico(s) sobre {tema} en {materia}.

Contexto académico:
{contexto}

Nivel de dificultad: {nivel_dificultad}

Formato de respuesta (JSON):
{{
  "ejercicios": [
    {{
      "pregunta": "Problema práctico con datos específicos",
      "pista": "Pista sobre el método de resolución",
      "solucion": "Solución paso a paso con cálculos",
      "datos": "Datos necesarios para resolver el problema"
    }}
  ]
}}""",
                "validation": self._validate_practico
            },
            
            "teorico": {
                "template": """Genera {cantidad} ejercicio(s) teórico(s) sobre {tema} en {materia}.

Contexto académico:
{contexto}

Nivel de dificultad: {nivel_dificultad}

Formato de respuesta (JSON):
{{
  "ejercicios": [
    {{
      "pregunta": "Pregunta conceptual teórica",
      "pista": "Pista sobre el concepto clave",
      "solucion": "Explicación teórica detallada",
      "conceptos_clave": ["Concepto 1", "Concepto 2", "Concepto 3"]
    }}
  ]
}}""",
                "validation": self._validate_teorico
            }
        }
    
    def generate_exercises(
        self,
        query_params: Dict[str, Any],
        context_documents: List[Any],
        tipo_ejercicio: str = "multiple_choice"
    ) -> Dict[str, Any]:
        """
        Genera ejercicios basados en los parámetros y contexto
        
        Args:
            query_params: Parámetros de la consulta (materia, unidad, cantidad, etc.)
            context_documents: Documentos recuperados como contexto
            tipo_ejercicio: Tipo de ejercicio a generar
            
        Returns:
            Diccionario con ejercicios generados
        """
        try:
            # Validar tipo de ejercicio
            if tipo_ejercicio not in self.exercise_templates:
                raise ValueError(f"Tipo de ejercicio no soportado: {tipo_ejercicio}")
            
            # Preparar contexto
            contexto = self._prepare_context(context_documents)
            
            # Preparar prompt
            prompt = self._prepare_prompt(
                query_params=query_params,
                contexto=contexto,
                tipo_ejercicio=tipo_ejercicio
            )
            
            # Generar ejercicios
            response = self._call_openai(prompt)
            
            # Validar y procesar respuesta
            exercises = self._process_response(response, tipo_ejercicio)
            
            # Agregar metadata
            result = {
                "ejercicios": exercises,
                "metadata": {
                    "materia": query_params.get("materia"),
                    "unidad": query_params.get("unidad"),
                    "tipo_ejercicio": tipo_ejercicio,
                    "cantidad": query_params.get("cantidad", 1),
                    "nivel_dificultad": query_params.get("nivel_dificultad"),
                    "chunks_recuperados": len(context_documents),
                    "fuentes": [doc.metadata.get("source", "desconocido") for doc in context_documents],
                    "modelo_usado": self.model_name
                }
            }
            
            logger.info(f"Ejercicios generados: {len(exercises)} de tipo {tipo_ejercicio}")
            return result
            
        except Exception as e:
            logger.error(f"Error generando ejercicios: {str(e)}")
            raise
    
    def _prepare_context(self, context_documents: List[Any]) -> str:
        """
        Prepara el contexto a partir de los documentos recuperados
        
        Args:
            context_documents: Lista de documentos
            
        Returns:
            Contexto formateado como string
        """
        if not context_documents:
            return "No se encontró contexto relevante."
        
        context_parts = []
        for i, doc in enumerate(context_documents, 1):
            content = doc.page_content.strip()
            source = doc.metadata.get("source", f"Documento {i}")
            context_parts.append(f"[Fuente {i}: {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _prepare_prompt(
        self,
        query_params: Dict[str, Any],
        contexto: str,
        tipo_ejercicio: str
    ) -> str:
        """
        Prepara el prompt para la generación
        
        Args:
            query_params: Parámetros de la consulta
            contexto: Contexto formateado
            tipo_ejercicio: Tipo de ejercicio
            
        Returns:
            Prompt formateado
        """
        template = self.exercise_templates[tipo_ejercicio]["template"]
        
        # Extraer parámetros
        materia = query_params.get("materia", "materia no especificada")
        unidad = query_params.get("unidad", "tema no especificado")
        cantidad = query_params.get("cantidad", 1)
        nivel_dificultad = query_params.get("nivel_dificultad", "intermedio")
        
        # Formatear prompt
        prompt = template.format(
            cantidad=cantidad,
            tema=unidad,
            materia=materia,
            contexto=contexto,
            nivel_dificultad=nivel_dificultad
        )
        
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """
        Llama a la API de OpenAI
        
        Args:
            prompt: Prompt a enviar
            
        Returns:
            Respuesta de la API
        """
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                extra_headers=self.extra_headers
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error llamando a OpenAI: {str(e)}")
            raise
    
    def _process_response(
        self,
        response: str,
        tipo_ejercicio: str
    ) -> List[Dict[str, Any]]:
        """
        Procesa la respuesta de OpenAI
        
        Args:
            response: Respuesta de la API
            tipo_ejercicio: Tipo de ejercicio
            
        Returns:
            Lista de ejercicios procesados
        """
        try:
            # Extraer JSON de la respuesta
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No se encontró JSON válido en la respuesta")
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            # Validar estructura
            if "ejercicios" not in data:
                raise ValueError("Respuesta no contiene campo 'ejercicios'")
            
            exercises = data["ejercicios"]
            
            # Validar cada ejercicio
            validator = self.exercise_templates[tipo_ejercicio]["validation"]
            validated_exercises = []
            
            for exercise in exercises:
                if validator(exercise):
                    validated_exercises.append(exercise)
                else:
                    logger.warning(f"Ejercicio no válido omitido: {exercise}")
            
            return validated_exercises
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error procesando respuesta: {str(e)}")
            raise
    
    def _validate_multiple_choice(self, exercise: Dict[str, Any]) -> bool:
        """Valida un ejercicio de opción múltiple"""
        required_fields = ["pregunta", "opciones", "respuesta_correcta", "pista", "solucion"]
        return all(field in exercise for field in required_fields) and len(exercise.get("opciones", [])) == 4
    
    def _validate_desarrollo(self, exercise: Dict[str, Any]) -> bool:
        """Valida un ejercicio de desarrollo"""
        required_fields = ["pregunta", "pista", "solucion", "puntos_clave"]
        return all(field in exercise for field in required_fields)
    
    def _validate_practico(self, exercise: Dict[str, Any]) -> bool:
        """Valida un ejercicio práctico"""
        required_fields = ["pregunta", "pista", "solucion", "datos"]
        return all(field in exercise for field in required_fields)
    
    def _validate_teorico(self, exercise: Dict[str, Any]) -> bool:
        """Valida un ejercicio teórico"""
        required_fields = ["pregunta", "pista", "solucion", "conceptos_clave"]
        return all(field in exercise for field in required_fields)
    
    def set_model(self, model_name: str):
        """Actualiza el modelo a usar"""
        self.model_name = model_name
        logger.info(f"Modelo actualizado a: {model_name}")
    
    def set_temperature(self, temperature: float):
        """Actualiza la temperatura"""
        self.temperature = temperature
        logger.info(f"Temperatura actualizada a: {temperature}")


def create_generator(
    model_name: str = None,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> ExerciseGenerator:
    """
    Función de conveniencia para crear un generador
    
    Args:
        model_name: Nombre del modelo
        temperature: Temperatura
        max_tokens: Máximo número de tokens
        
    Returns:
        Instancia del generador
    """
    if model_name is None:
        model_name = os.getenv('LLM_MODEL', 'gpt-4o-mini')
    
    return ExerciseGenerator(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Crear generador
    generator = create_generator()
    
    # Crear documentos de contexto de ejemplo
    from langchain_core.documents import Document
    
    context_docs = [
        Document(
            page_content="La distribución normal es una distribución de probabilidad continua con forma de campana.",
            metadata={"source": "apunte_probabilidad.pdf"}
        ),
        Document(
            page_content="Para calcular probabilidades en la distribución normal estándar se usan tablas o software estadístico.",
            metadata={"source": "guia_ejercicios.pdf"}
        )
    ]
    
    # Parámetros de consulta
    query_params = {
        "materia": "Probabilidad y estadística",
        "unidad": "Distribución normal",
        "cantidad": 2,
        "nivel_dificultad": "intermedio"
    }
    
    # Generar ejercicios de opción múltiple
    print("Generando ejercicios de opción múltiple...")
    try:
        result = generator.generate_exercises(
            query_params=query_params,
            context_documents=context_docs,
            tipo_ejercicio="multiple_choice"
        )
        
        print(f"Ejercicios generados: {len(result['ejercicios'])}")
        for i, exercise in enumerate(result['ejercicios'], 1):
            print(f"\nEjercicio {i}:")
            print(f"Pregunta: {exercise['pregunta']}")
            print(f"Opciones: {exercise['opciones']}")
            print(f"Respuesta correcta: {exercise['respuesta_correcta']}")
            print(f"Pista: {exercise['pista']}")
        
        print(f"\nMetadata: {result['metadata']}")
        
    except Exception as e:
        print(f"Error generando ejercicios: {str(e)}")
    
    # Generar ejercicios de desarrollo
    print("\n" + "="*50)
    print("Generando ejercicios de desarrollo...")
    try:
        result = generator.generate_exercises(
            query_params=query_params,
            context_documents=context_docs,
            tipo_ejercicio="desarrollo"
        )
        
        print(f"Ejercicios generados: {len(result['ejercicios'])}")
        for i, exercise in enumerate(result['ejercicios'], 1):
            print(f"\nEjercicio {i}:")
            print(f"Pregunta: {exercise['pregunta']}")
            print(f"Pista: {exercise['pista']}")
            print(f"Puntos clave: {exercise['puntos_clave']}")
        
    except Exception as e:
        print(f"Error generando ejercicios: {str(e)}")
