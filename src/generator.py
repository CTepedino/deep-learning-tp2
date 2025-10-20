"""
Generator Module for RAG System
Generación de ejercicios usando LangChain con ChatOpenAI y PromptTemplates
"""

import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

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
        Inicializa el generador de ejercicios con LangChain
        
        Args:
            model_name: Nombre del modelo de OpenAI (por defecto de LLM_MODEL)
            temperature: Temperatura para la generación
            max_tokens: Máximo número de tokens
        """
        # Leer valores de variables de entorno si no se especifican
        self.model_name = model_name or os.getenv('LLM_MODEL', 'gpt-4o-mini')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Inicializar LLM de LangChain
        self.llm = None
        self._initialize_langchain_llm()
        
        # Configurar prompts y output parsers
        self._setup_prompts()
        self._setup_output_parsers()
    
    def _initialize_langchain_llm(self):
        """Inicializa el LLM de LangChain con ChatOpenAI"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY no encontrada en variables de entorno")
            
            # Configurar headers adicionales si se especifica modo
            mode = os.getenv('OPENAI_MODE', 'standard').lower()
            valid_modes = {"flex", "standard", "priority"}
            
            if mode not in valid_modes:
                logger.warning(f"Modo OPENAI_MODE '{mode}' no válido. Usando 'standard' por defecto.")
                mode = 'standard'
            
            # Crear ChatOpenAI con LangChain
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                model_kwargs={
                    "extra_headers": {"x-openai-pricing-tier": mode}
                }
            )
            
            logger.info(f"LangChain ChatOpenAI inicializado con modelo: {self.model_name} y modo: {mode}")
            
        except ImportError:
            logger.error("langchain-openai no está instalado. Instalar con: pip install langchain-openai")
            raise
        except Exception as e:
            logger.error(f"Error inicializando LangChain LLM: {str(e)}")
            raise
    
    def _setup_prompts(self):
        """Configura los templates de prompts con LangChain"""
        self.system_prompt_text = """Eres un experto generador de ejercicios académicos para la carrera de Ingeniería Informática del ITBA. 
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
    
    def _setup_output_parsers(self):
        """Configura los output parsers de LangChain"""
        # Parser para JSON genérico
        self.json_parser = JsonOutputParser()
        
        # Crear chains para cada tipo de ejercicio usando LCEL
        self.chains = {}
        for tipo_ejercicio, config in self.exercise_templates.items():
            # Crear prompt template para cada tipo
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt_text),
                ("human", config["template"])
            ])
            
            # Crear chain: prompt | llm | json_parser
            self.chains[tipo_ejercicio] = prompt | self.llm | self.json_parser
            
        logger.info(f"Output parsers y chains configurados para {len(self.chains)} tipos de ejercicios")
    
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
            
            # Preparar variables para el prompt
            prompt_vars = self._prepare_prompt_variables(
                query_params=query_params,
                contexto=contexto,
                tipo_ejercicio=tipo_ejercicio
            )
            
            # Generar ejercicios usando el chain de LangChain
            response_data = self._invoke_chain(tipo_ejercicio, prompt_vars)
            
            # Validar y procesar respuesta
            exercises = self._validate_exercises(response_data, tipo_ejercicio)
            
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
    
    def _prepare_prompt_variables(
        self,
        query_params: Dict[str, Any],
        contexto: str,
        tipo_ejercicio: str
    ) -> Dict[str, Any]:
        """
        Prepara las variables para el prompt template de LangChain
        
        Args:
            query_params: Parámetros de la consulta
            contexto: Contexto formateado
            tipo_ejercicio: Tipo de ejercicio
            
        Returns:
            Diccionario con variables para el prompt
        """
        # Extraer parámetros
        materia = query_params.get("materia", "materia no especificada")
        unidad = query_params.get("unidad", "tema no especificado")
        cantidad = query_params.get("cantidad", 1)
        nivel_dificultad = query_params.get("nivel_dificultad", "intermedio")
        
        # Retornar diccionario con variables
        return {
            "cantidad": cantidad,
            "tema": unidad,
            "materia": materia,
            "contexto": contexto,
            "nivel_dificultad": nivel_dificultad
        }
    
    def _invoke_chain(
        self, 
        tipo_ejercicio: str, 
        prompt_vars: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoca el chain de LangChain para generar ejercicios
        
        Args:
            tipo_ejercicio: Tipo de ejercicio a generar
            prompt_vars: Variables para el prompt
            
        Returns:
            Datos parseados de la respuesta (diccionario)
        """
        try:
            # Invocar el chain usando LCEL
            chain = self.chains[tipo_ejercicio]
            result = chain.invoke(prompt_vars)
            
            logger.info(f"Chain invocado exitosamente para tipo: {tipo_ejercicio}")
            return result
            
        except Exception as e:
            logger.error(f"Error invocando chain de LangChain: {str(e)}")
            raise
    
    def _validate_exercises(
        self,
        response_data: Dict[str, Any],
        tipo_ejercicio: str
    ) -> List[Dict[str, Any]]:
        """
        Valida los ejercicios generados por el chain de LangChain
        
        Args:
            response_data: Datos parseados por JsonOutputParser
            tipo_ejercicio: Tipo de ejercicio
            
        Returns:
            Lista de ejercicios validados
        """
        try:
            # Validar estructura
            if "ejercicios" not in response_data:
                raise ValueError("Respuesta no contiene campo 'ejercicios'")
            
            exercises = response_data["ejercicios"]
            
            # Validar cada ejercicio
            validator = self.exercise_templates[tipo_ejercicio]["validation"]
            validated_exercises = []
            
            for exercise in exercises:
                if validator(exercise):
                    validated_exercises.append(exercise)
                else:
                    logger.warning(f"Ejercicio no válido omitido: {exercise}")
            
            logger.info(f"Ejercicios validados: {len(validated_exercises)} de {len(exercises)}")
            return validated_exercises
            
        except Exception as e:
            logger.error(f"Error validando ejercicios: {str(e)}")
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
