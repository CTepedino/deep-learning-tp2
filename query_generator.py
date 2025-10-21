#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constructor de consultas interactivo
Permite construir consultas paso a paso mediante menús
"""

import json
import os
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from src.rag_pipeline import create_rag_pipeline
from src.export_utils import ExerciseExporter

class QueryBuilder:
    def __init__(self):
        self.query = {}
        self.materias = [
            "Sistemas de Inteligencia Artificial",
            "Probabilidad y Estadística"
        ]
        
        self.unidades = {
            "Sistemas de Inteligencia Artificial": [
                "Introducción",
                "Agentes y Ambientes",
                "Métodos de búsqueda desinformados",
                "Métodos de Búsqueda Informados",
                "Algorítmos de Mejoramiento Iterativo",
                "Algoritmos Genéticos",
                "Optimización No Lineal",
                "Perceptrón Simple Escalón",
                "Perceptrón Lineal y No Lineal",
                "Perceptrón Multicapa",
                "Métricas y Sobreajuste",
                "Introducción Aprendizaje No Supervisado",
                "Modelo de Kohonen",
                "Autovalores y Autovectores",
                "PCA",
                "Regla de Oja y Sanger",
                "Modelo de Hopfield",
                "Deep Learning",
                "Autoencoders",
                "Redes Neuronales Convolucionales",
                "Redes Generativas Adversariales",
                "Transformers"
            ],
            "Probabilidad y Estadística": [
                "Introducción",
                "Estadística Descriptiva",
                "Axiomas",
                "Laplace",
                "Variable Aleatoria Discreta",
                "Variables Aleatorias Discretas Notables",
                "Variable Aleatoria Continua",
                "Distribución Normal",
                "Variable Aleatoria Continua (2)",
                "Función de Variable Aleatoria",
                "Mezcla de Variables Aleatorias",
                "Variables Aleatorias Bidimensionales Discretas",
                "Variables Aleatorias Bidimensionales Continuas",
                "Suma de Variables Aleatorias Independientes",
                "Teorema Central Del Límite",
                "Estimación",
                "Intervalos de Confianza",
                "Intervalos de Confianza para la Media - Distribución T",
                "Tesis de Hipótesis",
                "Tesis de Hipótesis (2)",
                "Tesis de Hipótesis sobre la media con desvío desconocido",
                "Tesis de Hipótesis sobre la proporción poblacional",
                "Procesos Estocásticos",
                "Cadenas de Markov"
            ]
        }
        
        self.niveles_dificultad = [
            "basico",
            "intermedio", 
            "avanzado"
        ]
        
        self.tipos_ejercicio = [
            "multiple_choice",
            "desarrollo",
            "practico",
            "teorico"
        ]
        
        self.formatos = [
            "txt",
            "pdf",
            "tex"
        ]

    def mostrar_paso(self, paso, total):
        """Muestra el progreso del constructor"""
        print("\n" + "="*60)
        print(f"    CONSTRUCTOR DE CONSULTAS - PASO {paso}/{total}")
        print("="*60)

    def seleccionar_materia(self):
        """Permite seleccionar una materia de la lista"""
        print("📚 MATERIAS DISPONIBLES:")
        for i, materia in enumerate(self.materias, 1):
            print(f"   {i}) {materia}")
        
        while True:
            try:
                opcion = int(input("\n👉 Ingrese el número de la materia: "))
                if 1 <= opcion <= len(self.materias):
                    self.query['materia'] = self.materias[opcion - 1]
                    print(f"✅ Materia seleccionada: {self.materias[opcion - 1]}")
                    return
                else:
                    print("❌ Opción inválida. Intente nuevamente.")
            except ValueError:
                print("❌ Por favor ingrese un número válido.")

    def seleccionar_unidad(self):
        """Permite seleccionar una unidad de la materia seleccionada"""
        if 'materia' not in self.query:
            print("❌ Primero debe seleccionar una materia.")
            return
        
        print(f"📖 UNIDADES DE {self.query['materia'].upper()}:")
        unidades = self.unidades.get(self.query['materia'], ["Unidad no disponible"])
        
        for i, unidad in enumerate(unidades, 1):
            print(f"   {i}) {unidad}")
        
        while True:
            try:
                opcion = int(input("\n👉 Ingrese el número de la unidad: "))
                if 1 <= opcion <= len(unidades):
                    self.query['unidad'] = unidades[opcion - 1]
                    print(f"✅ Unidad seleccionada: {unidades[opcion - 1]}")
                    return
                else:
                    print("❌ Opción inválida. Intente nuevamente.")
            except ValueError:
                print("❌ Por favor ingrese un número válido.")

    def seleccionar_cantidad(self):
        """Permite seleccionar la cantidad de ejercicios"""
        print("🔢 CANTIDAD DE EJERCICIOS:")
        print("   Ingresa la cantidad de ejercicios que deseas generar")
        
        while True:
            try:
                cantidad = int(input("\n👉 Ingrese la cantidad de ejercicios: "))
                if cantidad > 0:
                    self.query['cantidad'] = cantidad
                    print(f"✅ Cantidad seleccionada: {cantidad} ejercicios")
                    return
                else:
                    print("❌ La cantidad debe ser mayor a 0.")
            except ValueError:
                print("❌ Por favor ingrese un número válido.")

    def seleccionar_nivel_dificultad(self):
        """Permite seleccionar el nivel de dificultad"""
        print("📊 NIVEL DE DIFICULTAD:")
        for i, nivel in enumerate(self.niveles_dificultad, 1):
            print(f"   {i}) {nivel}")
        
        while True:
            try:
                opcion = int(input("\n👉 Ingrese el número del nivel: "))
                if 1 <= opcion <= len(self.niveles_dificultad):
                    self.query['nivel_dificultad'] = self.niveles_dificultad[opcion - 1]
                    print(f"✅ Nivel seleccionado: {self.niveles_dificultad[opcion - 1]}")
                    return
                else:
                    print("❌ Opción inválida. Intente nuevamente.")
            except ValueError:
                print("❌ Por favor ingrese un número válido.")

    def seleccionar_tipo_ejercicio(self):
        """Permite seleccionar el tipo de ejercicio"""
        print("📝 TIPO DE EJERCICIO:")
        for i, tipo in enumerate(self.tipos_ejercicio, 1):
            print(f"   {i}) {tipo}")
        
        while True:
            try:
                opcion = int(input("\n👉 Ingrese el número del tipo: "))
                if 1 <= opcion <= len(self.tipos_ejercicio):
                    self.query['tipo_ejercicio'] = self.tipos_ejercicio[opcion - 1]
                    print(f"✅ Tipo seleccionado: {self.tipos_ejercicio[opcion - 1]}")
                    return
                else:
                    print("❌ Opción inválida. Intente nuevamente.")
            except ValueError:
                print("❌ Por favor ingrese un número válido.")

    def seleccionar_formato(self):
        """Permite seleccionar el formato de salida"""
        print("📄 FORMATO DE SALIDA:")
        for i, formato in enumerate(self.formatos, 1):
            print(f"   {i}) {formato}")
        
        while True:
            try:
                opcion = int(input("\n👉 Ingrese el número del formato: "))
                if 1 <= opcion <= len(self.formatos):
                    self.query['formato'] = self.formatos[opcion - 1]
                    print(f"✅ Formato seleccionado: {self.formatos[opcion - 1]}")
                    return
                else:
                    print("❌ Opción inválida. Intente nuevamente.")
            except ValueError:
                print("❌ Por favor ingrese un número válido.")

    def mostrar_resumen_final(self):
        """Muestra el resumen final de la consulta"""
        print("\n" + "="*60)
        print("    🎯 RESUMEN DE LA CONSULTA GENERADA")
        print("="*60)
        for clave, valor in self.query.items():
            print(f"   📋 {clave.replace('_', ' ').title()}: {valor}")
        print("="*60)

    def generar_ejercicios(self):
        """Genera los ejercicios directamente usando el sistema RAG"""
        print("\n" + "="*60)
        print("    🚀 GENERANDO EJERCICIOS...")
        print("="*60)
        
        try:
            # Crear pipeline RAG
            print("📦 Cargando sistema RAG...")
            rag = create_rag_pipeline()
            
            # Generar ejercicios
            print(f"🎲 Generando {self.query.get('cantidad', 1)} ejercicio(s)...")
            result = rag.generate_exercises(query_params=self.query)
            
            if result.get('ejercicios') and len(result.get('ejercicios', [])) > 0:
                print("✅ ¡Ejercicios generados exitosamente!")
                
                # Exportar archivos
                formato = self.query.get('formato', 'txt')
                print(f"💾 Exportando a formato {formato.upper()}...")
                exporter = ExerciseExporter(output_directory="./output")
                archivos = exporter.export_all_versions(
                    result=result,
                    format=formato
                )
                
                print(f"\n✅ Archivos generados en: {archivos['session_folder']}")
                print("   📄 Completo (todo)")
                print("   📝 Ejercicio (solo preguntas)")
                print("   💡 Pistas")
                print("   ✅ Soluciones")
                
                return True
                
            else:
                print(f"❌ Error: {result.get('message', 'Error desconocido')}")
                print(f"🔍 Debug - Resultado completo: {result}")
                return False
                
        except Exception as e:
            print(f"❌ Error durante la generación: {str(e)}")
            return False

    def ejecutar(self):
        """Ejecuta el constructor de consultas paso a paso"""
        print("\n" + "="*60)
        print("    🚀 CONSTRUCTOR DE CONSULTAS INTERACTIVO")
        print("="*60)
        print("   Te guiaré paso a paso para crear tu consulta")
        print("="*60)
        
        # Paso 1: Seleccionar Materia
        self.mostrar_paso(1, 6)
        self.seleccionar_materia()
        
        # Paso 2: Seleccionar Unidad
        self.mostrar_paso(2, 6)
        self.seleccionar_unidad()
        
        # Paso 3: Seleccionar Cantidad
        self.mostrar_paso(3, 6)
        self.seleccionar_cantidad()
        
        # Paso 4: Seleccionar Nivel de Dificultad
        self.mostrar_paso(4, 6)
        self.seleccionar_nivel_dificultad()
        
        # Paso 5: Seleccionar Tipo de Ejercicio
        self.mostrar_paso(5, 6)
        self.seleccionar_tipo_ejercicio()
        
        # Paso 6: Seleccionar Formato
        self.mostrar_paso(6, 6)
        self.seleccionar_formato()
        
        # Mostrar resumen final
        self.mostrar_resumen_final()
        
        # Generar ejercicios directamente
        exito = self.generar_ejercicios()
        
        if exito:
            print(f"\n🎉 ¡Ejercicios generados exitosamente!")
            print("📁 Revisa la carpeta 'output' para ver los archivos generados")
        else:
            print("\n❌ Hubo un problema al generar los ejercicios.")
        
        print("\n👋 ¡Hasta luego!")

if __name__ == "__main__":
    builder = QueryBuilder()
    builder.ejecutar()
