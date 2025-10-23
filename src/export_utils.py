"""
Export Utilities Module
Exportación de ejercicios a diferentes formatos (.txt, .pdf, .tex)
"""

import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ExerciseExporter:
    """
    Exportador de ejercicios a múltiples formatos
    """
    
    def __init__(self, output_directory: str = "./output"):
        """
        Inicializa el exportador
        
        Args:
            output_directory: Directorio donde guardar los archivos
        """
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
    
    def _format_multiple_choice_options(self, opciones: List[str]) -> List[str]:
        """
        Formatea las opciones de multiple choice, evitando duplicación de letras
        
        Args:
            opciones: Lista de opciones
            
        Returns:
            Lista de opciones formateadas con letras A), B), C), D)
        """
        letras = ['A', 'B', 'C', 'D']
        opciones_formateadas = []
        
        for j, opcion in enumerate(opciones):
            # Verificar si la opción ya tiene una letra al inicio
            if j < len(letras) and opcion.strip().startswith(f"{letras[j]})"):
                # Ya tiene letra, usar tal como está
                opciones_formateadas.append(opcion)
            elif j < len(letras) and opcion.strip().startswith(f"{letras[j]}."):
                # Tiene formato A. en lugar de A), convertir
                opcion_clean = opcion.strip()[2:].strip()
                opciones_formateadas.append(f"{letras[j]}) {opcion_clean}")
            else:
                # No tiene letra, agregarla
                opciones_formateadas.append(f"{letras[j]}) {opcion}")
        
        return opciones_formateadas
    
    def export(
        self,
        result: Dict[str, Any],
        format: str = "txt",
        filename: Optional[str] = None,
        include_hints: bool = True,
        include_solutions: bool = True
    ) -> str:
        """
        Exporta ejercicios al formato especificado
        
        Args:
            result: Resultado de generate_exercises() con ejercicios y metadata
            format: Formato de exportación ('txt', 'pdf', 'tex')
            filename: Nombre del archivo (sin extensión). Si es None, se genera automáticamente
            include_hints: Si incluir pistas en la exportación
            include_solutions: Si incluir soluciones en la exportación
            
        Returns:
            Ruta del archivo generado
        """
        format = format.lower()
        
        if format not in ['txt', 'pdf', 'tex']:
            raise ValueError(f"Formato no soportado: {format}. Use 'txt', 'pdf' o 'tex'")
        
        # Generar nombre de archivo si no se proporciona
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            materia = result.get('metadata', {}).get('materia', 'ejercicios')
            materia_clean = materia.replace(' ', '_').replace('/', '_')
            filename = f"{materia_clean}_{timestamp}"
        
        # Exportar según formato
        if format == 'txt':
            filepath = self._export_txt(result, filename, include_hints, include_solutions)
        elif format == 'pdf':
            filepath = self._export_pdf(result, filename, include_hints, include_solutions)
        elif format == 'tex':
            filepath = self._export_tex(result, filename, include_hints, include_solutions)
        
        logger.info(f"Ejercicios exportados a: {filepath}")
        return str(filepath)
    
    def export_all_versions(
        self,
        result: Dict[str, Any],
        format: str = "txt",
        base_filename: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Exporta todas las versiones del ejercicio (completo, solo ejercicio, pista, solución)
        en una carpeta con timestamp para evitar sobrescribir archivos
        
        Args:
            result: Resultado de generate_exercises()
            format: Formato de exportación
            base_filename: Nombre base (sin extensión ni sufijos)
            
        Returns:
            Diccionario con las rutas de todos los archivos generados
        """
        format = format.lower()
        
        # Crear carpeta con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_folder = self.output_directory / timestamp
        session_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creando sesión de exportación en: {session_folder}")
        
        # Generar nombre base si no se proporciona
        if base_filename is None:
            materia = result.get('metadata', {}).get('materia', 'ejercicios')
            materia_clean = materia.replace(' ', '_').replace('/', '_')
            base_filename = f"{materia_clean}"
        
        # Guardar el output_directory original
        original_output_dir = self.output_directory
        # Cambiar temporalmente al directorio de la sesión
        self.output_directory = session_folder
        
        archivos = {}
        
        # 1. Versión completa (todo)
        filepath_completo = self.export(
            result, 
            format=format, 
            filename=f"{base_filename}_completo",
            include_hints=True,
            include_solutions=True
        )
        archivos['completo'] = filepath_completo
        
        # 2. Solo ejercicio (sin pistas ni soluciones)
        filepath_ejercicio = self.export(
            result, 
            format=format, 
            filename=f"{base_filename}_ejercicio",
            include_hints=False,
            include_solutions=False
        )
        archivos['ejercicio'] = filepath_ejercicio
        
        # 3. Solo pistas
        filepath_pistas = self._export_only_hints(result, format, f"{base_filename}_pistas")
        archivos['pistas'] = filepath_pistas
        
        # 4. Solo soluciones
        filepath_soluciones = self._export_only_solutions(result, format, f"{base_filename}_soluciones")
        archivos['soluciones'] = filepath_soluciones
        
        # Restaurar el directorio original
        self.output_directory = original_output_dir
        
        logger.info(f"Generados {len(archivos)} archivos en formato {format.upper()} en {session_folder}")
        
        # Agregar información de la carpeta de sesión
        archivos['session_folder'] = str(session_folder)
        
        return archivos
    
    def _export_txt(self, result: Dict[str, Any], filename: str, include_hints: bool = True, include_solutions: bool = True) -> Path:
        """Exporta a formato TXT"""
        filepath = self.output_directory / f"{filename}.txt"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("EJERCICIOS ACADÉMICOS GENERADOS CON RAG\n")
            f.write("=" * 80 + "\n\n")
            
            # Metadata
            metadata = result.get('metadata', {})
            f.write("INFORMACIÓN:\n")
            f.write("-" * 80 + "\n")
            if metadata.get('materia'):
                f.write(f"Materia: {metadata['materia']}\n")
            if metadata.get('unidad'):
                f.write(f"Unidad: {metadata['unidad']}\n")
            if metadata.get('tipo_ejercicio'):
                f.write(f"Tipo: {metadata['tipo_ejercicio']}\n")
            if metadata.get('nivel_dificultad'):
                f.write(f"Dificultad: {metadata['nivel_dificultad']}\n")
            if metadata.get('modelo_usado'):
                f.write(f"Modelo: {metadata['modelo_usado']}\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")
            
            # Ejercicios
            ejercicios = result.get('ejercicios', [])
            f.write(f"EJERCICIOS ({len(ejercicios)}):\n")
            f.write("=" * 80 + "\n\n")
            
            for i, ejercicio in enumerate(ejercicios, 1):
                f.write(f"EJERCICIO {i}\n")
                f.write("-" * 80 + "\n\n")
                
                # Pregunta
                f.write(f"PREGUNTA:\n{ejercicio['pregunta']}\n\n")
                
                # Según tipo de ejercicio
                if 'opciones' in ejercicio:
                    # Multiple choice
                    f.write("OPCIONES:\n")
                    opciones_formateadas = self._format_multiple_choice_options(ejercicio['opciones'])
                    for opcion in opciones_formateadas:
                        f.write(f"  {opcion}\n")
                    if include_solutions:
                        f.write(f"\nRESPUESTA CORRECTA: {ejercicio.get('respuesta_correcta', 'N/A')}\n")
                    else:
                        f.write("\n")
                
                if ejercicio.get('datos'):
                    f.write(f"\nDATOS:\n{ejercicio['datos']}\n")
                
                if include_hints and ejercicio.get('pista'):
                    f.write(f"\nPISTA:\n{ejercicio['pista']}\n")
                
                if include_solutions and ejercicio.get('solucion'):
                    f.write(f"\nSOLUCIÓN:\n{ejercicio['solucion']}\n")
                
                if ejercicio.get('puntos_clave'):
                    f.write(f"\nPUNTOS CLAVE:\n")
                    for punto in ejercicio['puntos_clave']:
                        f.write(f"  • {punto}\n")
                
                if ejercicio.get('conceptos_clave'):
                    f.write(f"\nCONCEPTOS CLAVE:\n")
                    for concepto in ejercicio['conceptos_clave']:
                        f.write(f"  • {concepto}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Footer
            f.write("\n")
            f.write("Generado con Sistema RAG - ITBA\n")
            if metadata.get('fuentes'):
                f.write(f"Basado en {len(metadata['fuentes'])} fuentes académicas\n")
        
        return filepath
    
    def _export_pdf(self, result: Dict[str, Any], filename: str, include_hints: bool = True, include_solutions: bool = True) -> Path:
        """Exporta a formato PDF"""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
            from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
            
            filepath = self.output_directory / f"{filename}.pdf"
            
            # Crear documento
            doc = SimpleDocTemplate(str(filepath), pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Estilos personalizados
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=16,
                textColor='darkblue',
                spaceAfter=20,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor='darkblue',
                spaceAfter=10
            )
            
            subheading_style = ParagraphStyle(
                'CustomSubheading',
                parent=styles['Heading3'],
                fontSize=12,
                textColor='navy',
                spaceAfter=8
            )
            
            # Título
            story.append(Paragraph("EJERCICIOS ACADÉMICOS", title_style))
            story.append(Spacer(1, 0.2 * inch))
            
            # Metadata
            metadata = result.get('metadata', {})
            info_text = "<b>Información del documento:</b><br/>"
            if metadata.get('materia'):
                info_text += f"<b>Materia:</b> {metadata['materia']}<br/>"
            if metadata.get('unidad'):
                info_text += f"<b>Unidad:</b> {metadata['unidad']}<br/>"
            if metadata.get('tipo_ejercicio'):
                info_text += f"<b>Tipo:</b> {metadata['tipo_ejercicio']}<br/>"
            if metadata.get('nivel_dificultad'):
                info_text += f"<b>Dificultad:</b> {metadata['nivel_dificultad']}<br/>"
            info_text += f"<b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            story.append(Paragraph(info_text, styles['Normal']))
            story.append(Spacer(1, 0.3 * inch))
            
            # Ejercicios
            ejercicios = result.get('ejercicios', [])
            
            for i, ejercicio in enumerate(ejercicios, 1):
                # Título del ejercicio
                story.append(Paragraph(f"Ejercicio {i}", heading_style))
                story.append(Spacer(1, 0.1 * inch))
                
                # Pregunta
                pregunta_text = f"<b>Pregunta:</b><br/>{ejercicio['pregunta']}"
                story.append(Paragraph(pregunta_text, styles['BodyText']))
                story.append(Spacer(1, 0.1 * inch))
                
                # Opciones (si es multiple choice)
                if 'opciones' in ejercicio:
                    story.append(Paragraph("<b>Opciones:</b>", subheading_style))
                    opciones_formateadas = self._format_multiple_choice_options(ejercicio['opciones'])
                    for opcion in opciones_formateadas:
                        story.append(Paragraph(f"&nbsp;&nbsp;{opcion}", styles['Normal']))
                    
                    if include_solutions:
                        respuesta = ejercicio.get('respuesta_correcta', 'N/A')
                        story.append(Spacer(1, 0.1 * inch))
                        story.append(Paragraph(f"<b>Respuesta correcta:</b> {respuesta}", styles['Normal']))
                
                # Datos (si existen)
                if ejercicio.get('datos'):
                    story.append(Spacer(1, 0.1 * inch))
                    story.append(Paragraph("<b>Datos:</b>", subheading_style))
                    story.append(Paragraph(ejercicio['datos'], styles['BodyText']))
                
                # Pista
                if include_hints and ejercicio.get('pista'):
                    story.append(Spacer(1, 0.1 * inch))
                    story.append(Paragraph("<b>Pista:</b>", subheading_style))
                    story.append(Paragraph(ejercicio['pista'], styles['BodyText']))
                
                # Solución
                if include_solutions and ejercicio.get('solucion'):
                    story.append(Spacer(1, 0.1 * inch))
                    story.append(Paragraph("<b>Solución:</b>", subheading_style))
                    story.append(Paragraph(ejercicio['solucion'], styles['BodyText']))
                
                # Puntos clave
                if ejercicio.get('puntos_clave'):
                    story.append(Spacer(1, 0.1 * inch))
                    story.append(Paragraph("<b>Puntos clave:</b>", subheading_style))
                    for punto in ejercicio['puntos_clave']:
                        story.append(Paragraph(f"• {punto}", styles['Normal']))
                
                # Conceptos clave
                if ejercicio.get('conceptos_clave'):
                    story.append(Spacer(1, 0.1 * inch))
                    story.append(Paragraph("<b>Conceptos clave:</b>", subheading_style))
                    for concepto in ejercicio['conceptos_clave']:
                        story.append(Paragraph(f"• {concepto}", styles['Normal']))
                
                # Espacio antes del siguiente ejercicio
                if i < len(ejercicios):
                    story.append(Spacer(1, 0.4 * inch))
                    story.append(PageBreak())
            
            # Construir PDF
            doc.build(story)
            return filepath
            
        except ImportError:
            logger.error("reportlab no está instalado. Instala con: pip install reportlab")
            raise ImportError("Se requiere reportlab para exportar a PDF. Instala con: pip install reportlab")
    
    def _export_tex(self, result: Dict[str, Any], filename: str, include_hints: bool = True, include_solutions: bool = True) -> Path:
        """Exporta a formato LaTeX"""
        filepath = self.output_directory / f"{filename}.tex"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            # Preámbulo LaTeX
            f.write("\\documentclass[12pt,a4paper]{article}\n")
            f.write("\\usepackage[utf8]{inputenc}\n")
            f.write("\\usepackage[spanish]{babel}\n")
            f.write("\\usepackage{amsmath}\n")
            f.write("\\usepackage{amsfonts}\n")
            f.write("\\usepackage{amssymb}\n")
            f.write("\\usepackage{enumitem}\n")
            f.write("\\usepackage{geometry}\n")
            f.write("\\geometry{margin=2.5cm}\n")
            f.write("\\usepackage{xcolor}\n")
            f.write("\\definecolor{darkblue}{rgb}{0.0, 0.0, 0.55}\n\n")
            
            f.write("\\title{\\textbf{Ejercicios Académicos}}\n")
            
            metadata = result.get('metadata', {})
            author_parts = []
            if metadata.get('materia'):
                author_parts.append(metadata['materia'])
            if metadata.get('unidad'):
                author_parts.append(metadata['unidad'])
            author_str = " - ".join(author_parts) if author_parts else "Sistema RAG"
            f.write(f"\\author{{{author_str}}}\n")
            f.write(f"\\date{{{datetime.now().strftime('%d de %B de %Y')}}}\n\n")
            
            f.write("\\begin{document}\n\n")
            f.write("\\maketitle\n\n")
            
            # Información del documento
            f.write("\\section*{Información}\n")
            f.write("\\begin{itemize}\n")
            if metadata.get('materia'):
                f.write(f"  \\item \\textbf{{Materia:}} {self._escape_latex(metadata['materia'])}\n")
            if metadata.get('unidad'):
                f.write(f"  \\item \\textbf{{Unidad:}} {self._escape_latex(metadata['unidad'])}\n")
            if metadata.get('tipo_ejercicio'):
                f.write(f"  \\item \\textbf{{Tipo:}} {self._escape_latex(metadata['tipo_ejercicio'])}\n")
            if metadata.get('nivel_dificultad'):
                f.write(f"  \\item \\textbf{{Dificultad:}} {self._escape_latex(metadata['nivel_dificultad'])}\n")
            f.write("\\end{itemize}\n\n")
            
            # Ejercicios
            ejercicios = result.get('ejercicios', [])
            f.write("\\section*{Ejercicios}\n\n")
            
            for i, ejercicio in enumerate(ejercicios, 1):
                f.write(f"\\subsection*{{Ejercicio {i}}}\n\n")
                
                # Pregunta
                pregunta = self._escape_latex(ejercicio['pregunta'])
                f.write(f"\\textcolor{{darkblue}}{{\\textbf{{Pregunta:}}}}\n\n")
                f.write(f"{pregunta}\n\n")
                
                # Opciones (si es multiple choice)
                if 'opciones' in ejercicio:
                    f.write("\\textcolor{darkblue}{\\textbf{Opciones:}}\n\n")
                    f.write("\\begin{enumerate}[label=\\Alph*)]\n")
                    for opcion in ejercicio['opciones']:
                        opcion_latex = self._escape_latex(opcion)
                        f.write(f"  \\item {opcion_latex}\n")
                    f.write("\\end{enumerate}\n\n")
                    
                    if include_solutions:
                        respuesta = ejercicio.get('respuesta_correcta', 'N/A')
                        f.write(f"\\textcolor{{darkblue}}{{\\textbf{{Respuesta correcta:}}}} {respuesta}\n\n")
                
                # Datos
                if ejercicio.get('datos'):
                    datos = self._escape_latex(ejercicio['datos'])
                    f.write(f"\\textcolor{{darkblue}}{{\\textbf{{Datos:}}}}\n\n")
                    f.write(f"{datos}\n\n")
                
                # Pista
                if include_hints and ejercicio.get('pista'):
                    pista = self._escape_latex(ejercicio['pista'])
                    f.write(f"\\textcolor{{darkblue}}{{\\textbf{{Pista:}}}}\n\n")
                    f.write(f"{pista}\n\n")
                
                # Solución
                if include_solutions and ejercicio.get('solucion'):
                    solucion = self._escape_latex(ejercicio['solucion'])
                    f.write(f"\\textcolor{{darkblue}}{{\\textbf{{Solución:}}}}\n\n")
                    f.write(f"{solucion}\n\n")
                
                # Puntos clave
                if ejercicio.get('puntos_clave'):
                    f.write("\\textcolor{darkblue}{\\textbf{Puntos clave:}}\n\n")
                    f.write("\\begin{itemize}\n")
                    for punto in ejercicio['puntos_clave']:
                        punto_latex = self._escape_latex(punto)
                        f.write(f"  \\item {punto_latex}\n")
                    f.write("\\end{itemize}\n\n")
                
                # Conceptos clave
                if ejercicio.get('conceptos_clave'):
                    f.write("\\textcolor{darkblue}{\\textbf{Conceptos clave:}}\n\n")
                    f.write("\\begin{itemize}\n")
                    for concepto in ejercicio['conceptos_clave']:
                        concepto_latex = self._escape_latex(concepto)
                        f.write(f"  \\item {concepto_latex}\n")
                    f.write("\\end{itemize}\n\n")
                
                # Separador
                if i < len(ejercicios):
                    f.write("\\vspace{1cm}\n")
                    f.write("\\hrule\n")
                    f.write("\\vspace{1cm}\n\n")
            
            # Footer
            f.write("\\vfill\n")
            f.write("\\begin{center}\n")
            f.write("\\textit{Generado con Sistema RAG - ITBA}\n")
            if metadata.get('fuentes'):
                f.write(f"\\\\ \\small Basado en {len(metadata['fuentes'])} fuentes académicas\n")
            f.write("\\end{center}\n\n")
            
            f.write("\\end{document}\n")
        
        return filepath
    
    def _export_only_hints(self, result: Dict[str, Any], format: str, filename: str) -> str:
        """Exporta solo las pistas de los ejercicios"""
        filepath = self.output_directory / f"{filename}.{format}"
        
        if format == 'txt':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("PISTAS DE LOS EJERCICIOS\n")
                f.write("=" * 80 + "\n\n")
                
                metadata = result.get('metadata', {})
                if metadata.get('materia'):
                    f.write(f"Materia: {metadata['materia']}\n")
                if metadata.get('unidad'):
                    f.write(f"Unidad: {metadata['unidad']}\n")
                f.write("\n")
                
                ejercicios = result.get('ejercicios', [])
                for i, ejercicio in enumerate(ejercicios, 1):
                    f.write(f"PISTA - EJERCICIO {i}\n")
                    f.write("-" * 80 + "\n")
                    if ejercicio.get('pista'):
                        f.write(f"{ejercicio['pista']}\n")
                    else:
                        f.write("No hay pista disponible\n")
                    f.write("\n")
        
        elif format == 'tex':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\\documentclass[12pt,a4paper]{article}\n")
                f.write("\\usepackage[utf8]{inputenc}\n")
                f.write("\\usepackage[spanish]{babel}\n")
                f.write("\\title{\\textbf{Pistas de Ejercicios}}\n")
                f.write("\\begin{document}\n\\maketitle\n\n")
                
                ejercicios = result.get('ejercicios', [])
                for i, ejercicio in enumerate(ejercicios, 1):
                    f.write(f"\\subsection*{{Pista - Ejercicio {i}}}\n")
                    if ejercicio.get('pista'):
                        pista = self._escape_latex(ejercicio['pista'])
                        f.write(f"{pista}\n\n")
                    else:
                        f.write("No hay pista disponible\n\n")
                
                f.write("\\end{document}\n")
        
        elif format == 'pdf':
            # Para PDF, crear un documento temporal simple
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.units import inch
            
            doc = SimpleDocTemplate(str(filepath), pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            story.append(Paragraph("<b>PISTAS DE EJERCICIOS</b>", styles['Title']))
            story.append(Spacer(1, 0.3 * inch))
            
            ejercicios = result.get('ejercicios', [])
            for i, ejercicio in enumerate(ejercicios, 1):
                story.append(Paragraph(f"<b>Pista - Ejercicio {i}</b>", styles['Heading2']))
                if ejercicio.get('pista'):
                    story.append(Paragraph(ejercicio['pista'], styles['BodyText']))
                else:
                    story.append(Paragraph("No hay pista disponible", styles['BodyText']))
                story.append(Spacer(1, 0.2 * inch))
            
            doc.build(story)
        
        return str(filepath)
    
    def _export_only_solutions(self, result: Dict[str, Any], format: str, filename: str) -> str:
        """Exporta solo las soluciones de los ejercicios"""
        filepath = self.output_directory / f"{filename}.{format}"
        
        if format == 'txt':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("SOLUCIONES DE LOS EJERCICIOS\n")
                f.write("=" * 80 + "\n\n")
                
                metadata = result.get('metadata', {})
                if metadata.get('materia'):
                    f.write(f"Materia: {metadata['materia']}\n")
                if metadata.get('unidad'):
                    f.write(f"Unidad: {metadata['unidad']}\n")
                f.write("\n")
                
                ejercicios = result.get('ejercicios', [])
                for i, ejercicio in enumerate(ejercicios, 1):
                    f.write(f"SOLUCIÓN - EJERCICIO {i}\n")
                    f.write("-" * 80 + "\n")
                    if ejercicio.get('solucion'):
                        f.write(f"{ejercicio['solucion']}\n")
                    else:
                        f.write("No hay solución disponible\n")
                    if ejercicio.get('respuesta_correcta'):
                        f.write(f"\nRespuesta correcta: {ejercicio['respuesta_correcta']}\n")
                    f.write("\n")
        
        elif format == 'tex':
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("\\documentclass[12pt,a4paper]{article}\n")
                f.write("\\usepackage[utf8]{inputenc}\n")
                f.write("\\usepackage[spanish]{babel}\n")
                f.write("\\usepackage{amsmath}\n")
                f.write("\\title{\\textbf{Soluciones de Ejercicios}}\n")
                f.write("\\begin{document}\n\\maketitle\n\n")
                
                ejercicios = result.get('ejercicios', [])
                for i, ejercicio in enumerate(ejercicios, 1):
                    f.write(f"\\subsection*{{Solución - Ejercicio {i}}}\n")
                    if ejercicio.get('respuesta_correcta'):
                        f.write(f"\\textbf{{Respuesta:}} {ejercicio['respuesta_correcta']}\n\n")
                    if ejercicio.get('solucion'):
                        solucion = self._escape_latex(ejercicio['solucion'])
                        f.write(f"{solucion}\n\n")
                    else:
                        f.write("No hay solución disponible\n\n")
                
                f.write("\\end{document}\n")
        
        elif format == 'pdf':
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.units import inch
            
            doc = SimpleDocTemplate(str(filepath), pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            story.append(Paragraph("<b>SOLUCIONES DE EJERCICIOS</b>", styles['Title']))
            story.append(Spacer(1, 0.3 * inch))
            
            ejercicios = result.get('ejercicios', [])
            for i, ejercicio in enumerate(ejercicios, 1):
                story.append(Paragraph(f"<b>Solución - Ejercicio {i}</b>", styles['Heading2']))
                if ejercicio.get('respuesta_correcta'):
                    story.append(Paragraph(f"<b>Respuesta:</b> {ejercicio['respuesta_correcta']}", styles['Normal']))
                if ejercicio.get('solucion'):
                    story.append(Paragraph(ejercicio['solucion'], styles['BodyText']))
                else:
                    story.append(Paragraph("No hay solución disponible", styles['BodyText']))
                story.append(Spacer(1, 0.2 * inch))
            
            doc.build(story)
        
        return str(filepath)
    
    def _escape_latex(self, text: str) -> str:
        """Escapa caracteres especiales de LaTeX"""
        replacements = {
            '\\': '\\textbackslash{}',
            '{': '\\{',
            '}': '\\}',
            '$': '\\$',
            '&': '\\&',
            '%': '\\%',
            '#': '\\#',
            '_': '\\_',
            '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}'
        }
        
        result = text
        for char, replacement in replacements.items():
            result = result.replace(char, replacement)
        
        return result


def export_exercises(
    result: Dict[str, Any],
    format: str = "txt",
    filename: Optional[str] = None,
    output_directory: str = "./output"
) -> str:
    """
    Función de conveniencia para exportar ejercicios
    
    Args:
        result: Resultado de generate_exercises()
        format: Formato ('txt', 'pdf', 'tex')
        filename: Nombre del archivo (opcional)
        output_directory: Directorio de salida
        
    Returns:
        Ruta del archivo generado
    """
    exporter = ExerciseExporter(output_directory=output_directory)
    return exporter.export(result, format=format, filename=filename)


if __name__ == "__main__":
    # Ejemplo de uso
    logging.basicConfig(level=logging.INFO)
    
    # Datos de ejemplo
    example_result = {
        "ejercicios": [
            {
                "pregunta": "¿Qué es una variable aleatoria?",
                "opciones": [
                    "Una variable que cambia aleatoriamente",
                    "Una función que asigna valores numéricos a eventos",
                    "Un número aleatorio",
                    "Una distribución de probabilidad"
                ],
                "respuesta_correcta": "B",
                "pista": "Piensa en términos de funciones y espacios muestrales",
                "solucion": "Una variable aleatoria es una función que asigna valores numéricos a cada evento del espacio muestral."
            }
        ],
        "metadata": {
            "materia": "Probabilidad y estadística",
            "unidad": "Variables Aleatorias",
            "tipo_ejercicio": "multiple_choice",
            "nivel_dificultad": "intermedio",
            "modelo_usado": "gpt-4o-mini",
            "fuentes": ["apunte1.pdf", "guia2.pdf"]
        }
    }
    
    # Probar exportación
    print("Exportando a TXT...")
    txt_file = export_exercises(example_result, format="txt", filename="ejemplo")
    print(f"✅ Creado: {txt_file}")
    
    print("\nExportando a TEX...")
    tex_file = export_exercises(example_result, format="tex", filename="ejemplo")
    print(f"✅ Creado: {tex_file}")
    
    try:
        print("\nExportando a PDF...")
        pdf_file = export_exercises(example_result, format="pdf", filename="ejemplo")
        print(f"✅ Creado: {pdf_file}")
    except ImportError as e:
        print(f"⚠️  {e}")

