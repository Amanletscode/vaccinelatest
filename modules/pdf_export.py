"""
pdf_export.py — Generate downloadable PDF summaries of intelligence briefs.

Uses ReportLab for rich formatting (bold, headings, bullet points) with
automatic fallback to fpdf2 if ReportLab is not installed.
"""

import re
from io import BytesIO
from datetime import datetime


def generate_pdf_summary(summary_text: str, title: str = "Vaccine Pipeline Summary"):
    """Generate a professionally formatted PDF from a markdown-like summary string."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
        from reportlab.lib.colors import HexColor

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=letter,
            topMargin=0.75 * inch, bottomMargin=0.75 * inch,
            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "CustomTitle", parent=styles["Heading1"],
            fontSize=22, textColor=HexColor("#1f4788"),
            spaceAfter=20, spaceBefore=0, alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        )
        heading_style = ParagraphStyle(
            "CustomHeading", parent=styles["Heading2"],
            fontSize=16, textColor=HexColor("#2c5aa0"),
            spaceAfter=10, spaceBefore=20,
            fontName="Helvetica-Bold", leftIndent=0,
        )
        subheading_style = ParagraphStyle(
            "SubHeading", parent=styles["Heading3"],
            fontSize=13, textColor=HexColor("#3d6bb3"),
            spaceAfter=8, spaceBefore=12,
            fontName="Helvetica-Bold", leftIndent=0,
        )
        normal_style = ParagraphStyle(
            "NormalText", parent=styles["Normal"],
            fontSize=11, leading=14,
            alignment=TA_JUSTIFY, spaceAfter=6,
        )
        bullet_style = ParagraphStyle(
            "BulletText", parent=normal_style,
            leftIndent=20, bulletIndent=10, spaceAfter=4,
        )

        def _md_bold_to_html(text):
            if not text:
                return text
            text = re.sub(r"\*\*([^*]+?)\*\*", r"<b>\1</b>", text)
            text = re.sub(r"__([^_]+?)__", r"<b>\1</b>", text)
            text = text.replace("**", "").replace("__", "")
            if text.count("<b>") != text.count("</b>"):
                text = re.sub(r"<b>|</b>", "", text)
            return text

        story = []
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            Paragraph(
                f"<i>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>",
                ParagraphStyle("DateStyle", parent=normal_style, alignment=TA_CENTER, fontSize=9),
            )
        )
        story.append(Spacer(1, 0.3 * inch))

        lines = summary_text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.05 * inch))
            elif line.startswith("##"):
                heading_text = re.sub(r"\*\*|__", "", line.replace("##", "").strip())
                story.append(Paragraph(heading_text, heading_style))
            elif line.startswith("#"):
                heading_text = re.sub(r"\*\*|__", "", line.replace("#", "").strip())
                story.append(Paragraph(heading_text, subheading_style))
            elif line.startswith("-") or line.startswith("•") or line.startswith("*"):
                bullet_text = line.lstrip("-•*").strip()
                bullet_text = _md_bold_to_html(bullet_text)
                bullet_text = bullet_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                bullet_text = re.sub(r"&lt;b&gt;([^&]+?)&lt;/b&gt;", r"<b>\1</b>", bullet_text)
                story.append(Paragraph(f"• {bullet_text}", bullet_style))
            else:
                text = _md_bold_to_html(line)
                text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                text = re.sub(r"&lt;b&gt;([^&]+?)&lt;/b&gt;", r"<b>\1</b>", text)
                story.append(Paragraph(text, normal_style))

        doc.build(story)
        buffer.seek(0)
        return buffer

    except ImportError:
        # Fallback: fpdf2
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.set_fill_color(31, 71, 136)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(0, 12, title, ln=1, align="C", fill=True)
            pdf.set_text_color(0, 0, 0)
            pdf.ln(10)
            pdf.set_font("Arial", size=10)

            for line in summary_text.split("\n"):
                line = line.strip()
                if line:
                    if line.startswith("##"):
                        pdf.ln(5)
                        pdf.set_font("Arial", "B", 14)
                        clean = line.replace("##", "").replace("**", "").replace("__", "").strip()
                        pdf.cell(0, 8, clean, ln=1)
                        pdf.set_font("Arial", size=10)
                    elif line.startswith("#"):
                        pdf.ln(3)
                        pdf.set_font("Arial", "B", 12)
                        clean = line.replace("#", "").replace("**", "").replace("__", "").strip()
                        pdf.cell(0, 7, clean, ln=1)
                        pdf.set_font("Arial", size=10)
                    else:
                        clean = line.replace("**", "").replace("__", "")
                        pdf.multi_cell(0, 5, clean)

            buffer = BytesIO()
            buffer.write(pdf.output(dest="S").encode("latin-1"))
            buffer.seek(0)
            return buffer
        except ImportError:
            return None
