"""
utils/pdf_report.py
-------------------
Generates a one-page PDF "maintenance report" for a machine.

The report bundles everything a maintenance planner needs:
  * machine identity and current status,
  * the latest sensor readings,
  * the ML prediction (failure probability, health, remaining useful life),
  * recent alerts,
  * prioritised maintenance recommendations.

We use ReportLab's "Platypus" high-level API, which lays out a document from a
list of "flowables" (paragraphs, tables, spacers). The function returns the PDF
as raw bytes so the FastAPI backend can stream it straight to the browser.
"""

from datetime import datetime, timezone
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

# Status -> colour, reused across the document for visual consistency.
_STATUS_COLOR = {
    "Healthy":  colors.HexColor("#2e7d32"),
    "Warning":  colors.HexColor("#f59e0b"),
    "Critical": colors.HexColor("#c62828"),
}
_SEVERITY_COLOR = {
    "Critical": colors.HexColor("#c62828"),
    "Warning":  colors.HexColor("#f59e0b"),
    "Info":     colors.HexColor("#0277bd"),
}


def build_maintenance_report(
    machine: dict,
    reading: dict,
    prediction: dict,
    alerts: list[dict],
    recommendations: list[dict],
) -> bytes:
    """
    Build the PDF and return it as bytes.

    All inputs are plain dicts (not ORM objects) so this function has no
    database dependency and is trivial to unit-test.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        leftMargin=1.8 * cm, rightMargin=1.8 * cm,
        title=f"Maintenance Report — {machine.get('machine_id', '')}",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=styles["Normal"], fontSize=9,
                         textColor=colors.grey, spaceAfter=14)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                        textColor=colors.HexColor("#1e88e5"), spaceBefore=12, spaceAfter=6)
    body = styles["Normal"]

    story = []  # the list of flowables that make up the document

    # ── Title block ───────────────────────────────────────────────────────────
    story.append(Paragraph("CNC Maintenance Report", h1))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(f"Generated {generated}", sub))

    # ── Machine summary table ─────────────────────────────────────────────────
    story.append(Paragraph("Machine", h2))
    status = prediction.get("status", "Unknown")
    machine_rows = [
        ["Machine ID", machine.get("machine_id", "—"), "Name", machine.get("name", "—")],
        ["Type", machine.get("type", "—"), "Location", machine.get("location", "—")],
        ["Current status", status, "Health score",
         f"{prediction.get('health_score', '—')} / 100"],
    ]
    t = Table(machine_rows, colWidths=[3.2 * cm, 5.0 * cm, 3.2 * cm, 5.0 * cm])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0f4f8")),
        ("TEXTCOLOR", (1, 2), (1, 2), _STATUS_COLOR.get(status, colors.black)),
        ("FONTNAME", (1, 2), (1, 2), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t)

    # ── Prediction summary ────────────────────────────────────────────────────
    story.append(Paragraph("Prediction", h2))
    prob = prediction.get("failure_probability", prediction.get("probability", 0))
    pred_rows = [
        ["Failure predicted", "Yes" if prediction.get("failure_prediction") else "No",
         "Failure probability", f"{prob:.1%}"],
        ["Remaining useful life", f"{prediction.get('rul_minutes', '—')} min",
         "RUL band", prediction.get("rul_band", "—")],
    ]
    t2 = Table(pred_rows, colWidths=[3.6 * cm, 4.6 * cm, 3.6 * cm, 4.6 * cm])
    t2.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0f4f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t2)

    # ── Latest sensor readings ────────────────────────────────────────────────
    story.append(Paragraph("Latest sensor readings", h2))
    sensor_rows = [
        ["Air temp [K]", f"{reading.get('air_temp', 0):.2f}",
         "Process temp [K]", f"{reading.get('process_temp', 0):.2f}"],
        ["Rotational speed [rpm]", f"{reading.get('rotational_speed', 0):.0f}",
         "Torque [Nm]", f"{reading.get('torque', 0):.2f}"],
        ["Tool wear [min]", f"{reading.get('tool_wear', 0):.0f}",
         "Machine type", reading.get("type", "—")],
    ]
    t3 = Table(sensor_rows, colWidths=[4.2 * cm, 4.0 * cm, 4.2 * cm, 4.0 * cm])
    t3.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#f0f4f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t3)

    # ── Recommendations ───────────────────────────────────────────────────────
    story.append(Paragraph("Maintenance recommendations", h2))
    if recommendations:
        for rec in recommendations:
            sev = rec.get("severity", "Info")
            colour = _SEVERITY_COLOR.get(sev, colors.black)
            # reportlab's <font color> wants a "#rrggbb" string; hexval() returns
            # "0xrrggbb", so we swap the "0x" prefix for "#".
            hex_str = "#" + colour.hexval()[2:]
            line = ParagraphStyle("rec", parent=body, fontSize=9.5,
                                  leftIndent=6, spaceAfter=4)
            story.append(Paragraph(
                f'<font color="{hex_str}"><b>[{sev}] {rec.get("title","")}</b></font> '
                f'— {rec.get("recommendation","")}', line))
    else:
        story.append(Paragraph("No recommendations — machine operating normally.", body))

    # ── Recent alerts ─────────────────────────────────────────────────────────
    story.append(Paragraph("Recent alerts", h2))
    if alerts:
        alert_rows = [["Severity", "Title", "Message"]]
        for a in alerts[:8]:
            alert_rows.append([a.get("severity", ""), a.get("title", ""),
                               a.get("message", "")])
        ta = Table(alert_rows, colWidths=[2.2 * cm, 4.0 * cm, 10.2 * cm])
        ta.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e88e5")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7de")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(ta)
    else:
        story.append(Paragraph("No recent alerts.", body))

    story.append(Spacer(1, 0.6 * cm))
    story.append(Paragraph(
        "This report was generated automatically by the CNC Predictive "
        "Maintenance Platform. RUL is a heuristic estimate, not a guarantee.",
        sub))

    # Render the flowables into the buffer, then return the raw bytes.
    doc.build(story)
    buffer.seek(0)
    return buffer.read()
