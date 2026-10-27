import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, KeepTogether, ListFlowable, ListItem
from PIL import Image as PI
J=json.load(open("steps.json")); S=J["steps"]; TAG=J["data_tag"]; NAVY=colors.HexColor("#1B2A49"); TEAL=colors.HexColor("#139A8C")
h1=ParagraphStyle("h1",fontName="Helvetica-Bold",fontSize=22,textColor=NAVY,spaceAfter=4,leading=26)
h2=ParagraphStyle("h2",fontName="Helvetica",fontSize=10.5,textColor=TEAL,spaceAfter=10)
b=ParagraphStyle("b",fontName="Helvetica",fontSize=10.5,leading=14.5)
cap=ParagraphStyle("c",fontName="Helvetica-Oblique",fontSize=9,textColor=colors.grey,spaceBefore=3,spaceAfter=8)
tk=ParagraphStyle("t",fontName="Helvetica-Bold",fontSize=10.5,textColor=NAVY,backColor=colors.HexColor("#E4F4F1"),borderPadding=7,leading=14)
def foot(c,d):
    c.setFont("Helvetica",8); c.setFillColor(colors.grey); c.drawString(2*cm,1.1*cm,"QPO - code walkthrough  |  numbers: "+TAG); c.drawRightString(A4[0]-2*cm,1.1*cm,f"Page {d.page}")
doc=SimpleDocTemplate("QPO_Code_Walkthrough.pdf",pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=1.8*cm,bottomMargin=1.8*cm,title="Quantum Portfolio Optimizer - Code Walkthrough")
E=[Spacer(1,5*cm),Paragraph("Fault-Tolerant Quantum Portfolio Optimizer",ParagraphStyle("T",fontName="Helvetica-Bold",fontSize=30,leading=36,textColor=NAVY)),Spacer(1,.4*cm),
Paragraph("Step-by-step code walkthrough with figures generated from the project's own code",ParagraphStyle("s",fontName="Helvetica",fontSize=14,textColor=TEAL,leading=18)),Spacer(1,1*cm),
Paragraph("Figures and numbers in this guide are produced by running the code in the zip (illustrative 12-qubit instance from a synthetic Oct 2008). Data used for all performance numbers: "+TAG+". "+("" if TAG=="REAL DATA" else "They illustrate how the pipeline behaves; they do not describe real markets.")+"",b),Spacer(1,.8*cm)]
E.append(Paragraph("<b>Contents</b>",h1))
for i,s in enumerate(S): E.append(Paragraph(f"{i+1}. {s['title']}",b))
E.append(PageBreak())
for i,s in enumerate(S):
    E.append(Paragraph(s["title"],h1)); E.append(Paragraph("Where: "+s["sub"],h2))
    E.append(ListFlowable([ListItem(Paragraph(x,b),leftIndent=14) for x in s["bullets"]],bulletType="bullet",start="•",leftIndent=14))
    E.append(Spacer(1,.3*cm))
    if s["img"]:
        w,h=PI.open(s["img"]).size; W=16.5*cm; H=W*h/w
        if H>11*cm: H=11*cm; W=H*w/h
        E.append(Image(s["img"],W,H)); E.append(Paragraph(s["cap"],cap))
    else: E.append(Paragraph(s["cap"],cap))
    E.append(Paragraph("Takeaway: "+s["take"],tk)); E.append(PageBreak())
doc.build(E[:-1],onFirstPage=foot,onLaterPages=foot)
