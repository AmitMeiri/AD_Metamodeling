import PyPDF2

reader = PyPDF2.PdfReader('pySuStaIn/tau_trajectories_paper.pdf')

with open('extracted_methods.txt', 'w', encoding='utf-8') as out:
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if 'Methods' in text or 'Subtype and Stage Inference' in text or 'cross-validation' in text or 'CVIC' in text:
            out.write(f"\n\n--- PAGE {idx+1} ---\n\n")
            out.write(text)

print("Extracted pages.")
