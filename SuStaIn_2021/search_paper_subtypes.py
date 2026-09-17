import PyPDF2

reader = PyPDF2.PdfReader('pySuStaIn/tau_trajectories_paper.pdf')
print('Num pages:', len(reader.pages))

terms = ['CVIC', 'cross-validation', 'cross validation', 'information criterion', 'four subtypes', '4 subtypes', 'model selection', 'number of subtypes', 'BIC', 'AIC', 'out-of-sample', 'MCMC', 'subtype', 'optimum', 'optimal']

for idx, page in enumerate(reader.pages):
    text = page.extract_text()
    for term in terms:
        if term.lower() in text.lower():
            print(f'=== Term "{term}" on Page {idx+1} ===')
            lines = text.split('\n')
            for line in lines:
                if term.lower() in line.lower():
                    print('  ', line.strip()[:140])
