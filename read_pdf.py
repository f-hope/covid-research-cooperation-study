# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import pdfplumber

import pdfplumber
import os
import pandas as pd
import docx

if __name__ == '__main__':

    files = os.listdir('EOI')

    text_dict = {}
    for file in files:
        if '.pdf' in file:
            with pdfplumber.open(r'EOI/' + file) as pdf:
                text = [page.extract_text() for page in pdf.pages]
                text_dict[file] = text
        if '.doc' in file:
            doc = docx.Document(r'EOI/' + file)
            text = [p.text for p in doc.paragraphs]
            text_dict[file] = text

        else:
            print(file + ' not read')


    df = pd.DataFrame(text_dict)
