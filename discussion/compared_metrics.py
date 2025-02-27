import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

def compare_metrics(input_file):
    # Carregar a planilha
    df = pd.read_excel(input_file, sheet_name="comparar")
    
    # Criar DataFrames para cada base
    orig_400 = df[df['Base'] == 'Orig 400']
    eq_400 = df[df['Base'] == 'EQ 400']
    pl_400 = df[df['Base'] == 'PL 400']
    
    # Função para comparar métricas
    def compare(df1, df2):
        return df1.merge(df2, on='Classes', suffixes=('_orig', '_comp'))
    
    # Comparações
    comparison_eq = compare(orig_400, eq_400)
    comparison_pl = compare(orig_400, pl_400)
    
    # Criar um escritor de planilhas
    with pd.ExcelWriter(input_file, engine='openpyxl', mode="a") as writer:
        comparison_eq.to_excel(writer, sheet_name='Comparison_Orig_EQ', index=False)
        comparison_pl.to_excel(writer, sheet_name='Comparison_Orig_PL', index=False)
    
    # Destacar valores melhorados/neutros em negrito
    wb = load_workbook(input_file)
    for sheet_name in ['Comparison_Orig_EQ', 'Comparison_Orig_PL']:
        ws = wb[sheet_name]
        header = {cell.value: cell.column for cell in ws[1]}
        
        for row in range(2, ws.max_row + 1):
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                col_comp = header[f'{metric}_comp']
                col_orig = header[f'{metric}_orig']
                
                if ws.cell(row, col_comp).value >= ws.cell(row, col_orig).value:  # Se melhorou ou manteve
                    ws.cell(row, col_comp).font = Font(bold=True)
    
    wb.save(input_file)
    print(f'✅ Comparação salva e destacada em {input_file}')

# Exemplo de uso
input_file = "discussion/Comparar_literatura_CPD1.xlsx"  # Nome do arquivo de entrada
compare_metrics(input_file)