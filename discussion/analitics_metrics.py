import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font
import os

def compare_and_format_metrics(original_df, view_df, sheet_name, writer):
    # Filtrar classes com Support > 0 na base consolidada
    view_df = view_df[view_df['Support'] > 0]

    # Verificar se há interseção de classes entre os dois DataFrames
    common_classes = original_df['Classes'].isin(view_df['Classes'])
    original_filtered = original_df[common_classes]
    view_filtered = view_df[view_df['Classes'].isin(original_df['Classes'])]

    if original_filtered.empty or view_filtered.empty:
        print(f"Nenhuma classe em comum encontrada para a aba {sheet_name}.")
        return

    # Criar um DataFrame para armazenar os resultados
    comparison_df = pd.DataFrame()
    comparison_df["Classes"] = original_filtered["Classes"]

    # Iterar sobre as colunas relevantes (métricas)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for metric in metrics:
        original_metric = f"{metric}_orig"
        view_metric = f"{metric}_comp"

        # Substituir vírgulas por pontos e converter para float
        original_filtered[metric] = original_filtered[metric].astype(str).str.replace(",", ".").astype(float)
        view_filtered[metric] = view_filtered[metric].astype(str).str.replace(",", ".").astype(float)

        # Adicionar colunas ao DataFrame de comparação
        comparison_df[original_metric] = original_filtered[metric]
        comparison_df[view_metric] = view_filtered[metric]

        # Identificar casos iguais, superiores ou inferiores
        comparison_df[f"{metric}_Comparison"] = "Equal"
        comparison_df.loc[
            original_filtered[metric] < view_filtered[metric], f"{metric}_Comparison"
        ] = "Better"
        comparison_df.loc[
            original_filtered[metric] > view_filtered[metric], f"{metric}_Comparison"
        ] = "Worse"

    # Salvar o DataFrame de comparação na aba correspondente
    if not comparison_df.empty:
        comparison_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Acessar a aba para aplicar formatação
        worksheet = writer.sheets[sheet_name]

        # Iterar sobre as linhas e aplicar formatação
        for row in range(2, len(comparison_df) + 2):  # Linhas começam no índice 2 (cabeçalho na linha 1)
            for col, metric in enumerate(metrics, start=2):  # Colunas começam no índice 2 (após "Classes")
                cell_orig = worksheet.cell(row=row, column=col * 2)      # Coluna da métrica original
                cell_view = worksheet.cell(row=row, column=col * 2 + 1)  # Coluna da métrica da vista

                comparison = comparison_df.iloc[row - 2][f"{metric}_Comparison"]

                if comparison == "Equal":
                    cell_orig.font = Font(italic=True)
                    cell_view.font = Font(italic=True)
                elif comparison == "Better":
                    cell_view.font = Font(bold=True)
                elif comparison == "Worse":
                    cell_view.font = Font(color="FF0000")  # Vermelho
    else:
        print(f"Nenhum dado disponível para comparação na aba {sheet_name}.")

def initialize_excel(output_dir):
    try:
        # Criar o diretório de saída, se necessário
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Criar um arquivo Excel vazio com uma aba padrão
        excel_path = os.path.join(output_dir, "Comparison_Metrics.xlsx")
        workbook = Workbook()
        workbook.save(excel_path)
        print(f"Arquivo Excel criado com sucesso: {excel_path}")
        return excel_path
    except Exception as e:
        print(f"Erro ao criar o arquivo Excel: {e}")
        return None

def main():
    # Caminhos dos arquivos CSV
    original_path = "results/phase1/AT_densenet+cbam_exp/0_DenseNet201_reports_consolidated/class_report_test_0_DenseNet201.csv"
    equatorial_path = "results/phase3/CPD1_TEST_VIEW/0_DenseNet201_EQUATORIAL_consolidated/class_report_test_0_DenseNet201.csv"
    polar_path = "results/phase3/CPD1_TEST_VIEW/0_DenseNet201_POLAR_consolidated/class_report_test_0_DenseNet201.csv"

    try:
        # Carregar os arquivos CSV com delimitador ;
        original_df = pd.read_csv(original_path, sep=";")
        equatorial_df = pd.read_csv(equatorial_path, sep=";")
        polar_df = pd.read_csv(polar_path, sep=";")

        # Diretório de saída
        output_dir = "discussion/"

        # Criar a planilha inicial
        excel_path = initialize_excel(output_dir)
        if not excel_path:
            return

        # Preencher a planilha com as comparações
        with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            # Comparação Original x Equatorial
            compare_and_format_metrics(original_df, equatorial_df, "Original_vs_Equatorial", writer)

            # Comparação Original x Polar
            compare_and_format_metrics(original_df, polar_df, "Original_vs_Polar", writer)

            # Se nenhuma aba foi criada, criar uma aba padrão
            if not writer.sheets:
                workbook = writer.book
                workbook.create_sheet("Empty")
                print("Nenhuma aba válida foi criada. Uma aba padrão foi adicionada.")
    except Exception as e:
        print(f"Erro ao salvar o arquivo Excel: {e}")

if __name__ == "__main__":
    main()
