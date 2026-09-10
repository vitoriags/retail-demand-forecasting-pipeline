from pathlib import Path

import pandas as pd

# Definindo os caminhos principais do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "favorita"
BRONZE_ROOT = PROJECT_ROOT / "data" / "bronze"


# Localizando automaticamente a pasta que contém os arquivos brutos
train_files = list(RAW_ROOT.rglob("train.csv"))

if not train_files:
    raise FileNotFoundError("Arquivo train.csv não encontrado na camada Raw.")

DATA_PATH = train_files[0].parent


# Definindo os arquivos de origem utilizados na camada Bronze
SOURCE_FILES = {
    "sales": "train.csv",
    "stores": "stores.csv",
    "transactions": "transactions.csv",
    "holidays": "holidays_events.csv",
    "oil": "oil.csv",
}


# Definindo as colunas de data de cada fonte
DATE_COLUMNS = {
    "sales": ["date"],
    "transactions": ["date"],
    "holidays": ["date"],
    "oil": ["date"],
}


# Definindo as colunas obrigatórias de cada fonte
REQUIRED_COLUMNS = {
    "sales": ["id", "date", "store_nbr", "family", "sales", "onpromotion"],
    "stores": ["store_nbr", "city", "state", "type", "cluster"],
    "transactions": ["date", "store_nbr", "transactions"],
    "holidays": [
        "date",
        "type",
        "locale",
        "locale_name",
        "description",
        "transferred",
    ],
    "oil": ["date", "dcoilwtico"],
}


# Carregando um arquivo da camada Raw
def load_raw_file(filename: str) -> pd.DataFrame:
    file_path = DATA_PATH / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    return pd.read_csv(file_path)


# Validando a presença das colunas obrigatórias
def validate_required_columns(df: pd.DataFrame, table_name: str) -> None:
    required_columns = REQUIRED_COLUMNS[table_name]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Tabela '{table_name}' com colunas ausentes: {missing_columns}"
        )


# Convertendo colunas de data para o tipo datetime
def parse_date_columns(
    df: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    date_columns = DATE_COLUMNS.get(table_name, [])

    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors="raise")

    return df


# Validando se a tabela possui registros
def validate_not_empty(df: pd.DataFrame, table_name: str) -> None:
    if df.empty:
        raise ValueError(f"A tabela '{table_name}' está vazia.")


# Exibindo informações básicas do processamento
def log_table_info(df: pd.DataFrame, table_name: str) -> None:
    print(f"[{table_name}] {len(df):,} registros | {len(df.columns)} colunas")


# Salvando uma tabela na camada Bronze em formato Parquet
def save_bronze_table(df: pd.DataFrame, table_name: str) -> None:
    BRONZE_ROOT.mkdir(parents=True, exist_ok=True)

    output_path = BRONZE_ROOT / f"{table_name}.parquet"

    df.to_parquet(output_path, index=False)

    print(f"[{table_name}] salvo em: {output_path}")


# Processando uma única tabela da camada Raw para a Bronze
def process_table(table_name: str, filename: str) -> None:
    print(f"\nProcessando tabela: {table_name}")

    df = load_raw_file(filename)

    validate_not_empty(df, table_name)
    validate_required_columns(df, table_name)

    df = parse_date_columns(df, table_name)

    log_table_info(df, table_name)
    save_bronze_table(df, table_name)


# Construindo todas as tabelas da camada Bronze
def build_bronze_layer() -> None:
    print("Iniciando construção da camada Bronze...")

    for table_name, filename in SOURCE_FILES.items():
        process_table(table_name, filename)

    print("\nCamada Bronze construída com sucesso.")


# Executando o pipeline quando o arquivo for chamado diretamente
if __name__ == "__main__":
    build_bronze_layer()
