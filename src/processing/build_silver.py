from pathlib import Path

import pandas as pd

# Definindo os caminhos principais do projeto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

BRONZE_ROOT = PROJECT_ROOT / "data" / "bronze"
SILVER_ROOT = PROJECT_ROOT / "data" / "silver"


# Definindo as tabelas utilizadas na construção da Silver
BRONZE_TABLES = {
    "sales": "sales.parquet",
    "stores": "stores.parquet",
    "transactions": "transactions.parquet",
    "holidays": "holidays.parquet",
}


# Carregando uma tabela da camada Bronze
def load_bronze_table(table_name: str) -> pd.DataFrame:
    file_path = BRONZE_ROOT / BRONZE_TABLES[table_name]

    if not file_path.exists():
        raise FileNotFoundError(f"Tabela Bronze não encontrada: {file_path}")

    return pd.read_parquet(file_path)


# Validando unicidade de uma chave
def validate_unique_key(
    df: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    duplicated_rows = df.duplicated(subset=columns).sum()

    if duplicated_rows > 0:
        raise ValueError(
            f"Tabela '{table_name}' possui {duplicated_rows} "
            f"duplicidades na chave {columns}."
        )


# Preparando os feriados aplicáveis a cada loja
def build_store_holidays(
    holidays: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    # Selecionando apenas eventos que representam feriados efetivos
    holiday_types = ["Holiday", "Transfer", "Additional", "Bridge"]

    active_holidays = holidays.loc[
        (~holidays["transferred"]) & (holidays["type"].isin(holiday_types))
    ].copy()

    national = active_holidays.loc[active_holidays["locale"] == "National"].merge(
        stores[["store_nbr"]],
        how="cross",
    )

    regional = active_holidays.loc[active_holidays["locale"] == "Regional"].merge(
        stores[["store_nbr", "state"]],
        left_on="locale_name",
        right_on="state",
        how="inner",
    )

    local = active_holidays.loc[active_holidays["locale"] == "Local"].merge(
        stores[["store_nbr", "city"]],
        left_on="locale_name",
        right_on="city",
        how="inner",
    )

    store_holidays = pd.concat(
        [national, regional, local],
        ignore_index=True,
    )

    store_holidays = store_holidays.groupby(
        ["date", "store_nbr"],
        as_index=False,
    ).agg(
        holiday_count=("description", "size"),
        holiday_description=(
            "description",
            lambda values: " | ".join(sorted(set(values))),
        ),
    )

    store_holidays["is_holiday"] = 1

    return store_holidays


# Validando a consistência da tabela consolidada da camada Silver
def validate_silver_output(
    df: pd.DataFrame,
    expected_rows: int,
) -> None:
    if len(df) != expected_rows:
        raise ValueError(
            f"Quantidade de registros inesperada: "
            f"esperado {expected_rows:,}, obtido {len(df):,}."
        )

    validate_unique_key(
        df,
        ["date", "store_nbr", "family"],
        "sales_enriched",
    )

    required_non_null = [
        "city",
        "state",
        "type",
        "cluster",
        "transactions_missing",
        "holiday_count",
        "is_holiday",
    ]

    null_counts = df[required_non_null].isna().sum()

    if (null_counts > 0).any():
        raise ValueError(
            f"Valores ausentes inesperados na Silver:\n{null_counts[null_counts > 0]}"
        )

    transactions_flag_valid = (
        df["transactions"].isna() == df["transactions_missing"].eq(1)
    ).all()

    if not transactions_flag_valid:
        raise ValueError("A flag transactions_missing está inconsistente.")

    holiday_flag_valid = (
        df["holiday_description"].notna() == df["is_holiday"].eq(1)
    ).all()

    if not holiday_flag_valid:
        raise ValueError("A flag is_holiday está inconsistente.")


# Construindo a tabela consolidada da camada Silver
def build_sales_enriched() -> pd.DataFrame:
    sales = load_bronze_table("sales")
    stores = load_bronze_table("stores")
    transactions = load_bronze_table("transactions")
    holidays = load_bronze_table("holidays")

    validate_unique_key(
        sales,
        ["date", "store_nbr", "family"],
        "sales",
    )

    validate_unique_key(
        stores,
        ["store_nbr"],
        "stores",
    )

    validate_unique_key(
        transactions,
        ["date", "store_nbr"],
        "transactions",
    )

    store_holidays = build_store_holidays(
        holidays,
        stores,
    )

    silver = sales.merge(
        stores,
        on="store_nbr",
        how="left",
        validate="many_to_one",
    )

    silver = silver.merge(
        transactions,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
    )

    silver["transactions_missing"] = silver["transactions"].isna().astype("int8")

    silver = silver.merge(
        store_holidays,
        on=["date", "store_nbr"],
        how="left",
        validate="many_to_one",
    )

    silver["is_holiday"] = silver["is_holiday"].fillna(0).astype("int8")
    silver["holiday_count"] = silver["holiday_count"].fillna(0).astype("int16")

    validate_silver_output(
        silver,
        expected_rows=len(sales),
    )

    return silver


# Salvando a tabela consolidada na camada Silver
def save_silver_table(
    df: pd.DataFrame,
    table_name: str,
) -> None:
    SILVER_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = SILVER_ROOT / f"{table_name}.parquet"

    df.to_parquet(
        output_path,
        index=False,
    )

    print(f"[{table_name}] {len(df):,} registros | {len(df.columns)} colunas")

    print(f"[{table_name}] salvo em: {output_path}")


# Construindo a camada Silver
def build_silver_layer() -> None:
    print("Iniciando construção da camada Silver...")

    sales_enriched = build_sales_enriched()

    save_silver_table(
        sales_enriched,
        "sales_enriched",
    )

    print("\nCamada Silver construída com sucesso.")


# Executando o pipeline quando o arquivo for chamado diretamente
if __name__ == "__main__":
    build_silver_layer()
