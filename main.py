import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --------------------------------------------
# Constantes de paths
# --------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
NOTEBOOK_DIR = ROOT / "notebook"

DATASET_REF = "sulianova/cardiovascular-disease-dataset"  # Kaggle dataset slug
ZIP_NAME = "cardio.zip"  # nome local para o zip
CSV_NAME = "cardio_train.csv"  # nome do csv dentro do zip/dataset


# --------------------------------------------
# Kaggle helper
# --------------------------------------------
def _ensure_kaggle_creds():
    """
    Lê credenciais do .env (se existirem) ou usa ~/.kaggle/kaggle.json.
    Evita crash quando usamos o pacote kaggle programaticamente.
    """
    load_dotenv()
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    if username and key:
        kaggle_dir.mkdir(exist_ok=True)
        if not kaggle_json.exists():
            kaggle_json.write_text(
                f'{{"username":"{username}","key":"{key}"}}', encoding="utf-8"
            )
            os.chmod(kaggle_json, 0o600)
    elif not kaggle_json.exists():
        raise RuntimeError(
            "Credenciais Kaggle não encontradas. Define KAGGLE_USERNAME e KAGGLE_KEY no .env "
            "ou coloca ~/.kaggle/kaggle.json (permissões 600)."
        )


def download():
    """Faz download do dataset via API Kaggle para data/raw."""
    _ensure_kaggle_creds()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Import aqui para evitar custo quando não é preciso
    from kaggle import api  # type: ignore

    print(f"⤵️  A descarregar {DATASET_REF} …")
    api.dataset_download_files(
        DATASET_REF,
        path=str(RAW_DIR),
        quiet=False,
        force=True,
    )

    # O Kaggle guarda como .zip; vamos normalizar o nome
    # Procura o único zip na pasta raw
    zips = list(RAW_DIR.glob("*.zip"))
    if not zips:
        raise FileNotFoundError("Nenhum .zip encontrado após o download do Kaggle.")
    zips[0].replace(RAW_DIR / ZIP_NAME)
    print(f"📦 Guardado em {RAW_DIR / ZIP_NAME}")

    # Extrair
    with zipfile.ZipFile(RAW_DIR / ZIP_NAME, "r") as zf:
        zf.extractall(RAW_DIR)
    print(f"🗃️  Ficheiros extraídos para {RAW_DIR}")


def preprocess():
    """Limpeza básica e guardado em data/processed/cardio_clean.parquet"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RAW_DIR / CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} não existe. Corre primeiro: python main.py download"
        )

    df = pd.read_csv(csv_path, sep=";")

    # Renomeações simples
    df.columns = [c.strip().lower() for c in df.columns]
    # target é 'cardio' (0/1)

    # Corrigir idades (dataset tem 'age' em dias)
    if "age" in df.columns:
        df["age_years"] = (df["age"] / 365.25).round(1)
        df.drop(columns=["age"], inplace=True)

    # Discretizar 'ap_hi'/'ap_bp' outliers básicos
    # (remoção de outliers grosseiros por percentis)
    for col in ["ap_hi", "ap_lo", "height", "weight"]:
        if col in df.columns:
            q01, q99 = df[col].quantile([0.01, 0.99])
            df[col] = df[col].clip(lower=q01, upper=q99)

    # IMC
    if {"height", "weight"}.issubset(df.columns):
        h_m = df["height"] / 100.0
        df["bmi"] = (df["weight"] / (h_m**2)).round(2)

    # Categóricas para tipo category
    cat_cols = [
        c
        for c in ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
        if c in df.columns
    ]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    # Ordenação de colunas (target no fim)
    cols = [c for c in df.columns if c != "cardio"] + ["cardio"]
    df = df[cols]

    out_path = PROCESSED_DIR / "cardio_clean.parquet"
    df.to_parquet(out_path, index=False)
    print(f"✅ Dataset processado: {out_path} ({len(df):,} linhas)")


def summary():
    """Pequeno resumo para validação rápida."""
    path = PROCESSED_DIR / "cardio_clean.parquet"
    if not path.exists():
        raise FileNotFoundError("Falta processed. Corre: python main.py preprocess")
    df = pd.read_parquet(path)

    print("\nFormato:", df.shape)
    print("\nTipos:")
    print(df.dtypes)
    print("\nTarget balance:")
    print(df["cardio"].value_counts(normalize=True).round(3))
    print("\nPré-visualização:")
    print(df.head(5))


def build_parser():
    p = argparse.ArgumentParser(description="Pipeline M1 (Kaggle → raw → processed)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "download", help="Descarrega e extrai dataset do Kaggle para data/raw"
    )
    sub.add_parser("preprocess", help="Limpa e guarda em data/processed")
    sub.add_parser("summary", help="Mostra um resumo do processed")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "download":
        download()
    elif args.cmd == "preprocess":
        preprocess()
    elif args.cmd == "summary":
        summary()
