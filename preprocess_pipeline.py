# ===============================================
# preprocess_pipeline.py
# Criação da pipeline de transformação de dados
# ===============================================
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# Caminhos
DATA_PROCESSED = Path("data/processed/cardio_clean.parquet")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

def build_preprocessor(df: pd.DataFrame):
    """
    Cria uma pipeline de transformação de dados para o dataset Cardio.
    Inclui:
        - OneHotEncoder para variáveis categóricas
        - StandardScaler para variáveis numéricas
    """
    # Separar colunas por tipo
    cat_cols = df.select_dtypes("category").columns.tolist()
    num_cols = df.select_dtypes("number").drop(columns=["cardio"]).columns.tolist()

    # Transformers individuais
    cat_transformer = OneHotEncoder(handle_unknown="ignore")
    num_transformer = StandardScaler()

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ]
    )

    # Pipeline (podes adicionar steps extra no futuro, ex: imputação)
    pipe = Pipeline(steps=[("preprocessor", preprocessor)])
    return pipe

def save_preprocessor():
    """Treina e guarda o preprocessor baseado no dataset processado."""
    df = pd.read_parquet(DATA_PROCESSED)

    pipe = build_preprocessor(df)
    X = df.drop(columns=["cardio"])
    y = df["cardio"]

    # Ajusta a pipeline (fit) para aprender dimensões e categorias
    pipe.fit(X, y)

    out_path = MODELS_DIR / "preprocessor.pkl"
    joblib.dump(pipe, out_path)
    print(f"✅ Pipeline guardada em {out_path}")

if __name__ == "__main__":
    save_preprocessor()
