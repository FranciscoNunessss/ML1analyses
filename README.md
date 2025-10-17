# Cardiovascular Disease Prediction

**Machine Learning M1 Assignment**  
Francisco Nunes / Daniel Rodrigues — 2022147843 / 2022103368

## Como correr

### 1. Instalar dependências
```bash
pip install uv
uv sync
```

### 2. Configurar Kaggle
Criar ficheiro `.env`:
```
KAGGLE_USERNAME=teu_username
KAGGLE_KEY=tua_api_key
```

### 3. Executar pipeline
```bash
uv run python main.py download
uv run python main.py preprocess
uv run python main.py summary
```

### 4. Análise no Jupyter
```bash
uv run jupyter lab
```
Abrir `notebook/01_EDA_Cardio.ipynb`

## Estrutura
- `main.py` - Pipeline principal
- `preprocess_pipeline.py` - Preprocessamento
- `data/` - Dados raw e processados
- `models/` - Modelos salvos
- `notebook/` - Análise exploratória

## Dataset
70.000 registos de pacientes com 14 features para prever doença cardiovascular (binário, classes balanceadas 50/50).
