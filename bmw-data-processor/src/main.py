import argparse
import logging
from pathlib import Path

if __package__:
    
    from .analysis.eda import save_eda_outputs
    from .data.loader import load_csv
    from .preprocessing.prepare import preparar_datos
    from .utils.cleaner import limpiar_y_verificar_data
else:
    from analysis.eda import save_eda_outputs
    from data.loader import load_csv
    from preprocessing.prepare import preparar_datos
    from utils.cleaner import limpiar_y_verificar_data

logger = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPOSITORY_ROOT / "data" / "data.csv"
DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "data" / "outputs" / "plots"


def main(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Run loading, cleaning, EDA, and model-data preparation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_csv(data_path)
    df = limpiar_y_verificar_data(df, ["Price_USD", "Year"])

    save_eda_outputs(df, output_dir)
    X_train, X_test, _, _, num_cols, cat_cols = preparar_datos(df)
    logger.info(
        "Datos preparados: %s filas de entrenamiento, %s de prueba; "
        "%s variables numéricas y %s categóricas.",
        len(X_train),
        len(X_test),
        len(num_cols),
        len(cat_cols),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Procesa y analiza datos de ventas BMW.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="Ruta al archivo CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directorio donde guardar los resúmenes y gráficos.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    main(args.data, args.output_dir)
