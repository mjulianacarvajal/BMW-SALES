import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "outputs" / "plots"


def save_eda_outputs(
    df: pd.DataFrame, out_dir: str | Path = DEFAULT_OUTPUT_DIR
) -> None:
    """Save descriptive summaries and exploratory plots for the dataset."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    df[num_cols].describe().to_csv(out_dir / "numeric_summary.csv")
    df[cat_cols].describe().to_csv(out_dir / "categorical_summary.csv")

    metadata = {
        "shape": list(df.shape),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    if "Sales_Classification" in df.columns:
        counts = df["Sales_Classification"].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set(title="Distribution of Sales_Classification", xlabel="Class", ylabel="Count")
        fig.tight_layout()
        fig.savefig(out_dir / "target_distribution_count.png")
        plt.close(fig)
        counts.rename_axis("Sales_Classification").to_csv(
            out_dir / "sales_classification_counts.csv"
        )

    if "Price_USD" in df.columns:
        prices = df["Price_USD"].dropna()
        if not prices.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(prices, bins=40, color="#4e79a7", edgecolor="white", ax=ax)
            ax.set(title="Distribución de Price_USD", xlabel="Precio en USD", ylabel="Frecuencia")
            fig.tight_layout()
            fig.savefig(out_dir / "target_distribution.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(prices, bins=30, kde=True, color="#59a14f", ax=ax)
            ax.axvline(prices.mean(), color="red", linestyle="dashed", label="Media")
            ax.axvline(prices.median(), color="blue", linestyle="dashed", label="Mediana")
            ax.set_title("Distribución de Precios con Media y Mediana (en USD)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "target_distribution_with_stats.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=prices, ax=ax)
            ax.set(title="Boxplot de Price_USD", xlabel="Price in USD")
            fig.tight_layout()
            fig.savefig(out_dir / "target_boxplot.png")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 7))
            ax.hist(prices, bins=5, color="#4e79a7", edgecolor="white")
            ax.set(title="Distribución del Precio (USD)", xlabel="Precio en USD", ylabel="Frecuencia")
            fig.tight_layout()
            fig.savefig(out_dir / "target_histogram.png")
            plt.close(fig)

            metadata["price_usd"] = {
                "mean": float(prices.mean()),
                "median": float(prices.median()),
                "min": float(prices.min()),
                "max": float(prices.max()),
            }
        else:
            logger.warning("Price_USD no contiene valores para generar gráficos.")

    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[num_cols].corr(), annot=False, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matriz de correlación (numéricas)")
        fig.tight_layout()
        fig.savefig(out_dir / "corr_heatmap.png")
        plt.close(fig)

    with (out_dir / "eda_meta.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2, ensure_ascii=False)

    logger.info("EDA guardado en %s", out_dir)
