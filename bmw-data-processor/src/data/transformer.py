# filepath: /bmw-data-processor/bmw-data-processor/src/data/transformer.py
def save_eda_outputs(df, out_dir):
    """
    Guarda plots y resúmenes (CSV + JSON) de EDA en out_dir.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {}
    try:
        # resumen numérico y categórico
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        df[num_cols].describe().to_csv(out_dir / "numeric_summary.csv")
        df[cat_cols].describe().to_csv(out_dir / "categorical_summary.csv")
        meta["shape"] = df.shape
        meta["num_cols"] = num_cols
        meta["cat_cols"] = cat_cols

        # histogramas solo si existe la columna target
        if "Price_USD" in df.columns:
            plt.figure(figsize=(6,4))
            sns.histplot(df["Price_USD"].dropna(), bins=40, kde=True, color="#59a14f")
            plt.title("Distribución de Price_USD")
            plt.tight_layout()
            plt.savefig(out_dir / "target_distribution.png")
            plt.close()

            plt.figure(figsize=(8,6))
            sns.boxplot(x=df["Price_USD"].dropna())
            plt.title("Boxplot Price_USD")
            plt.tight_layout()
            plt.savefig(out_dir / "target_boxplot.png")
            plt.close()
        else:
            logging.warning("No existe columna 'Price_USD' para generar plots.")

        # correlación si hay suficientes numéricas
        if len(num_cols) > 1:
            corr = df[num_cols].corr()
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
            plt.title("Matriz de correlación")
            plt.tight_layout()
            plt.savefig(out_dir / "corr_heatmap.png")
            plt.close()

        # Guardar metadatos
        with open(out_dir / "eda_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

        logging.info("EDA guardado en %s", out_dir)
    except Exception as e:
        logging.exception("Error al guardar EDA: %s", e)