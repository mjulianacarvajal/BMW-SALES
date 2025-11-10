def load_csv(path, encoding="latin-1", on_bad_lines="skip", **kwargs):
    """
    Carga un CSV con fallback de encoding y control de líneas malas.
    """
    from pathlib import Path
    import pandas as pd
    import logging

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} no existe")
    try:
        return pd.read_csv(path, encoding=encoding, on_bad_lines=on_bad_lines, **kwargs)
    except UnicodeDecodeError:
        logging.warning("UnicodeDecodeError con %s; reintentando con latin1", encoding)
        return pd.read_csv(path, encoding="latin-1", on_bad_lines=on_bad_lines, **kwargs)