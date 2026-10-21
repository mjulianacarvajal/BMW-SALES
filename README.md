# BMW Data Processor

Aplicación para cargar, limpiar, analizar y preparar datos de ventas BMW. El
análisis y el preprocesamiento están separados en módulos para que puedan
ejecutarse y probarse de forma independiente.

## Estructura

```text
src/
├── analysis/       # Estadísticas descriptivas y gráficos de EDA
├── data/           # Carga de CSV y compatibilidad con el módulo anterior, exporta salidas gráficas y procesos númericos
├── preprocessing/  # Creación de features y división entrenamiento/prueba
├── utils/          # Limpieza y validación de datos
└── main.py         # Orquestación del flujo
tests/              # Pruebas automatizadas
```

## Instalación

Desde esta carpeta:

```bash
pip install -r requirements.txt
```

## Ejecución

El CSV predeterminado es `../data/data.csv`; los reportes se guardan en
`../data/outputs/plots`.

```bash
python -m src.main
```

Se puede indicar otra ubicación para los datos o los resultados:

```bash
python -m src.main --data /ruta/ventas.csv --output-dir /ruta/resultados
```

El script heredado `../bmwData.py` sigue disponible como acceso rápido y ejecuta
el mismo flujo modular.

## Salidas de EDA

Se generan resúmenes numéricos y categóricos, metadatos JSON, matriz de
correlación cuando hay varias columnas numéricas, gráficos de precio y un
gráfico/conteo de `Sales_Classification` cuando existe esa columna.

## Pruebas

Desde esta carpeta:

```bash
python -m pytest
```
