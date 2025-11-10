# bmw-data-processor

## Project Overview
The BMW Data Processor is a Python application designed for loading, cleaning, transforming, and analyzing BMW sales data. It provides functionalities for exploratory data analysis (EDA) and prepares the data for modeling.

## Directory Structure
```
bmw-data-processor
├── src
│   ├── data
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   └── transformer.py
│   ├── utils
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   └── validator.py
│   ├── analysis
│   │   ├── __init__.py
│   │   └── eda.py
│   ├── preprocessing
│   │   ├── __init__.py
│   │   └── prepare.py
│   └── main.py
├── tests
│   ├── __init__.py
│   ├── test_loader.py
│   ├── test_transformer.py
│   ├── test_cleaner.py
│   ├── test_validator.py
│   └── test_eda.py
├── requirements.txt
└── README.md
```

## Installation
To install the required dependencies, run:
```
pip install -r requirements.txt
```

## Usage
1. Place your CSV data file in the appropriate directory.
2. Run the main application using:
```
python src/main.py
```

## Functions
- **load_csv**: Loads a CSV file with error handling for encoding and bad lines.
- **save_eda_outputs**: Saves exploratory data analysis outputs, including plots and summaries, to a specified directory.
- **limpiar_y_verificar_data**: Cleans a DataFrame by stripping whitespace from headers and values, and checks for expected columns.
- **transformar_datos**: Transforms a DataFrame by removing duplicates, null values, and unwanted columns.
- **preparar_datos**: Prepares data for modeling, including feature creation and splitting into training and testing sets.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.