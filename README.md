# Report for Global Weather Repository Analysis & Forecasting
## Creator- Anupreet Singh

## PM Accelerator Mission
**By making industry-leading tools and education available to individuals from all backgrounds, we level the playing field for future PM leaders. This is the PM Accelerator motto, as we grant aspiring and experienced PMs what they need most – Access. We introduce you to industry leaders, surround you with the right PM ecosystem, and discover the new world of AI product management skills.**

## Project Overview
This project is meant to be submission for testing out my suitability as Data Scientist/ ML intern at PM Accelarator . It analyzes the Global Weather Repository dataset to forecast future weather trends and showcase a mix of basic and advanced data science techniques. The dataset contains daily weather information for cities worldwide with over 40 features including temperature, precipitation, humidity, air quality, and geographical data.

## Dataset Information
- **Source**: [World Weather Repository on Kaggle](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository/code)
- **Size**: 91,392 records with 41 features
- **Time Period**: Recent weather data with timestamps
- **Geographic Coverage**: Global coverage with latitude/longitude coordinates

## Project Structure
```
├── GlobalWeatherRepository.csv         # Original dataset
├── weather_analysis_complete.py        # Comprehensive analysis script
├── requirements.txt                    # Python dependencies
├── ASSESSMENT_REPORT.md                # Doc Describing what the project does
├── README.md                           # Project documentation
├── visualizations/                     # Generated charts and plots
├── models/                             # Saved model files
└── reports/                            # Analysis reports
```

## Technologies Used
- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Machine learning models
- **SciPy**: Statistical analysis
- **Joblib**: Model persistence


## Running the Script 
Although you don't need to run the script since it has been run twice but if you want to check the validation of the project you can do so by following these steps:
1. **Clone / extract** the project code (from GitHub or the provided `.zip`).
   - After extracting the `.zip`, **navigate into the project directory**:
     ```bash
     cd PM-Accelerator-Weather-Analysis-main
     ```
2. **Create a virtual environment** (recommended) and **install dependencies**:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   
    ```
3. **Run the script**:
   ```bash
   python weather_analysis.py
   ```
4. **Review outputs** in:
   - `visualizations/` (all charts)
   - `models/` (saved model & scaler)
   - `reports/` (the auto‑generated summary Markdown)

