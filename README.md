# 🚕 NYC Taxi Trip Analysis & Weather-Driven Predictive Modeling

An end-to-end **Data Analysis and Machine Learning project** investigating NYC taxi trip patterns and their relationship with weather and environmental conditions using real-world datasets from the **National Oceanic and Atmospheric Administration (NOAA)** and the **New York City Taxi & Limousine Commission (NYC TLC)**.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- Statsmodels
- Expectation-Maximization (EM) Clustering
- Regression Analysis
- Causal Inference
- Statistical Validation
- Bootstrap Confidence Intervals

---

## 📥 Datasets

**NOAA** 
https://www.noaa.gov/

**NYC Taxi & Limousine Commission**
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

### Dataset Setup

Download the required NOAA weather data and NYC TLC taxi trip data from the official sources and place them inside:

    data/

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

    git clone <your-github-repository-url>
    cd <repository-name>

### 2️⃣ Install Dependencies

Install all required Python packages:

    pip install pandas numpy scipy scikit-learn matplotlib seaborn statsmodels

### 3️⃣ Add the Datasets

Download the required datasets from the official NOAA and NYC TLC sources and place them inside:

    data/

### 4️⃣ Run the Analysis

The project scripts can be executed in the following order:

    python cleaning.py
    python EDA.py
    python EMCluster.py
    python EMValidation.py
    python Regression.py
    python causal_inference.py
    python modelComparison.py
    python bootstrap_confidence.py

---

## 📊 Project Workflow

     Original NOAA Weather Data
                    +
     NYC Taxi Trip Data
                    ↓
          Data Cleaning
                    ↓
          Data Integration
                    ↓
     Exploratory Data Analysis
                    ↓
        Feature Engineering
                    ↓
          ┌─────────┴─────────┐
          ↓                   ↓
    EM Clustering          Regression
          ↓                   ↓
    EM Validation       Model Analysis
          └─────────┬─────────┘
                    ↓
             Causal Inference
                    ↓
       Bootstrap Confidence
             Intervals
                    ↓
          Model Comparison
                    ↓
             Final Findings

---
