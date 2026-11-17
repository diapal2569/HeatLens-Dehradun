#!/usr/bin/env python
# coding: utf-8

# HeatLens Dehradun

# 1. Project Overview
#    This project analyses historical weather conditions in Dehradun.
#    It uses previous weather observations to predict whether the
#    apparent temperature may reach an unusually high level during
#    the next six hours.
# 
# 2. Importing Libraries
# 
# 
# 
# 3. Collecting Weather Data
# 
# 
# 
# 4. Data Understanding
# 
# 
# 
# 5. Data Cleaning
# 
# 
# 
# 6. Exploratory Data Analysis
# 
# 
# 
# 7. Feature Engineering
# 
# 
# 
# 8. Model Building
# 
# 
# 
# 9. Model Evaluation
# 
# 
# 
# 10. Conclusion

# In[1]:


get_ipython().run_line_magic('pip', 'install pandas numpy matplotlib seaborn scikit-learn requests')


# In[2]:


import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

print("Libraries imported successfully.")


# 1. Collecting Weather Data
# 
# The project uses historical hourly weather data for Dehradun.
# The selected variables include temperature, apparent temperature,
# humidity, rainfall, wind speed, cloud cover and solar radiation.

# In[3]:


url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 30.3165,
    "longitude": 78.0322,
    "start_date": "2025-01-01",
    "end_date": "2026-10-10",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "wind_speed_10m",
        "cloud_cover",
        "shortwave_radiation"
    ],
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params, timeout=60)
response.raise_for_status()

weather_data = response.json()

df = pd.DataFrame(weather_data["hourly"])
df["time"] = pd.to_datetime(df["time"])

df.to_csv(DATA_DIR / "dehradun_weather.csv", index=False)

print("Data downloaded successfully.")
print("Dataset shape:", df.shape)

df.head()


# 2. Data Understanding and Quality Check

# In[4]:


print("Dataset information:")
df.info()

print("\nFirst five records:")
display(df.head())

print("\nMissing values:")
display(df.isnull().sum())

print("\nDuplicate records:", df.duplicated().sum())

print("\nStatistical summary:")
display(df.describe())


# 3. Exploratory Data Analysis

# In[5]:


plt.figure(figsize=(11, 5))

plt.plot(df["time"], df["apparent_temperature"], linewidth=0.7)

plt.title("Hourly Apparent Temperature in Dehradun")
plt.xlabel("Date")
plt.ylabel("Apparent Temperature (°C)")
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "apparent_temperature_trend.png", dpi=300)
plt.show()


# 4. Feature Engineering
# 
# Feature engineering is the process of creating useful input variables
# for a machine learning model.
# 
# In this project, previous weather observations are used to create
# lag features and rolling averages. These features help the model
# understand recent weather patterns.

# In[6]:


df = df.sort_values("time").drop_duplicates(subset="time").copy()
df = df.set_index("time")

weather_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation"
]

# Create features using previous observations only
features = pd.DataFrame(index=df.index)

for col in weather_columns:
    for lag in [1, 3, 6, 12, 24]:
        features[f"{col}_lag_{lag}"] = df[col].shift(lag)

    for window in [6, 12, 24]:
        features[f"{col}_rolling_mean_{window}"] = (
            df[col].shift(1).rolling(window).mean()
        )

features["hour"] = df.index.hour
features["month"] = df.index.month

# Maximum apparent temperature in the next six hours
future_values = pd.concat(
    [df["apparent_temperature"].shift(-i) for i in range(1, 7)],
    axis=1
)

target_temperature = future_values.max(axis=1)

dataset = features.copy()
dataset["future_max_apparent_temperature"] = target_temperature

dataset = dataset.dropna()

print("Feature engineering completed.")
print("New dataset shape:", dataset.shape)

display(dataset.head())


# In[7]:


split_index = int(len(dataset) * 0.80)

train_data = dataset.iloc[:split_index].copy()
test_data = dataset.iloc[split_index:].copy()

threshold = train_data[
    "future_max_apparent_temperature"
].quantile(0.90)

train_data["high_temperature"] = (
    train_data["future_max_apparent_temperature"] > threshold
).astype(int)

test_data["high_temperature"] = (
    test_data["future_max_apparent_temperature"] > threshold
).astype(int)

print("High-temperature threshold:", round(threshold, 2), "°C")
print("\nTraining records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining target distribution:")
display(train_data["high_temperature"].value_counts())


# In[8]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

X_train = train_data.drop(
    columns=["future_max_apparent_temperature", "high_temperature"]
)

X_test = test_data.drop(
    columns=["future_max_apparent_temperature", "high_temperature"]
)

y_train = train_data["high_temperature"]
y_test = test_data["high_temperature"]

# Remove the final six training rows to avoid crossing the split boundary
X_train = X_train.iloc[:-6]
y_train = y_train.iloc[:-6]

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# In[10]:


from sklearn.impute import SimpleImputer

# Calculate threshold without the six boundary observations
threshold = train_data[
    "future_max_apparent_temperature"
].iloc[:-6].quantile(0.90)

train_data["high_temperature"] = (
    train_data["future_max_apparent_temperature"] > threshold
).astype(int)

test_data["high_temperature"] = (
    test_data["future_max_apparent_temperature"] > threshold
).astype(int)

X_train = train_data.drop(
    columns=["future_max_apparent_temperature", "high_temperature"]
).iloc[:-6]

X_test = test_data.drop(
    columns=["future_max_apparent_temperature", "high_temperature"]
)

y_train = train_data["high_temperature"].iloc[:-6]
y_test = test_data["high_temperature"]

print("Threshold:", round(threshold, 2), "°C")
print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())


# 5. Machine Learning Model Building
# 
# Three classification algorithms are used in this project:
# 1. Logistic Regression
# 2. Decision Tree
# 3. Random Forest
# 
# A baseline model is also included for comparison.

# In[11]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

models = {
    "Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ]),

    "Decision Tree": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

print("Models are ready.")


# In[12]:


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

results = []
trained_models = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    trained_models[name] = model

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

display(
    results_df.round(3).sort_values(
        by="F1 Score", ascending=False
    )
)


# In[22]:


plt.figure(figsize=(10, 5))

results_plot = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
]

results_plot.plot(kind="bar", figsize=(11, 5))

plt.title("Comparison of Machine Learning Models")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_comparison.png",
    dpi=300
)

plt.show()


# In[23]:


best_model_name = (
    results_df.sort_values("F1 Score", ascending=False)
    .iloc[0]["Model"]
)

best_model = trained_models[best_model_name]

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not High", "High"],
    yticklabels=["Not High", "High"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.show()

print("Best model based on F1 Score:", best_model_name)


# In[15]:


print("Classification Report:", best_model_name)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["Not High", "High"],
        zero_division=0
    )
)


# In[16]:


results_df.to_csv(
    OUTPUT_DIR / "model_results.csv",
    index=False
)

print("Results saved successfully.")
print("Output folder:", OUTPUT_DIR.resolve())


#  6. Final Data Validation
# 
# Before model evaluation, incomplete future prediction windows are
# removed. Each target must contain six complete future hourly values.

# In[19]:


# Ensure that the data has a complete hourly time index
df_hourly = df[weather_columns].sort_index().asfreq("h")

expected_hours = pd.date_range(
    start=df_hourly.index.min(),
    end=df_hourly.index.max(),
    freq="h"
)

missing_hours = len(expected_hours.difference(df_hourly.index))

print("Missing hourly timestamps:", missing_hours)

# Rebuild lag and rolling features
features = pd.DataFrame(index=df_hourly.index)

for col in weather_columns:
    for lag in [1, 3, 6, 12, 24]:
        features[f"{col}_lag_{lag}"] = df_hourly[col].shift(lag)

    for window in [6, 12, 24]:
        features[f"{col}_rolling_mean_{window}"] = (
            df_hourly[col].shift(1).rolling(window).mean()
        )

features["hour"] = features.index.hour
features["month"] = features.index.month

# Calculate the following six complete hourly values
future_values = pd.concat(
    [
        df_hourly["apparent_temperature"].shift(-i)
        for i in range(1, 7)
    ],
    axis=1
)

# Keep only rows with all six future values
complete_horizon = future_values.notna().all(axis=1)

target_temperature = future_values.max(axis=1)

dataset = features.copy()
dataset["future_max_apparent_temperature"] = target_temperature

dataset = dataset.loc[complete_horizon].dropna().copy()

print("\nFinal validated dataset shape:", dataset.shape)
print("First timestamp:", dataset.index.min())
print("Last timestamp:", dataset.index.max())


# In[20]:


split_index = int(len(dataset) * 0.80)

train_data = dataset.iloc[:split_index].copy()
test_data = dataset.iloc[split_index:].copy()

# Do not train on examples whose six-hour target enters the test period
test_start = test_data.index.min()

valid_train = (
    train_data.index + pd.Timedelta(hours=6)
) < test_start

train_data = train_data.loc[valid_train].copy()

# Calculate threshold using training data only
threshold = train_data[
    "future_max_apparent_temperature"
].quantile(0.90)

train_data["high_temperature"] = (
    train_data["future_max_apparent_temperature"] > threshold
).astype(int)

test_data["high_temperature"] = (
    test_data["future_max_apparent_temperature"] > threshold
).astype(int)

X_train = train_data.drop(
    columns=["future_max_apparent_temperature", "high_temperature"]
)

X_test = test_data.drop(
    columns=["future_max_apparent_temperature", "high_temperature"]
)

y_train = train_data["high_temperature"]
y_test = test_data["high_temperature"]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Threshold:", round(threshold, 2), "°C")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# In[21]:


results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    trained_models[name] = model

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

display(
    results_df.round(3).sort_values(
        "F1 Score", ascending=False
    )
)


# In[25]:


# Save the final model comparison
results_df.to_csv(
    OUTPUT_DIR / "final_model_comparison.csv",
    index=False
)

# Save the final confusion matrix
final_cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

pd.DataFrame(
    final_cm,
    index=["Actual Not High", "Actual High"],
    columns=["Predicted Not High", "Predicted High"]
).to_csv(OUTPUT_DIR / "final_confusion_matrix.csv")

# Save dataset information
project_summary = {
    "Dataset rows": len(dataset),
    "Dataset columns": dataset.shape[1],
    "Training records": len(X_train),
    "Testing records": len(X_test),
    "Training start": str(X_train.index.min()),
    "Training end": str(X_train.index.max()),
    "Testing start": str(X_test.index.min()),
    "Testing end": str(X_test.index.max()),
    "High temperature threshold": threshold,
    "Best model": best_model_name
}

pd.Series(project_summary).to_csv(
    OUTPUT_DIR / "project_summary.csv",
    header=["Value"]
)

print("Final project results saved successfully.")
display(results_df.round(4))
display(pd.Series(project_summary))


# In[26]:


# Scatter plot: Temperature vs Apparent Temperature

plt.figure(figsize=(9, 5))

plt.scatter(
    df["temperature_2m"],
    df["apparent_temperature"],
    alpha=0.35,
    s=12
)

plt.title("Air Temperature vs Apparent Temperature")
plt.xlabel("Air Temperature (°C)")
plt.ylabel("Apparent Temperature (°C)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "temperature_scatter_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[27]:


# Correlation heatmap of weather variables

correlation_columns = [
    "temperature_2m",
    "relative_humidity_2m",
    "apparent_temperature",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "shortwave_radiation"
]

correlation_matrix = df[correlation_columns].corr()

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap of Weather Variables")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[28]:


# Histogram of apparent temperature

plt.figure(figsize=(9, 5))

sns.histplot(
    df_hourly["apparent_temperature"].dropna(),
    bins=30,
    kde=True
)

plt.title("Distribution of Apparent Temperature in Dehradun")
plt.xlabel("Apparent Temperature (°C)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "apparent_temperature_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[29]:


# Monthly average apparent temperature

monthly_avg = (
    df_hourly["apparent_temperature"]
    .resample("ME")
    .mean()
    .dropna()
)

plt.figure(figsize=(10, 5))

monthly_avg.plot(kind="bar")

plt.title("Monthly Average Apparent Temperature in Dehradun")
plt.xlabel("Month")
plt.ylabel("Average Apparent Temperature (°C)")
plt.xticks(
    range(len(monthly_avg)),
    monthly_avg.index.strftime("%b %Y"),
    rotation=45
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "monthly_average_temperature.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

display(monthly_avg.round(2))

