# Analyzing-Data-with-Pandas-and-Visualizing-Results-with-Matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
try:
    df = pd.read_csv("iris.csv")
except FileNotFoundError:
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display basic information and clean the dataset
print(df.head())
df.info()
print(df.isnull().sum())
df = df.dropna()

# Basic statistics and grouping
print(df.describe())
if "petal length (cm)" in df.columns:
    species_group = df.groupby("species")["petal length (cm)"].mean()
else:
    species_group = df.groupby("species")["petal_length"].mean()
print(species_group)

# Visualization 1: Line chart (Trend of Sepal Length)
plt.figure(figsize=(10, 6))
x = df.index
if "sepal length (cm)" in df.columns:
    y = df["sepal length (cm)"]
else:
    y = df["sepal_length"]
plt.plot(x, y, marker='o', linestyle='-', label="Sepal Length")
plt.title("Trend of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.grid(True)
plt.show()

# Visualization 2: Bar chart (Average Petal Length by Species)
plt.figure(figsize=(8, 6))
species_group.plot(kind="bar", color="skyblue")
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.xticks(rotation=45)
plt.show()

# Visualization 3: Histogram (Distribution of Sepal Width)
plt.figure(figsize=(8, 6))
if "sepal width (cm)" in df.columns:
    width = df["sepal width (cm)"]
else:
    width = df["sepal_width"]
plt.hist(width, bins=15, color="lightgreen", edgecolor="black")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Visualization 4: Scatter plot (Sepal Length vs Petal Length)
plt.figure(figsize=(8, 6))
if "sepal length (cm)" in df.columns and "petal length (cm)" in df.columns:
    sepal_len = df["sepal length (cm)"]
    petal_len = df["petal length (cm)"]
else:
    sepal_len = df["sepal_length"]
    petal_len = df["petal_length"]
plt.scatter(sepal_len, petal_len, c='purple', alpha=0.7)
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.grid(True)
plt.show()
