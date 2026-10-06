#ROTIMI DANIEL O. (256706_COMPUTER_SCIENCE)

# Iris Species Classification & Exploratory Data Analysis

An exploratory data analysis (EDA) and machine learning project focused on predicting species in the classic **Iris Dataset** from Kaggle using Python, Pandas, Seaborn, and Scikit-Learn.

---

## 📌 Project Overview

This project is split into two major phases:
1. **Exploratory Data Analysis (EDA) & Data Visualization:** Inspecting data structure, calculating summary statistics, and visualizing feature distributions and relationships.
2. **Decision Tree Classification & Train/Test Split Experiment:** Building a Decision Tree model to classify species (`Setosa`, `Versicolor`, `Virginica`) and analyzing how varying the train/test split percentage impacts evaluation metrics.

---

## 📊 Phase 1: Exploratory Data Analysis & Visualizations

### Descriptive Statistics
* **Dataset Inspection:** Examined dataset structure using `.head()`, `.tail()`, `.shape`, `.info()`, and `.dtypes`.
* **Summary Metrics:** Calculated measures of central tendency and dispersion, including **Mean**, **Median**, **Mode**, **Variance**, **Standard Deviation**, **Skewness**, and **Kurtosis**.

### Data Visualizations
* **Histograms & KDE:** Analyzed feature distributions and density.
* **Box Plots:** Identified outliers and distribution spread across species.
* **Count Plots:** Verified class balance across the target species.
* **Scatter Plots & Pair Plots:** Visualized pairwise relationships and feature separability.
* **Correlation Heatmap:** Mapped linear relationships between numerical features.

---

## ⚙️ Phase 2: Decision Tree Classifier & Evaluation

A `DecisionTreeClassifier` was trained to classify Iris species based on numerical features (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`).

### Model Visualizations
* **Confusion Matrix:** Evaluated actual vs. predicted label distribution.
* **Decision Tree Structure:** Visualized learned split nodes and decision boundaries using `plot_tree`.

---

## 💡 Key Takeaways & Experiment Insights

A major focus of this project was testing how changing the **train/test split ratio** affects overall model performance:

| Train / Test Split Ratio | Observations & Performance |
| :---: | :--- |
| **50% / 50%** | Higher variance; smaller training sample led to lower stability across evaluation metrics. |
| **80% / 20%** | Strong performance, but higher risk of slight overfitting on small evaluation sets. |
| **75% / 25%** | **Optimal balance.** Produced consistent **Accuracy**, **Precision**, and **Recall** while maintaining strong generalization, making it the most reliable baseline for presentation. |

> **Key Conclusion:** Discrepancies in data splitting directly influence evaluation metrics. The **75/25 split** provided the ideal balance between training data density and reliable model validation.

---

## 🛠️ Tools & Libraries Used

* **Language:** Python
* **Data Manipulation:** `pandas`, `numpy`
* **Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (`DecisionTreeClassifier`, `train_test_split`, `metrics`)
* **Dataset Source:** [Kaggle - Iris Species](https://www.kaggle.com/datasets/uciml/iris)

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
