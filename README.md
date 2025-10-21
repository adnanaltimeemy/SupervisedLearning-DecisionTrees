# Decision Tree Classification on Titanic Dataset 🚢

This project demonstrates how to use a **Decision Tree Classifier** to predict passenger survival on the famous **Titanic dataset**.  
It forms part of a data science learning series focused on understanding supervised machine learning algorithms through hands-on Python notebooks.

---

## 🧠 Project Overview

The notebook walks through:
- Importing and cleaning the Titanic dataset  
- Handling missing data and encoding categorical features  
- Splitting data into training and testing sets  
- Building and training a **Decision Tree Classifier** using Scikit-learn  
- Evaluating the model using accuracy, confusion matrix, and classification metrics  
- Visualising the trained decision tree and feature importance  

The main objective is to explore how tree-based models make decisions and identify which passenger attributes are most influential in survival prediction.

---

## 📂 Files Included

| File | Description |
|------|--------------|
| `decision_tree_titanic 2.ipynb` | Jupyter Notebook implementing the entire project |
| `titanic.csv` | Dataset used for training and testing |
| `README.md` | Project documentation (this file) |

---

## 📊 Dataset Description

The **Titanic dataset** contains data about passengers aboard the RMS Titanic.  
Each record includes information such as age, gender, class, fare, and survival status.

| Column | Description |
|---------|-------------|
| `PassengerId` | Passenger identifier |
| `Survived` | 0 = Did not survive, 1 = Survived |
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Name` | Passenger’s name |
| `Sex` | Male or Female |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number (may contain missing values) |
| `Embarked` | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## ⚙️ Installation and Dependencies

Ensure you have **Python 3.8+** and the following libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

You can then start Jupyter Notebook with:
```bash
jupyter notebook
```

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/Decision_Tree_Titanic.git
   cd Decision_Tree_Titanic
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook "decision_tree_titanic 2.ipynb"
   ```

3. Run all cells to reproduce the analysis, model training, and evaluation.

---

## 🧩 Model Explanation

The **Decision Tree Classifier** is a non-parametric supervised learning algorithm that splits data recursively based on feature values to create decision rules.

Key steps:
- **Preprocessing:** handle missing values and convert categorical features into numerical format using `LabelEncoder` or `get_dummies`.  
- **Training:** use Scikit-learn’s `DecisionTreeClassifier` with a suitable criterion (`entropy` or `gini`).  
- **Evaluation:** assess model performance using accuracy, confusion matrix, and classification report.  
- **Visualisation:** display the decision tree structure and feature importance plot.

Example code snippet:
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=0)
model.fit(X_train, y_train)
```

---

## 📈 Results and Visualisations

### ✅ Model Performance

| Metric | Score |
|--------|--------|
| **Accuracy (test set)** | **0.83 (83%)** |
| **Training accuracy** | **0.89 (89%)** |

### 🔢 Confusion Matrix
| | Predicted Survived | Predicted Died |
|---|---|---|
| **Actual Survived** | 97 | 13 |
| **Actual Died** | 19 | 50 |

### 🌳 Visualisations
The notebook generates:
1. **Correlation heatmap** showing relationships between variables  
2. **Decision tree plot** using `plot_tree()` for interpretability  
3. **Feature importance chart** ranking key predictors (e.g., `Sex`, `Pclass`, `Fare`, `Age`)  

---

## 🔮 Future Improvements

- Add hyperparameter tuning using `GridSearchCV`  
- Compare performance with **Random Forest** and **Logistic Regression**  
- Implement **cross-validation** for model robustness  
- Handle missing data using imputation techniques (e.g., median, KNN)  
- Deploy model with **Streamlit** or **Flask**

---

## 🤝 Contributing

Contributions are welcome!  
1. Fork the repository  
2. Create a branch (`git checkout -b feature-update`)  
3. Commit your changes (`git commit -m "Added confusion matrix visualisation"`)  
4. Push and open a Pull Request  

---

## 🪪 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and share it for educational or research purposes.

---

## 🙏 Acknowledgements

- Dataset: [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  
- Developed by **Adnan Altimeemy**  
  Data Scientist and Educater 
  Educational purpose: introducing students to decision tree algorithms and data preprocessing in Python.
