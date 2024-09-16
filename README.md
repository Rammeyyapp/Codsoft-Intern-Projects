# Titanic Survival Prediction

This project analyzes the Titanic dataset to predict the likelihood of a passenger's survival based on various factors like age, gender, ticket class, etc. The project uses data preprocessing and machine learning techniques to build a predictive model.

## Project Overview

The Titanic disaster is one of the most infamous shipwrecks in history. In this project, we use data from the Titanic passenger manifest to predict survival outcomes using logistic regression and other machine learning models. The project involves:

- Data cleaning and preprocessing
- Exploratory data analysis (EDA) with visualizations
- Feature engineering and selection
- Building and training machine learning models
- Evaluating model performance

## Dataset

The dataset contains information on 891 passengers, including:
- `PassengerId`
- `Survived` (0 = No, 1 = Yes)
- `Pclass` (Ticket class)
- `Name`, `Sex`, `Age`
- `SibSp`, `Parch` (Number of siblings/spouses/parents aboard)
- `Ticket`, `Fare`
- `Cabin` and `Embarked` (Port of Embarkation)

You can download the Titanic dataset from [Kaggle](https://www.kaggle.com/c/titanic/data).

## Project Workflow

1. **Data Preprocessing**: Handling missing values, encoding categorical features, scaling numerical features.
2. **Exploratory Data Analysis (EDA)**: Visualizing patterns and correlations using libraries like Matplotlib and Seaborn.
3. **Feature Engineering**: Creating new features such as family size, cabin group, etc.
4. **Model Building**: Using logistic regression and testing other models (like Decision Trees or Random Forest).
5. **Model Evaluation**: Evaluating accuracy, precision, recall, and plotting confusion matrices.

## Key Insights

- Gender and class have a strong influence on survival rate.
- Younger passengers and females had higher survival rates.
- Passengers with higher ticket classes were more likely to survive.

## Results

The final logistic regression model achieved an accuracy of **X%** on the test set, with precision, recall, and F1-score being used to measure model performance.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/Codsoft-Intern-Projects/titanic-survival-prediction.git
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Titanic_Survival_Prediction.ipynb
   ```

## Libraries Used

- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Conclusion

This project highlights the importance of feature engineering and data preprocessing in building effective machine learning models. Logistic regression performed well in predicting passenger survival, with further improvements possible by trying advanced models.

## Acknowledgments

- Kaggle for providing the [Titanic dataset](https://www.kaggle.com/c/titanic/data).
- [Codesoft](https://www.codesoft.com) for the opportunity to work on this project.

-------------------------------------------------------------------

This `README.md` file gives an overview of the project, steps, results, and instructions for using the project. You can adjust the percentage in the results and any specific features of your project.