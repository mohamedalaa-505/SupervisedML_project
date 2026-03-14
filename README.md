# CharityML: Income Prediction Project

## Project Description
CharityML is a fictitious charity organization in Silicon Valley that aims to provide financial support to people eager to learn machine learning. After analyzing past donations, CharityML discovered that all donors earned more than $50,000 annually. The goal of this project is to **predict potential donors in California** using census data to reduce overhead costs and improve targeting efficiency. You will build and optimize supervised learning algorithms to identify the most likely donors.

## Task Overview & Requirements
The project addresses the following tasks:

### 1. Data Exploration
- Calculate the **number of records**, **number of individuals with income >$50K**, **number of individuals with income <=$50K**, and **percentage of individuals with income >$50K**.  
- Explore numerical and categorical features.

### 2. Data Preprocessing
- Handle missing values and outliers.
- Feature engineering (e.g., `capital_net`, `is_married`).
- Encode categorical features using **one-hot encoding**.
- Address skewness for numerical features.

### 3. Naive Predictor
- Implement a naive predictor as a baseline.
- Compute **accuracy** and **F1 score** for benchmark comparison.

### 4. Model Training & Evaluation
- Train multiple supervised models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - AdaBoost
  - XGBoost
- Build **pipelines** for preprocessing + model training.
- Perform **cross-validation** and visualize model performance.
- Select the **best model** based on accuracy, F1 score, and computational cost.

### 5. Model Tuning
- Tune the best model using **GridSearchCV** or explain why tuning is not required.
- Evaluate the **optimized vs unoptimized model** on training and test sets.

### 6. Feature Analysis
- Identify and rank the **top 5 relevant features** for income prediction.
- Compare chosen features with `feature_importances_` from the final model.
- Analyze model performance using only the top features.

### 7. Deployment
- Save the final model as `income_model.pkl`.
- Deploy using Streamlit (`streamlitapp.py`) for interactive prediction.

## Data Visualization
- Correlation heatmaps, distributions, boxplots.
- Target vs feature analysis (age, gender, marital status, occupation, education, relationship, race).
- Model comparison bar plots.
- Feature importance plots.

## Results
- Best model: **Tuned XGBoost**  
- Performance metrics:
  - Cross-validated Accuracy & F1
  - Test Accuracy & F1
- Top features identified and analyzed.

## How to Run
1. Clone the repository  
2. Install dependencies: `pip install -r requirements.txt`  
3. Run analysis notebook: `Final_Project.ipynb`  
4. Run Streamlit app: `streamlit run streamlitapp.py`

## Contributors 
[Habiba Furany](https://github.com/habiba-furany)
[Mohamed Alaa](https://github.com/mohamedalaa-505)
