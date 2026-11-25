# Student Performance Prediction

A command-line app that predicts if a student's performance is High, Medium or Low.

## About
- Inputs: attendance %, study hours per day, previous marks, assignments done
- Output: predicted performance and a short suggestion
- Models used: Logistic Regression, Decision Tree, Random Forest
- Metrics shown: accuracy, precision, recall, F1 score, confusion matrix

## Data
`students.csv` is a sample dataset of 300 students made with Python (synthetic, not real students). Scores in this project are on this practice data only.

## How to run
1. Install: `pip install pandas scikit-learn`
2. Keep `students.csv` and `main.py` in the same folder
3. Run: `python main.py`
4. Enter the 4 numbers when asked

## Files
- `main.py` - trains the 3 models, picks the best and predicts
- `students.csv` - sample data
- `eda_performance.png` and `eda_correlation.png` - graphs

## Author
Chinmaya Swain
