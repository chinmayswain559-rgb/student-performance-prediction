import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODEL_FILE = "model.pkl"
FEATURES = ["attendance", "study_hours", "previous_marks", "assignments"]

ADVICE = {
    "High": "Great work. Keep your routine.",
    "Medium": "Good. Study a bit more each day and finish all assignments.",
    "Low": "Needs improvement. Attend more classes and study daily.",
}


def load_data():
    df = pd.read_csv("students.csv")
    df = df.drop_duplicates()   # remove duplicate rows
    df = df.dropna()            # remove rows with missing values
    return df


def train():
    df = load_data()
    X = df[FEATURES]
    y = df["performance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # StandardScaler does feature scaling, Pipeline keeps scaling + model together
    models = {
        "Logistic Regression": Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000))]),
        "Decision Tree": Pipeline([("scale", StandardScaler()), ("model", DecisionTreeClassifier(random_state=42))]),
        "Random Forest": Pipeline([("scale", StandardScaler()), ("model", RandomForestClassifier(random_state=42))]),
    }
    best_model, best_acc, best_name = None, 0, ""
    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=5).mean()  # 5-fold cross-validation
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print("==", name, "==")
        print("CV accuracy:", round(cv, 3))
        print("Accuracy   :", round(acc, 3))
        print("Precision  :", round(precision_score(y_test, pred, average="weighted"), 3))
        print("Recall     :", round(recall_score(y_test, pred, average="weighted"), 3))
        print("F1 score   :", round(f1_score(y_test, pred, average="weighted"), 3))
        print(confusion_matrix(y_test, pred))
        if acc > best_acc:
            best_model, best_acc, best_name = model, acc, name
    print("Best model:", best_name)
    joblib.dump(best_model, MODEL_FILE)  # save the best model
    print("Saved as", MODEL_FILE)
    return best_model


def get_model():
    # use the saved model if it exists, otherwise train and save
    if os.path.exists(MODEL_FILE):
        print("Loaded saved model:", MODEL_FILE)
        return joblib.load(MODEL_FILE)
    return train()


def main():
    model = get_model()
    while True:
        print("\nStudent Performance Prediction")
        a = float(input("Attendance % (40-100): "))
        h = float(input("Study hours per day (0-8): "))
        m = float(input("Previous marks (30-100): "))
        s = float(input("Assignments done (0-10): "))
        row = pd.DataFrame([[a, h, m, s]], columns=FEATURES)
        result = model.predict(row)[0]
        print("Predicted performance:", result)
        print("Suggestion:", ADVICE[result])
        if input("Predict another? (y/n): ").lower() != "y":
            break


if __name__ == "__main__":
    main()
