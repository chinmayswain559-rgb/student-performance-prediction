import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ADVICE = {
    "High": "Great work. Keep your routine.",
    "Medium": "Good. Study a bit more each day and finish all assignments.",
    "Low": "Needs improvement. Attend more classes and study daily.",
}


def train():
    df = pd.read_csv("students.csv")
    X = df.drop(columns="performance")
    y = df["performance"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
    }
    best_model, best_acc, best_name = None, 0, ""
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        print("==", name, "==")
        print("Accuracy :", round(acc, 3))
        print("Precision:", round(precision_score(y_test, pred, average="weighted"), 3))
        print("Recall   :", round(recall_score(y_test, pred, average="weighted"), 3))
        print("F1 score :", round(f1_score(y_test, pred, average="weighted"), 3))
        print(confusion_matrix(y_test, pred))
        if acc > best_acc:
            best_model, best_acc, best_name = model, acc, name
    print("Best model:", best_name)
    return best_model, X.columns


def main():
    model, columns = train()
    while True:
        print("\nStudent Performance Prediction")
        a = float(input("Attendance % (40-100): "))
        h = float(input("Study hours per day (0-8): "))
        m = float(input("Previous marks (30-100): "))
        s = float(input("Assignments done (0-10): "))
        row = pd.DataFrame([[a, h, m, s]], columns=columns)
        result = model.predict(row)[0]
        print("Predicted performance:", result)
        print("Suggestion:", ADVICE[result])
        if input("Predict another? (y/n): ").lower() != "y":
            break


if __name__ == "__main__":
    main()
