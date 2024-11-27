def build_model():
    from sklearn.datasets import load_iris
    from sklearn.tree import DecisionTreeClassifier
    import joblib

    X, y = load_iris(return_X_y=True)
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, "decision_tree.joblib")

build_model()