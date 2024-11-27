def build_model():
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    import joblib

    # Load the dataset
    df = pd.read_csv('houses.csv')

    # Define features and target
    X = df[['size', 'nb_rooms', 'garden']]

    # Convert price into a binary classification (e.g., high = 1, low = 0)
    median_price = df['price'].median()
    y = (df['price'] > median_price).astype(int)  # 1 for high price, 0 for low price

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X, y)

    # Save the trained model
    joblib.dump(model, "logistic_regression.joblib")

build_model()