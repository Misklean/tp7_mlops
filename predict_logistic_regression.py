import joblib
import numpy as np

# Load the trained logistic regression model
model = joblib.load("logistic_regression.joblib")

# Define a sample input (e.g., size, nb_rooms, garden)
sample_input = np.array([[60, 1, 0]])  # Example: 120 sqm, 4 rooms, garden (1 = yes)

# Make a prediction
prediction = model.predict(sample_input)

# Display the result
print(f"Predicted class: {prediction[0]}")  # 0 = low price, 1 = high price
