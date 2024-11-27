import joblib
import numpy as np

# Load the trained decision tree model
model = joblib.load("decision_tree.joblib")

# Define a sample input (e.g., features like size, number of rooms, etc.)
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])  # Example input from Iris dataset

# Make a prediction
prediction = model.predict(sample_input)

# Display the result
print(f"Predicted class: {prediction[0]}")
