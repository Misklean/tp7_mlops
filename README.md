# Model Transpiler TP7

## Supported Models

This tool supports the following machine learning models:

- **Decision Tree**
- **Linear Regression**
- **Logistic Regression**

## Binary Sizes

The generated binary files for the models are as follows:

- **Decision Tree Model**: 16K
- **Linear Model**: 16K
- **Logistic Model**: 16K

## Usage

### Requirements

To use the tool, you need a `.joblib` file for the model. The `.joblib` file should be located in the root directory of the project.

### Creating a `.joblib` File

You can either use a pre-existing `.joblib` file or create one using the training scripts provided in the repository.

To create a `.joblib` file, run the relevant training script:

- **Train Decision Tree**: `train_decision_tree.py`
- **Train Linear Regression**: `train_linear_regression.py`
- **Train Logistic Regression**: `train_logistic_regression.py`

These scripts will generate `.joblib` files that can be used with the transpiler.

### Running the Transpiler

Once you have your `.joblib` file, you can run the model transpiler with the following command:

```bash
python transpile_model.py <joblib_file>
```

The script will detect the type of model based on the `.joblib` file and generate the corresponding binary for the model.

### Changing Predictions

If you need to modify the prediction logic or features, you will need to edit the relevant transpile file corresponding to the model type:

- For **Decision Tree**, modify the `transpile_decision_tree.py` file.
- For **Linear Regression**, modify the `transpile_linear_regression.py` file.
- For **Logistic Regression**, modify the `transpile_logistic_regression.py` file.

In these files, you can adjust the features or the logic for how predictions are made.

### Testing the Predictions

Each model includes a `predict` file that can be used to test whether the predictions are correct based on the input features.

To test the predictions, call the `predict` file with the appropriate input and verify the output.

## Notes

- The generated binaries are small (16K each), making them efficient to deploy.
- The script automatically recognizes the type of model in the `.joblib` file and handles the transpilation accordingly.
- If you need to adjust the prediction logic, it’s straightforward to modify the corresponding transpile function.

Feel free to modify the transpiler to support additional model types or customize the existing ones!