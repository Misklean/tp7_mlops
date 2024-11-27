import joblib

def transpile_linear(model_path):
    # Charger le modèle joblib
    model = joblib.load(model_path)

    # Récupérer les coefficients et l'intercept
    coefficients = model.coef_
    intercept = model.intercept_

    # Générer le code C
    def generate_c_code(coefficients, intercept):
        n_features = len(coefficients)
        coeffs_array = ", ".join(f"{coef:.6f}f" for coef in coefficients)

        code = f"""
    #include <stdio.h>

    float prediction(float *features, int n_features) {{
        float coefficients[{n_features}] = {{ {coeffs_array} }};
        float intercept = {intercept:.6f}f;
        float result = intercept;

        for (int i = 0; i < n_features; i++) {{
            result += coefficients[i] * features[i];
        }}

        return result;
    }}

    int main() {{
        float features[{n_features}] = {{ 60, 1, 0 }};
        int n_features = {n_features};
        float result = prediction(features, n_features);
        printf("Prediction: %f\\n", result);
        return 0;
    }}
    """
        return code

    # Générer le fichier C
    c_code = generate_c_code(coefficients, intercept)
    output_file = "linear_model.c"
    with open(output_file, "w") as f:
        f.write(c_code)

    # Commande de compilation
    compile_command = f"gcc {output_file} -o linear_model -lm"

    # Afficher ou compiler directement
    import os
    if os.system(compile_command) == 0:
        print(f"Compilation réussie. Exécutez le binaire avec ./linear_model")
    else:
        print(f"Commande de compilation : {compile_command}")

# transpile_linear(model_path="linear_regression.joblib")