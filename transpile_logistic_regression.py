import joblib
import numpy as np

def transpile_logisitc(model_path):
    # Charger le modèle joblib
    model = joblib.load(model_path)

    # Vérifier si le modèle est une régression logistique
    if "LogisticRegression" not in str(type(model)):
        raise ValueError("Le modèle chargé n'est pas une régression logistique.")

    # Récupérer les coefficients et l'intercept
    coefficients = model.coef_[0]  # Pour une seule classe positive
    intercept = model.intercept_[0]

    # Générer le code C
    def generate_c_code_logistic(coefficients, intercept):
        n_features = len(coefficients)
        coeffs_array = ", ".join(f"{coef:.6f}f" for coef in coefficients)

        code = f"""
    #include <stdio.h>
    #include <math.h>

    // Fonction sigmoïde
    float sigmoid(float x) {{
        return 1.0f / (1.0f + expf(-x));
    }}

    // Fonction de prédiction
    float prediction(float *features, int n_features) {{
        float coefficients[{n_features}] = {{ {coeffs_array} }};
        float intercept = {intercept:.6f}f;
        float linear_combination = intercept;

        for (int i = 0; i < n_features; i++) {{
            linear_combination += coefficients[i] * features[i];
        }}

        return sigmoid(linear_combination);
    }}

    int main() {{
        float features[{n_features}] = {{ 60, 1, 0 }};  // Remplacez par vos valeurs
        int n_features = {n_features};
        float probability = prediction(features, n_features);
        printf("Probability: %f\\n", probability);
        return 0;
    }}
    """
        return code

    # Générer le fichier C
    c_code = generate_c_code_logistic(coefficients, intercept)
    output_file = "logistic_model.c"
    with open(output_file, "w") as f:
        f.write(c_code)

    # Commande de compilation
    compile_command = f"gcc {output_file} -o logistic_model -lm"

    # Afficher ou compiler directement
    import os
    if os.system(compile_command) == 0:
        print(f"Compilation réussie. Exécutez le binaire avec ./logistic_model")
    else:
        print(f"Commande de compilation : {compile_command}")

# transpile_logisitc(model_path="logistic_regression.joblib")