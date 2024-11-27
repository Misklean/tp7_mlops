import joblib

def transpile_tree(model_path):
    # Charger le modèle d'arbre de décision
    model = joblib.load(model_path)
    if hasattr(model, 'tree_') is False:
        raise ValueError("Le modèle chargé n'est pas un arbre de décision.")

    # Extraire les informations nécessaires
    tree = model.tree_
    feature = tree.feature  # Index des features utilisées pour les splits
    threshold = tree.threshold  # Valeurs seuils utilisées pour les splits
    children_left = tree.children_left  # Fils gauche
    children_right = tree.children_right  # Fils droit
    values = tree.value  # Valeurs des feuilles

    def generate_code(node):
        """Génère récursivement le code C pour un nœud donné."""
        if children_left[node] == -1 and children_right[node] == -1:
            # C'est une feuille : retourner la classe prédite
            class_id = values[node].argmax()
            return f"return {class_id};"
        else:
            # C'est un nœud interne : ajouter des conditions
            left_code = generate_code(children_left[node])
            right_code = generate_code(children_right[node])
            return (
                f"if (features[{feature[node]}] <= {threshold[node]:.6f}) {{\n"
                f"    {left_code}\n"
                f"}} else {{\n"
                f"    {right_code}\n"
                f"}}"
            )

    # Générer le corps de la fonction
    tree_code = generate_code(0)

    # Générer le code C complet
    c_code = f"""
    #include <stdio.h>

    int predict_tree(float *features, int n_features) {{
    {tree_code}
    }}

    int main() {{
    // Exemple d'entrée
    float features[] = {{5.1, 3.5, 1.4, 0.2}};
    int n_features = 4;

    int prediction = predict_tree(features, n_features);
    printf("Prediction: %d\\n", prediction);

    return 0;
    }}
    """

    output_file = "decision_tree_model.c"
    with open(output_file, "w") as f:
        f.write(c_code)

    # Commande de compilation
    compile_command = f"gcc {output_file} -o decision_tree_model -lm"

    # Afficher ou compiler directement
    import os
    if os.system(compile_command) == 0:
        print(f"Compilation réussie. Exécutez le binaire avec ./decision_tree_model")
    else:
        print(f"Commande de compilation : {compile_command}")

# transpile_tree(model_path="decision_tree.joblib")