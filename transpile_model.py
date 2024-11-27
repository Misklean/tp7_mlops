import joblib
import sys
from transpile_decision_tree import transpile_tree
from transpile_linear_regression import transpile_linear
from transpile_logistic_regression import transpile_logisitc

model_path = sys.argv[1]

model = joblib.load(model_path)

# Vérifier si le modèle est un decision tree
if hasattr(model, 'tree_') is True:
    transpile_tree(model_path=model_path)

# Vérifier si le modèle est une régression logistique
if "LogisticRegression" in str(type(model)):
    transpile_logisitc(model_path=model_path)

# Vérifier si le modèle est une régression lineaire
if "LinearRegression" in str(type(model)):
    transpile_linear(model_path=model_path)

