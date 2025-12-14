import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, classification_report

# Importamos los modelos específicos necesarios para la re-inicialización dentro de las funciones
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Importaciones para SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def get_preprocessor(X):
    """
    Helper para crear el preprocesador estándar (numérico y categórico).
    """
    categorical_features = X.select_dtypes(include=['category', 'object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def evaluate_models(X, y, models, dataset_name):
    """
    Estrategia 1: Benchmark Convencional.
    Entrena y evalúa una lista de modelos en un dataset sin modificar pesos ni datos.
    """
    print(f"--- Evaluando Dataset: {dataset_name} ---")
    
    results_list = []
    
    # Preprocesamiento
    preprocessor = get_preprocessor(X)
    
    # División de datos (70/30 estratificada)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    for model_name, model in models.items():
        print(f"\nProbando modelo: {model_name}...")
        
        # Pipeline estándar
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Métricas
        print("Matriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))
        
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        print(f"Balanced Accuracy: {bal_acc:.3f}")
        print(f"F1-Score (clase 'mala'): {f1:.3f}")
        
        results_list.append({
            'dataset': dataset_name,
            'model': model_name,
            'balanced_accuracy': bal_acc,
            'f1_score_bad_class': f1,
            'Tipo': '1. Convencional'
        })
        
    return results_list

def evaluate_cost_sensitive_models(X, y, models, dataset_name):
    """
    Estrategia 2: Coste Sensible.
    Entrena modelos usando class_weight='balanced'.
    """
    print(f"--- Evaluando Dataset (Sensible a Costes): {dataset_name} ---")
    
    results_list = []
    preprocessor = get_preprocessor(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    for model_name, model in models.items():
        print(f"\nProbando modelo: {model_name} (Sensible al Coste)...")
        
        # Re-inicializamos el modelo con class_weight='balanced'
        if model_name == 'Logistic Regression':
            model_cs = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        elif model_name == 'Random Forest':
            model_cs = RandomForestClassifier(random_state=42, class_weight='balanced')
        elif model_name == 'SVC':
            model_cs = SVC(random_state=42, class_weight='balanced')
        elif model_name == 'Decision Tree':
            model_cs = DecisionTreeClassifier(random_state=42, class_weight='balanced')
        else:
            # Fallback si el modelo no está en la lista explícita
            try:
                model_cs = model.set_params(class_weight='balanced')
            except:
                print(f"Advertencia: No se pudo aplicar class_weight al modelo {model_name}")
                model_cs = model

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model_cs)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        print("Matriz de Confusión:")
        print(confusion_matrix(y_test, y_pred))
        
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        print(f"Balanced Accuracy: {bal_acc:.3f}")
        print(f"F1-Score (clase 'mala'): {f1:.3f}")
        
        try:
            target_names = [f'Clase {i}' for i in sorted(y.unique())]
            print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        except:
            print(classification_report(y_test, y_pred, zero_division=0))

        results_list.append({
            'dataset': dataset_name,
            'model': model_name,
            'balanced_accuracy': bal_acc,
            'f1_score_bad_class': f1,
            'Tipo': '2. Coste Sensible (class_weight)'
        })
        
    return results_list

def evaluate_smote_models(X, y, models, dataset_name):
    """
    Estrategia 3: Balanceo de Datos (SMOTE).
    Usa un pipeline de imblearn para aplicar SMOTE solo en el train set.
    """
    print(f"--- Evaluando Dataset (Balanceo SMOTE): {dataset_name} ---")
    
    results_list = []
    preprocessor = get_preprocessor(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    for model_name, model in models.items():
        
        # Pipeline de IMBLearn: SMOTE + Modelo Convencional
        pipeline_smote = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', model.set_params(class_weight=None)) # Aseguramos que no tenga pesos
        ])
        
        pipeline_smote.fit(X_train, y_train)
        y_pred = pipeline_smote.predict(X_test)
        
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        results_list.append({
            'dataset': dataset_name,
            'model': model_name,
            'balanced_accuracy': bal_acc,
            'f1_score_bad_class': f1,
            'Tipo': '3. Balanceo (SMOTE)'
        })
        
    return results_list