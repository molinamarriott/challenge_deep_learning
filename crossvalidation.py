import pandas as pd
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.metrics import AUC
from scikeras.wrappers import KerasClassifier

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')



def get_data():
    data = pd.read_csv('data/train_data.csv')
    return data


def split_data(data):
    
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,2:], data.iloc[:,1], test_size=0.20)
    print(f'Train size: {X_train.shape}')
    print(f'Test size: {X_test.shape}')
    
    print(f'Train cases: {y_train.value_counts()}')
    print(f'Test cases: {y_test.value_counts()}')
    
    return X_train, X_test, y_train, y_test



# 1. Transformador para manejar outliers
class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.means_ = None
        self.stds_ = None
        self.is_numeric_ = None
        
    def fit(self, X, y=None):
        self.is_numeric_ = {col: np.issubdtype(X[col].dtype, np.number) 
                            for col in X.columns}
        self.means_ = {}
        self.stds_ = {}
        
        for col in X.columns:
            if self.is_numeric_[col]:
                lower = X[col].quantile(0.005)
                upper = X[col].quantile(0.995)
                filtered = X[col][(X[col] >= lower) & (X[col] <= upper)]
                self.means_[col] = filtered.mean()
                self.stds_[col] = filtered.std()
        
        return self
    
    
    def transform(self, X):
        X_transformed = X.copy()
        
        for col in X.columns:
            if self.is_numeric_[col] and self.stds_[col] > 0:
                upper_limit = self.means_[col] + self.threshold * self.stds_[col]
                lower_limit = self.means_[col] - self.threshold * self.stds_[col]
                
                # Recortar valores por encima y por debajo de los límites
                X_transformed[col] = X_transformed[col].clip(lower=lower_limit, upper=upper_limit)
                
        return X_transformed

# 2. Transformador para estandarización que maneja DataFrames
class DataFrameStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.columns_ = None
        self.is_numeric_ = None
        
    def fit(self, X, y=None):
        # Guardar nombres de columnas
        self.columns_ = X.columns.tolist()
        
        # Identificar columnas numéricas
        self.is_numeric_ = {col: np.issubdtype(X[col].dtype, np.number) 
                           for col in X.columns}
        
        # Filtrar solo columnas numéricas para estandarizar
        numeric_cols = [col for col in self.columns_ if self.is_numeric_[col]]
        if numeric_cols:
            self.scaler.fit(X[numeric_cols])
            
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # Filtrar solo columnas numéricas para estandarizar
        numeric_cols = [col for col in self.columns_ if self.is_numeric_[col]]
        if numeric_cols:
            X_transformed[numeric_cols] = self.scaler.transform(X[numeric_cols])
            
        return X_transformed

# 3. Transformador para selección de características basado en Gini/IV
class GiniSelector(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.4, max_features=20):
        self.threshold = threshold
        self.max_features = max_features
        self.feature_importances_ = None
        self.selected_features_ = None
        
    def _calculate_gini(self, X, y):
        # Inicializar diccionario para almacenar los valores de Gini
        gini_values = {}
        
        # Convertir y a Series si es un array
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        
        # Calcular Gini para cada variable
        for col in X.columns:
            if np.issubdtype(X[col].dtype, np.number):
                try:
                    # Ordenar los valores de X y calcular la curva de Lorenz
                    # usando el método del área bajo la curva
                    df_temp = pd.DataFrame({'X': X[col], 'y': y})
                    df_temp = df_temp.sort_values('X')
                    df_temp['y_cumsum'] = df_temp['y'].cumsum() / df_temp['y'].sum()
                    df_temp['X_cumsum'] = np.linspace(0, 1, len(df_temp))
                    
                    # Calcular el área bajo la curva
                    auc = np.trapz(df_temp['y_cumsum'], df_temp['X_cumsum'])
                    
                    # Gini = 2*AUC - 1 (normalizado entre -1 y 1)
                    gini = 2 * auc - 1
                    
                    # Valor absoluto del Gini para ordenamiento
                    gini_values[col] = abs(gini)
                except:
                    gini_values[col] = 0
            else:
                gini_values[col] = 0
                
        return gini_values
    
    def fit(self, X, y):
        # Calcular valores de Gini para todas las características
        self.feature_importances_ = self._calculate_gini(X, y)
        
        # Ordenar características por importancia
        sorted_features = sorted(self.feature_importances_.items(), 
                                 key=lambda x: x[1], reverse=True)
        
        # Seleccionar características que cumplen el umbral y el número máximo
        selected_features = []
        for feature, importance in sorted_features:
            if importance >= self.threshold and len(selected_features) < self.max_features:
                selected_features.append(feature)
                
        # Si no hay suficientes características que superen el umbral, tomar las top max_features
        if not selected_features or len(selected_features) < min(self.max_features, len(X.columns)):
            top_features = [feature for feature, _ in sorted_features[:self.max_features]]
            selected_features = list(set(selected_features + top_features))[:self.max_features]
            
        self.selected_features_ = selected_features
        return self
    
    def transform(self, X):
        # Devolver solo las columnas seleccionadas
        return X[self.selected_features_]

# 4. Transformador para eliminar variables correlacionadas
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.8, method='pearson'):
        self.threshold = threshold
        self.method = method
        self.features_to_drop_ = None
        self.feature_importances_ = None
        
    def fit(self, X, y=None):
        # Calcular matriz de correlación
        corr_matrix = X.corr(method=self.method)
        
        # Inicializar conjunto de características a eliminar
        self.features_to_drop_ = set()
        
        # Si tenemos importancia de características de un paso anterior, usarlas
        # De lo contrario, consideramos que todas las características tienen la misma importancia
        if hasattr(self, 'feature_importances_') and self.feature_importances_ is not None:
            importances = self.feature_importances_
        else:
            importances = {col: 1.0 for col in X.columns}
            
        # Recorrer la matriz triangular superior
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > self.threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    
                    # Eliminar la característica con menor importancia
                    if importances.get(col_i, 0) <= importances.get(col_j, 0):
                        self.features_to_drop_.add(col_i)
                    else:
                        self.features_to_drop_.add(col_j)
        
        return self
    
    def set_feature_importances(self, importances):
        """Establece la importancia de las características desde un paso anterior"""
        self.feature_importances_ = importances
        return self
    
    def transform(self, X):
        # Filtrar las características que no fueron eliminadas
        return X.drop(columns=list(self.features_to_drop_))

# Función para crear y ejecutar el pipeline completo
def crear_pipeline_preprocesamiento(X_train, y_train, 
                                   outlier_threshold=3,
                                   gini_threshold=0.1, 
                                   max_features=20,
                                   corr_threshold=0.7):
    """
    Crea y ejecuta un pipeline de preprocesamiento para modelado predictivo.
    
    Parámetros:
    -----------
    X_train : DataFrame
        Conjunto de datos de entrenamiento (características)
    y_train : Series o array
        Variable objetivo para el conjunto de entrenamiento
    outlier_threshold : float, default=3
        Umbral en desviaciones estándar para recortar outliers
    gini_threshold : float, default=0.1
        Umbral de Gini absoluto para selección de características
    max_features : int, default=20
        Número máximo de características a seleccionar
    corr_threshold : float, default=0.8
        Umbral de correlación para eliminar características redundantes
        
    Retorna:
    --------
    pipeline : Pipeline
        Pipeline entrenado
    X_train_transformed : DataFrame
        Datos de entrenamiento transformados
    """
    # Crear los componentes del pipeline
    outlier_clipper = OutlierClipper(threshold=outlier_threshold)
    scaler = DataFrameStandardScaler()
    gini_selector = GiniSelector(threshold=gini_threshold, max_features=max_features)
    correlation_filter = CorrelationFilter(threshold=corr_threshold)
    
    # Crear el pipeline
    pipeline = Pipeline([
        ('outlier_clipper', outlier_clipper),
        ('scaler', scaler),  # Añadido el paso de estandarización
        ('gini_selector', gini_selector),
        ('correlation_filter', correlation_filter)
    ])
    
    # Entrenar el pipeline
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    
    # Transferir las importancias de las características del selector al filtro de correlación
    correlation_filter.set_feature_importances(gini_selector.feature_importances_)
    
    # Imprimir información del proceso
    print(f"Variables originales: {X_train.shape[1]}")
    print(f"Variables después de selección por Gini: {len(gini_selector.selected_features_)}")
    print(f"Variables después de filtrar correlacionadas: {X_train_transformed.shape[1]}")
    print("\nTop 10 variables por importancia Gini:")
    sorted_importances = sorted(gini_selector.feature_importances_.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
    for feature, importance in sorted_importances:
        print(f"  {feature}: {importance:.4f}")
    
    print("\nVariables eliminadas por alta correlación:")
    for feature in correlation_filter.features_to_drop_:
        print(f"  {feature}")
    
    return pipeline, X_train_transformed




# Define your model creation function
def create_model(batch_size=10, epochs=10, optimizer='adam', activation='relu', neurons=[16, 16]):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=X_train_transformed.shape[1], activation=activation))
    for n in neurons[1:]:
        model.add(Dense(n, activation=activation))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


data = get_data()
X_train, X_test, y_train, y_test = split_data(data)


pipeline, X_train_transformed = crear_pipeline_preprocesamiento(
    X_train, y_train, 
    outlier_threshold=3,
    gini_threshold=0.3,
    max_features=20,
    corr_threshold=0.6
)

X_test_transformed = pipeline.transform(X_test)

model = KerasClassifier(build_fn=create_model, verbose=0)


param_grid = {
    'batch_size': [10, 16],
    'epochs': [8, 10, 12],
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'neurons': [[16, 16],
                [16, 8],
                [8, 8]]
}

all_results = []

# Manual grid search
for batch_size in [10, 16]:
    for epochs in [8, 10, 12]:
        for optimizer in ['adam', 'rmsprop']:
            for activation in ['relu', 'tanh']:
                for neurons in [[16, 16], [16, 8], [8, 8]]:
                    # Create and compile model
                    model = create_model(batch_size, epochs, optimizer, activation, neurons)
                    
                    # Use cross-validation manually
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)
                    cv_scores = []
                    
                    for train_idx, val_idx in kf.split(X_train_transformed):
                        # Use .iloc for pandas DataFrames
                        X_cv_train, X_cv_val = X_train_transformed.iloc[train_idx], X_train_transformed.iloc[val_idx]
                        
                        # Check if y_train is also a DataFrame/Series
                        if isinstance(y_train, (pd.DataFrame, pd.Series)):
                            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                        else:
                            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                        
                        # Train model
                        model.fit(X_cv_train, y_cv_train, epochs=epochs, batch_size=batch_size, verbose=0)
                        
                        # Calculate AUC instead of just accuracy
                        y_pred = model.predict(X_cv_val, verbose=0)
                        auc = roc_auc_score(y_cv_val, y_pred)
                        cv_scores.append(auc)
                    
                    # Average AUC across folds
                    mean_auc = np.mean(cv_scores)
                    params = {
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'optimizer': optimizer,
                        'activation': activation,
                        'neurons': neurons
                    }
                    
                    # Store the mean AUC and parameters
                    all_results.append((mean_auc, params))
                    
                    print(f"Params: {params}, Mean AUC: {mean_auc:.4f}")

# Sort all results by AUC (descending)
all_results.sort(reverse=True)

# Get top 3 models
top_3_models = all_results[:3]

print("\nTop 3 models based on cross-validation AUC:")
for i, (auc, params) in enumerate(top_3_models, 1):
    print(f"{i}. AUC: {auc:.4f} with parameters: {params}")

# Now train these top 3 models on the full training set and evaluate on test set
print("\nTraining top 3 models on full training set and evaluating on test set:")

test_results = []
for i, (cv_auc, params) in enumerate(top_3_models, 1):
    print(f"\nTraining model {i} with parameters: {params}")
    
    # Create and train model with best parameters on full training set
    model = create_model(
        batch_size=params['batch_size'],
        epochs=params['epochs'],
        optimizer=params['optimizer'],
        activation=params['activation'],
        neurons=params['neurons']
    )
    
    # Train on full training set
    model.fit(
        X_train_transformed, y_train,
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        verbose=1
    )
    
    # Evaluate on test set using AUC
    y_test_pred = model.predict(X_test_transformed, verbose=0)
    test_auc = roc_auc_score(y_test, y_test_pred)
    test_results.append((test_auc, params))
    
    print(f"Model {i} - Test AUC: {test_auc:.4f}")

# Final ranking of models based on test set AUC
print("\nFinal ranking of models based on test set AUC:")
test_results.sort(reverse=True)
for i, (auc, params) in enumerate(test_results, 1):
    print(f"{i}. Test AUC: {auc:.4f} with parameters: {params}")
