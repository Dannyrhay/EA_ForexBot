# strategies/ml_model.py
# ML Validator with RandomForest and XGBoost support

import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import os
from datetime import datetime
import warnings

logger = logging.getLogger(__name__)

class MLValidator:
    """
    Machine Learning Validator for trade signal validation.
    Supports both RandomForest and XGBoost models.
    Model type is configured in config.json via 'ml_model_type'.
    """

    def __init__(self, config):
        self.config = config
        self.predictors = {}  # {"{symbol}_{direction}": model}
        self.model_dir = "models"
        self.features_to_remove_indices = config.get('features_to_remove_indices', [])

        # Get model type from config (default to XGBoost)
        self.model_type = config.get('ml_model_type', 'XGBoost').lower()

        # Get hyperparameters from config
        if self.model_type == 'xgboost':
            self.params = config.get('xgboost_params', {})
        else:  # randomforest
            self.params = config.get('random_forest_params', {})

        # Create models directory if it doesn't exist
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            logger.info(f"Created directory: {self.model_dir}")

        logger.info(f"MLValidator initialized with model type: {self.model_type.upper()}")

        # Try to load existing models
        self.load_existing_models()

    def load_existing_models(self):
        """Load previously trained models from disk"""
        if not os.path.exists(self.model_dir):
            return

        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.joblib')]
        for model_file in model_files:
            try:
                model_path = os.path.join(self.model_dir, model_file)
                
                # Suppress XGBoost UserWarning about model serialization
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="pickle")
                    model = joblib.load(model_path)
                
                # Extract key from filename (e.g., "BTCUSDm_buy.joblib" -> "BTCUSDm_buy")
                key = model_file.replace('.joblib', '')
                self.predictors[key] = model
                logger.info(f"Loaded existing model: {key}")
                
                # Re-save the model to update serialization format if needed
                self._save_model(key, model)
                
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {e}")

    def create_predictors_for_all_symbols(self):
        """Create ML predictors for all configured symbols"""
        symbols = self.config.get('symbols', [])
        logger.info(f"Prepared to train {self.model_type.upper()} models for {len(symbols)} symbols")

    def is_fitted(self, symbol, direction):
        """Check if a model is fitted for the given symbol and direction"""
        key = f"{symbol}_{direction}"
        return key in self.predictors and self.predictors[key] is not None

    def _create_model(self):
        """
        Create a new ML model based on configured type

        Returns:
            Initialized model (RandomForest or XGBoost)
        """
        if self.model_type == 'xgboost':
            # XGBoost parameters (use first value from lists if provided)
            n_estimators = self.params.get('n_estimators', [100])[0] if isinstance(self.params.get('n_estimators', 100), list) else self.params.get('n_estimators', 100)
            max_depth = self.params.get('max_depth', [5])[0] if isinstance(self.params.get('max_depth', 5), list) else self.params.get('max_depth', 5)
            learning_rate = self.params.get('learning_rate', [0.1])[0] if isinstance(self.params.get('learning_rate', 0.1), list) else self.params.get('learning_rate', 0.1)
            subsample = self.params.get('subsample', [0.8])[0] if isinstance(self.params.get('subsample', 0.8), list) else self.params.get('subsample', 0.8)

            model = xgb.XGBClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth),
                learning_rate=float(learning_rate),
                subsample=float(subsample),
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                random_state=42,
                n_jobs=-1
            )
            logger.debug(f"Created XGBoost model: n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}, subsample={subsample}")

        else:  # RandomForest
            # RandomForest parameters (use first value from lists if provided)
            n_estimators = self.params.get('n_estimators', [100])[0] if isinstance(self.params.get('n_estimators', 100), list) else self.params.get('n_estimators', 100)
            max_depth = self.params.get('max_depth', [10])[0] if isinstance(self.params.get('max_depth', 10), list) else self.params.get('max_depth', 10)
            min_samples_split = self.params.get('min_samples_split', [20])[0] if isinstance(self.params.get('min_samples_split', 20), list) else self.params.get('min_samples_split', 20)

            model = RandomForestClassifier(
                n_estimators=int(n_estimators),
                max_depth=int(max_depth) if max_depth else None,
                min_samples_split=int(min_samples_split),
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            logger.debug(f"Created RandomForest model: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")

        return model

    def fit(self, symbol, X, y, direction):
        """
        Train the ML model (RandomForest or XGBoost based on config)

        Args:
            symbol: Trading symbol
            X: Feature matrix (numpy array or DataFrame)
            y: Target labels (numpy array)
            direction: 'buy' or 'sell'
        """
        key = f"{symbol}_{direction}"

        try:
            # Convert to numpy if needed
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(y, pd.Series):
                y = y.values

            # Remove NaN/inf values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
            X_clean = X[valid_mask]
            y_clean = y[valid_mask]

            min_samples = self.config.get('ml_min_samples_for_fit', 100)
            if len(X_clean) < min_samples:
                logger.warning(f"Insufficient clean data for training {key}: {len(X_clean)} samples (min: {min_samples})")
                return

            # Split data for validation
            test_size = 0.2
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=test_size, random_state=42, stratify=y_clean
                )
            except ValueError:
                # Stratify failed (probably not enough samples in one class)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=test_size, random_state=42
                )

            logger.info(f"Training {self.model_type.upper()} for {key} with {len(X_train)} training samples...")

            # Create model based on type
            model = self._create_model()

            # Train the model
            model.fit(X_train, y_train)

            # Evaluate on test set
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logger.info(f"Model {key} ({self.model_type.upper()}) trained successfully. Accuracy: {accuracy:.3f}")
            logger.debug(f"Classification report for {key}:\n{classification_report(y_test, y_pred)}")

            # Log feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_features = np.argsort(importances)[-5:][::-1]  # Top 5
                logger.debug(f"Top 5 feature indices for {key}: {top_features.tolist()}")

            # Store model
            self.predictors[key] = model

            # Save model to disk
            self._save_model(key, model)

        except Exception as e:
            logger.error(f"Error training model for {key}: {e}", exc_info=True)

    def _save_model(self, key, model):
        """Save trained model to disk"""
        try:
            model_path = os.path.join(self.model_dir, f"{key}.joblib")
            joblib.dump(model, model_path)
            logger.info(f"Model saved: {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {key}: {e}")

    def predict_proba(self, symbol, X, direction):
        """
        Predict probabilities for the given features

        Args:
            symbol: Trading symbol
            X: Feature matrix (can be single sample or batch)
            direction: 'buy' or 'sell'

        Returns:
            Probability matrix: [[prob_class_0, prob_class_1], ...]
        """
        key = f"{symbol}_{direction}"

        # If model not fitted, return neutral probabilities
        if not self.is_fitted(symbol, direction):
            logger.warning(f"Model {key} not fitted, returning neutral probabilities")
            if isinstance(X, pd.DataFrame):
                n = len(X)
            elif isinstance(X, (list, tuple)):
                n = len(X)
            elif hasattr(X, 'shape'):
                n = X.shape[0] if len(X.shape) > 1 else 1
            else:
                n = 1
            return np.array([[0.5, 0.5]] * n)

        try:
            model = self.predictors[key]

            # Convert to numpy if needed
            if isinstance(X, list):
                X = np.array(X)
            if isinstance(X, pd.DataFrame):
                X = X.values

            # Ensure 2D array
            if len(X.shape) == 1:
                X = X.reshape(1, -1)

            # Replace NaN/inf with 0
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Get predictions
            proba = model.predict_proba(X_clean)

            return proba

        except Exception as e:
            logger.error(f"Error predicting with model {key}: {e}", exc_info=True)
            # Return neutral on error
            n = X.shape[0] if len(X.shape) > 1 else 1
            return np.array([[0.5, 0.5]] * n)

    def _get_filtered_X(self, X):
        """Filter features based on features_to_remove_indices"""
        if not self.features_to_remove_indices:
            return X

        try:
            # Create mask of features to keep
            if isinstance(X, pd.DataFrame):
                all_indices = set(range(len(X.columns)))
            elif isinstance(X, np.ndarray):
                all_indices = set(range(X.shape[1]))
            else:
                return X

            indices_to_keep = sorted(all_indices - set(self.features_to_remove_indices))

            if isinstance(X, pd.DataFrame):
                return X.iloc[:, indices_to_keep]
            else:
                return X[:, indices_to_keep]
        except Exception as e:
            logger.error(f"Error filtering features: {e}")
            return X

    def get_model_info(self):
        """Get information about trained models"""
        info = {
            'model_type': self.model_type.upper(),
            'models': {}
        }

        for key, model in self.predictors.items():
            if model is not None:
                model_info = {
                    'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown',
                }

                if self.model_type == 'xgboost':
                    model_info.update({
                        'n_estimators': model.n_estimators if hasattr(model, 'n_estimators') else 'unknown',
                        'max_depth': model.max_depth if hasattr(model, 'max_depth') else 'unknown',
                        'learning_rate': model.learning_rate if hasattr(model, 'learning_rate') else 'unknown',
                    })
                else:  # RandomForest
                    model_info.update({
                        'n_estimators': model.n_estimators if hasattr(model, 'n_estimators') else 'unknown',
                        'max_depth': model.max_depth if hasattr(model, 'max_depth') else 'unknown',
                    })

                info['models'][key] = model_info

        return info
