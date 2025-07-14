from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.validation import check_is_fitted, NotFittedError
from sklearn.pipeline import make_pipeline
import numpy as np
from utils.logging import setup_logging

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import make_pipeline as make_pipeline_imblearn
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = setup_logging()

class MLValidator:
    """
    ML Validation class using predictive models.
    Can be configured to use either RandomForest or XGBoost.
    Includes optional SMOTE for handling imbalanced datasets.
    """
    def __init__(self, config):
        self.config = config
        self.predictors = {'buy': {}, 'sell': {}}
        self.features_to_remove_indices = sorted(self.config.get('features_to_remove_indices', []), reverse=True)
        if self.features_to_remove_indices:
            logger.info(f"Configured to remove features at indices: {self.features_to_remove_indices}")

        self.create_predictors_for_all_symbols()

    def create_predictors_for_all_symbols(self):
        """Create predictor pipelines for each symbol and direction."""
        model_type = self.config.get('ml_model_type', 'RandomForest').lower()
        use_smote = self.config.get('ml_use_smote', True)

        if use_smote and not IMBLEARN_AVAILABLE:
            logger.warning("Configuration specifies 'ml_use_smote: true', but imblearn library is not installed.")
            logger.warning("Please install it ('pip install imbalanced-learn') or disable SMOTE. Falling back to no SMOTE.")
            use_smote = False

        pipeline_builder = make_pipeline_imblearn if use_smote and IMBLEARN_AVAILABLE else make_pipeline

        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            logger.error("Configuration specifies 'XGBoost', but the xgboost library is not installed. Falling back to RandomForest.")
            model_type = 'randomforest'

        logger.info(f"Initializing ML predictors using: Model={model_type.capitalize()}, SMOTE={'Enabled' if use_smote else 'Disabled'}")

        if model_type == 'randomforest':
            base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            param_prefix = 'randomforestclassifier__'
            param_grid_config = self.config.get('random_forest_params', {})
        elif model_type == 'xgboost':
            # Note: We do NOT set scale_pos_weight here, as we will use sample_weight in fit()
            base_model = XGBClassifier(random_state=42, eval_metric='logloss')
            param_prefix = 'xgbclassifier__'
            param_grid_config = self.config.get('xgboost_params', {})
        else:
            logger.error(f"Unsupported ml_model_type '{model_type}'. Cannot create predictors.")
            return

        param_grid = {param_prefix + k: v for k, v in param_grid_config.items()}

        try:
            for direction in ['buy', 'sell']:
                for symbol in self.config.get('symbols', []):
                    pipeline_steps = [StandardScaler()]
                    if use_smote and IMBLEARN_AVAILABLE:
                        pipeline_steps.append(SMOTE(random_state=42))
                    pipeline_steps.append(base_model)

                    pipeline = pipeline_builder(*pipeline_steps)

                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        cv=TimeSeriesSplit(n_splits=self.config.get('ml_timeseries_splits', 5)),
                        n_jobs=self.config.get('ml_gridsearch_cv_jobs', -1),
                        scoring=self.config.get('ml_scoring_metric', 'f1_weighted'),
                        error_score='raise'
                    )
                    self.predictors[direction][symbol] = grid_search
        except Exception as e:
            logger.error(f"Error creating predictors: {e}", exc_info=True)

    def _get_filtered_X(self, X):
        """Removes specified features from the input data X."""
        if not self.features_to_remove_indices:
            return X
        X_np = np.asarray(X)
        for feature_index in self.features_to_remove_indices:
            if feature_index < X_np.shape[1]:
                X_np = np.delete(X_np, feature_index, axis=1)
        return X_np

    def fit(self, symbol, X, y, direction):
        """Fit the model for a given symbol and direction with training data."""
        if direction not in self.predictors or symbol not in self.predictors[direction]:
            logger.error(f"No predictor found for {direction} model of symbol {symbol}. Cannot fit.")
            return

        try:
            min_samples_for_fit = self.config.get('ml_min_samples_for_fit', 50)
            X_filtered = self._get_filtered_X(X)

            if len(X_filtered) < min_samples_for_fit or len(np.unique(y)) < 2:
                logger.warning(f"Insufficient data or classes for {direction} model of {symbol}: {len(X_filtered)} samples, {len(np.unique(y))} classes. Skipping fit.")
                return

            model_type = self.config.get('ml_model_type', 'RandomForest').lower()
            use_smote = self.config.get('ml_use_smote', True)

            fit_params = {}
            # --- FIX: Use sample_weight for XGBoost when not using SMOTE ---
            if model_type == 'xgboost' and not (use_smote and IMBLEARN_AVAILABLE):
                # Calculate the ratio for the minority class
                scale_pos_weight = np.sum(y == 0) / np.sum(y == 1) if np.sum(y == 1) > 0 else 1
                # Create a sample_weight array
                sample_weights = np.where(np.asarray(y) == 1, scale_pos_weight, 1)
                # This is the correct way to pass fit parameters to a pipeline step in GridSearchCV
                fit_params = {'xgbclassifier__sample_weight': sample_weights}
                logger.debug(f"Using sample_weight for {symbol} {direction} with scale_pos_weight factor: {scale_pos_weight:.2f}")

            self.predictors[direction][symbol].fit(X_filtered, y, **fit_params)

            grid_search = self.predictors[direction][symbol]
            if hasattr(grid_search, 'best_estimator_'):
                logger.info(f"Best parameters for {symbol} ({direction.upper()}): {grid_search.best_params_}")
                best_estimator = grid_search.best_estimator_
                model_step_name = ''
                if 'randomforestclassifier' in best_estimator.named_steps:
                    model_step_name = 'randomforestclassifier'
                elif 'xgbclassifier' in best_estimator.named_steps:
                    model_step_name = 'xgbclassifier'

                if model_step_name and hasattr(best_estimator.named_steps[model_step_name], 'feature_importances_'):
                    importances = best_estimator.named_steps[model_step_name].feature_importances_
                    original_indices = np.arange(np.asarray(X).shape[1])
                    # Ensure features_to_remove_indices are integers
                    valid_indices_to_remove = [int(i) for i in self.features_to_remove_indices if isinstance(i, (int, float))]
                    kept_indices = np.delete(original_indices, valid_indices_to_remove)
                    feature_importance_map = {f"feature_{kept_indices[i]}": imp for i, imp in enumerate(importances)}
                    sorted_importance = sorted(feature_importance_map.items(), key=lambda item: item[1], reverse=True)
                    logger.info(f"Feature Importances for {symbol} ({direction.upper()}) (Top 10): {sorted_importance[:10]}")
        except Exception as e:
            logger.error(f"Error fitting {direction} model for {symbol}: {e}", exc_info=True)

    def predict_proba(self, symbol, X, direction):
        """Predict probability of positive outcome."""
        if direction not in self.predictors or symbol not in self.predictors[direction]:
            return np.array([[0.5, 0.5]] * len(X))
        try:
            X_filtered = self._get_filtered_X(X)
            check_is_fitted(self.predictors[direction][symbol])
            return self.predictors[direction][symbol].predict_proba(X_filtered)
        except NotFittedError:
            logger.warning(f"{direction} model for {symbol} is not fitted. Returning default probabilities.")
            return np.array([[0.5, 0.5]] * len(X))
        except Exception as e:
            logger.error(f"Error predicting for {direction} model of {symbol}: {e}", exc_info=True)
            return np.array([[0.5, 0.5]] * len(X))

    def is_fitted(self, symbol, direction):
        """Checks if a specific model is fitted."""
        if direction not in self.predictors or symbol not in self.predictors[direction]:
            return False
        try:
            check_is_fitted(self.predictors[direction][symbol])
            return True
        except NotFittedError:
            return False
        except Exception:
            return False

    def any_models_fitted(self):
        """Check if ANY models are fitted."""
        for direction in self.predictors:
            for symbol in self.predictors[direction]:
                if self.is_fitted(symbol, direction):
                    return True
        return False
