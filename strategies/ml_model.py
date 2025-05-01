from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
import numpy as np
from utils.logging import setup_logging

logger = setup_logging()

class MLValidator:
    """ML Validation class using predictive models"""
    def __init__(self, config):
        """Initialize MLValidator with configuration."""
        self.config = config
        self.predictors = self.create_predictors()
        self.training_data = []

    def create_predictors(self):
        """Create a pipeline of StandardScaler and RandomForestClassifier with GridSearchCV for each symbol."""
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5]
        }
        try:
            predictors = {}
            for symbol in self.config['symbols']:
                base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=3,
                    n_jobs=1,  # Changed to 1 to disable parallel processing and avoid pickling issues
                    scoring='accuracy',
                    error_score='raise'
                )
                pipeline = make_pipeline(StandardScaler(), grid_search)
                predictors[symbol] = pipeline
                logger.debug(f"Created predictor pipeline for {symbol}")
            return predictors
        except Exception as e:
            logger.error(f"Error creating predictors: {e}")
            raise

    def fit(self, symbol, X, y):
        """Fit the model for a given symbol with training data."""
        try:
            if len(X) < 10 or len(np.unique(y)) < 2:
                raise ValueError(f"Insufficient data or classes for {symbol}: {len(X)} samples, {len(np.unique(y))} classes")
            self.predictors[symbol].fit(X, y)
            grid_search = self.predictors[symbol].named_steps['gridsearchcv']
            if hasattr(grid_search, 'best_params_'):
                logger.info(f"Best parameters for {symbol}: {grid_search.best_params_}")
            else:
                logger.warning(f"No best parameters found for {symbol}")
        except Exception as e:
            logger.error(f"Error fitting model for {symbol}: {e}")
            raise

    def predict_proba(self, symbol, X):
        """Predict probability of positive outcome for a given symbol."""
        try:
            check_is_fitted(self.predictors[symbol].named_steps['gridsearchcv'])
            return self.predictors[symbol].predict_proba(X)
        except Exception as e:
            logger.error(f"Error predicting for {symbol}: {e}")
            return np.array([[0.5, 0.5]] * len(X))