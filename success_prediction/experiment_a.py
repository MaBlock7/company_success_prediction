"""
Experiment A: Performance difference between different model types on baseline features (replication experiment)
"""
import warnings
import pandas as pd
import optuna
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from success_prediction.modelling.config import (
    ALL_BINARY_FEATURE_COLS, ALL_CATEGORICAL_FEATURE_COLS, ALL_CONTINUOUS_FEATURE_COLS,
    FOUNDING_WEBSITE_FEATURE_COLS, CURRENT_WEBSITE_FEATURE_COLS, TARGET_COLS
)
from modelling.trainer import ModelEvaluation
from modelling.logreg_config import (
    LOGREG_BINARY_FEATURES, LOGREG_CONTINUOUS_FEATURES,
    LOGREG_HIGH_CAT_FEATURES, LOGREG_LOW_CAT_FEATURES
)
from modelling.rf_config import (
    RF_BINARY_FEATURES, RF_CONTINUOUS_FEATURES,
    RF_HIGH_CAT_FEATURES, RF_LOW_CAT_FEATURES
)
from modelling.xgb_config import (
    XGB_BINARY_FEATURES, XGB_CONTINUOUS_FEATURES,
    XGB_HIGH_CAT_FEATURES, XGB_LOW_CAT_FEATURES
)
from success_prediction.config import PROCESSED_DATA_DIR, MODELS_DIR

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._encoders")
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

optuna.logging.set_verbosity(optuna.logging.ERROR)


# Global params
RANDOM_STATE = 42
WEBSITE_FEATURE_COLS = FOUNDING_WEBSITE_FEATURE_COLS + CURRENT_WEBSITE_FEATURE_COLS


# Config for replication study using only baseline features
MODEL_SPECS = {
    'vanilla_logreg': {
        'model': LogisticRegression,
        'preprocessor_steps': [
            ('categorical_low_card', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
            ('continuous', StandardScaler()),
            ('binary', 'passthrough'),
        ],
        'features': {
            'binary': [f for f in LOGREG_BINARY_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'categorical_low_card': [f for f in LOGREG_LOW_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'categorical_high_card': [f for f in LOGREG_HIGH_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS],  # empty
            'continuous': [f for f in LOGREG_CONTINUOUS_FEATURES if f not in WEBSITE_FEATURE_COLS],
        },
        'fit_params': {
            'random_state': RANDOM_STATE,
            'n_jobs': 1,  # Avoid thread thrashing, so model n_jobs should be set to 1 because Grid Search CV and Feature Selection is set to -1
            'max_iter': 10_000,
            'solver': 'saga',  # Fixed for computational efficiency
        },
        'param_grid': {},  # No hyperparams for vanilla LogReg
        'search_type': None,
        'account_for_class_weights': True
    },
    'logreg': {
        'model': LogisticRegression,
        'preprocessor_steps': [
            ('categorical_low_card', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
            ('continuous', StandardScaler()),
            ('binary', 'passthrough'),
        ],
        'features': {
            'binary': [f for f in LOGREG_BINARY_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'categorical_low_card': [f for f in LOGREG_LOW_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'categorical_high_card': [f for f in LOGREG_HIGH_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS],  # empty
            'continuous': [f for f in LOGREG_CONTINUOUS_FEATURES if f not in WEBSITE_FEATURE_COLS],
        },
        'fit_params': {
            'random_state': RANDOM_STATE,
            'n_jobs': 1,  # Avoid thread thrashing, so model n_jobs should be set to 1 because Grid Search CV and Feature Selection is set to -1
            'max_iter': 10_000,
            'solver': 'saga',  # Fixed for computational efficiency
        },
        'param_grid': {
            'penalty': ['l1', 'l2'],  # Test Lasso and Ridge regularization
            'C': [0.01, 0.1, 1, 10, 100],
        },
        'search_type': 'grid',
        'account_for_class_weights': True
    },
    'rf': {
        'model': RandomForestClassifier,
        'preprocessor_steps': [
            ('categorical_low_card', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ('categorical_high_card', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ('continuous', 'passthrough'),
            ('binary', 'passthrough'),
        ],
        'features': {
            'binary': [f for f in RF_BINARY_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'categorical_low_card': [f for f in RF_LOW_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'categorical_high_card': [f for f in RF_HIGH_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'continuous': [f for f in RF_CONTINUOUS_FEATURES if f not in WEBSITE_FEATURE_COLS],
        },
        'fit_params': {
            'random_state': RANDOM_STATE,
            'n_jobs': 1,
        },
        'param_grid': {
            'n_estimators': IntDistribution(100, 400, step=50),
            'max_depth': CategoricalDistribution([None, 10, 20, 30]),
            'min_samples_split': IntDistribution(2, 20),
            'min_samples_leaf': IntDistribution(1, 10),
            'max_features': CategoricalDistribution(['sqrt', 'log2', 0.5]),
        },
        'search_type': 'optuna',
        'account_for_class_weights': True
    },
    'xgb': {
        'model': XGBClassifier,
        'preprocessor_steps': [
            ('categorical_low_card', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
            ('categorical_high_card', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ('continuous', 'passthrough'),
            ('binary', 'passthrough'),
        ],
        'features': {
            'binary': [f for f in XGB_BINARY_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'categorical_low_card': [f for f in XGB_LOW_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'categorical_high_card': [f for f in XGB_HIGH_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS],
            'continuous': [f for f in XGB_CONTINUOUS_FEATURES if f not in WEBSITE_FEATURE_COLS],
        },
        'fit_params': {
            'random_state': RANDOM_STATE,
            'n_jobs': 1,
            'objective': 'binary:logistic',
            'verbosity': 0,
            'booster': 'gbtree',
            'tree_method': 'hist',
            'use_label_encoder': False,
            'eval_metric': 'aucpr',
        },
        'param_grid': {
            'max_depth': IntDistribution(3, 10),
            'min_child_weight': IntDistribution(1, 10),
            'gamma': FloatDistribution(0, 5.0),
            'subsample': FloatDistribution(0.5, 1.0),
            'colsample_bytree': FloatDistribution(0.5, 1.0),
            'learning_rate': FloatDistribution(0.005, 0.1, log=True),
            'n_estimators': IntDistribution(100, 400, step=50),
            'reg_alpha': FloatDistribution(0, 5.0),  # L1 regularization
            'reg_lambda': FloatDistribution(1.0, 10.0),  # L2 regularization
            'max_delta_step': IntDistribution(0, 10),
        },
        'search_type': 'optuna',
        'account_for_class_weights': True
    }
}


def run_experiment():
    # Load data sample
    company_sample = pd.read_csv(PROCESSED_DATA_DIR / 'company_sample' / '2020_sample_encoded_features.csv')

    # Drop the row with missing firm name length
    company_sample = company_sample[company_sample['firm_name_length'].notna()]

    # Fill missing population with 0
    company_sample['population'] = company_sample['population'].fillna(0)

    # Keep the following collumns as missings since they are in locations not belonging to any canton, missing is correct in this case
    for col in ['district_id', 'canton_id', 'urban_rural', 'typology_9c', 'typology_25c']:
        company_sample[col] = company_sample[col].fillna(-1)

    for col in ALL_BINARY_FEATURE_COLS + TARGET_COLS:
        if col in company_sample.columns:
            company_sample[col] = company_sample[col].astype('int8')

    for col in ALL_CONTINUOUS_FEATURE_COLS:
        if col in company_sample.columns:
            company_sample[col] = company_sample[col].astype('float32')

    for col in ALL_CATEGORICAL_FEATURE_COLS:
        if col in company_sample.columns:
            company_sample[col] = company_sample[col].astype('category')

    all_feature_df = company_sample[TARGET_COLS + ALL_BINARY_FEATURE_COLS + ALL_CATEGORICAL_FEATURE_COLS + ALL_CONTINUOUS_FEATURE_COLS]

    # 1. Load data for experiment A
    base_df = all_feature_df[[col for col in all_feature_df.columns if col not in WEBSITE_FEATURE_COLS]].copy()

    # 2. Initialize model evaluation with targets and model specs
    meval = ModelEvaluation(TARGET_COLS, MODEL_SPECS, random_state=RANDOM_STATE)
    meval.load_data(base_df)

    # 3. Training procedure on all features for the baseline reproduction
    out_folder = MODELS_DIR / 'experiment_A'
    meval.nested_cv_with_hyperparam_search(out_folder=out_folder)


if __name__ == '__main__':
    run_experiment()
