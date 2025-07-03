"""
Experiment C: Estimate predictive value of website features
"""
import ast
import warnings
import numpy as np
import pandas as pd

from scipy.stats import t, ttest_1samp

import optuna
from optuna.distributions import IntDistribution, FloatDistribution
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBClassifier
from success_prediction.modelling.config import (
    ALL_BINARY_FEATURE_COLS, ALL_CATEGORICAL_FEATURE_COLS, ALL_CONTINUOUS_FEATURE_COLS,
    FOUNDING_WEBSITE_FEATURE_COLS, CURRENT_WEBSITE_FEATURE_COLS, TARGET_COLS
)
from modelling.trainer import ModelEvaluation
from modelling.xgb_config import (
    XGB_BINARY_FEATURES, XGB_CONTINUOUS_FEATURES,
    XGB_HIGH_CAT_FEATURES, XGB_LOW_CAT_FEATURES
)
from success_prediction.config import RAW_DATA_DIR, MODELS_DIR

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.preprocessing._encoders")
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.ERROR)


# Global params
RANDOM_STATE = 42
WEBSITE_FEATURE_COLS = FOUNDING_WEBSITE_FEATURE_COLS + CURRENT_WEBSITE_FEATURE_COLS

FOUNDING_WEBSITE_STATS = [
    'founding_mean_text_len',
    'founding_n_internal_links_mean',
    'founding_n_external_links_mean',
    'founding_n_languages',
]

CURRENT_WEBSITE_STATS = [
    'current_mean_text_len',
    'current_n_internal_links_mean',
    'current_n_external_links_mean',
    'current_n_languages',
]

FEATURE_CONFIGS = {
    'founding_base': {
        'cont': [],
        'low_cat': []
    },
    'founding_doc2vec': {
        'cont': ['founding_doc2vec_diff'] + FOUNDING_WEBSITE_STATS,
        'low_cat': ['founding_dominant_language']
    },
    'founding_dim768': {
        'cont': ['founding_pr_sdg_similarity', 'founding_lp', 'founding_vp'] + FOUNDING_WEBSITE_STATS,
        'low_cat': ['founding_dominant_language']
    },
    'founding_dim768_w': {
        'cont': ['founding_pr_w_sdg_similarity', 'founding_lp_w', 'founding_vp_w'] + FOUNDING_WEBSITE_STATS,
        'low_cat': ['founding_dominant_language']
    },
    'founding_dim300_w': {
        'cont': ['founding_pr_w_red_sdg_similarity', 'founding_lp_w_red', 'founding_vp_w_red'] + FOUNDING_WEBSITE_STATS,
        'low_cat': ['founding_dominant_language']
    },
    'current_base': {
        'cont': [],
        'low_cat': []
    },
    'current_doc2vec': {
        'cont': ['current_doc2vec_diff'] + CURRENT_WEBSITE_STATS,
        'low_cat': ['current_dominant_language']
    },
    'current_dim768': {
        'cont': ['current_pr_sdg_similarity', 'current_lp', 'current_vp'] + CURRENT_WEBSITE_STATS,
        'low_cat': ['current_dominant_language']
    },
    'current_dim768_w': {
        'cont': ['current_pr_w_sdg_similarity', 'current_lp_w', 'current_vp_w'] + CURRENT_WEBSITE_STATS,
        'low_cat': ['current_dominant_language']
    },
    'current_dim300_w': {
        'cont': ['current_pr_w_red_sdg_similarity', 'current_lp_w_red', 'current_vp_w_red'] + CURRENT_WEBSITE_STATS,
        'low_cat': ['current_dominant_language']
    },
}


def get_per_fold_metric(df, target, metric_col):
    """Return list of per-fold metric values for given target."""
    values = df[df['target'] == target][metric_col].values
    return ast.literal_eval(values[0]) if len(values) > 0 else None


def create_individual_significance_report():
    results = []
    for kind in ['founding', 'current']:
        exp_B_base_df = pd.read_csv(MODELS_DIR / 'experiment_C' / f'{kind}_base' / 'cv_metrics_report.csv')
        for report_dir in [col for col in FEATURE_CONFIGS.keys() if kind in col]:
            for target in TARGET_COLS:
                comp_df = pd.read_csv(MODELS_DIR / 'experiment_C' / report_dir / 'cv_metrics_report.csv')
                for metric in ["all_roc_auc", "all_pr_auc"]:
                    # Get per-fold AP for website and base
                    web_ap = get_per_fold_metric(comp_df, target, metric)
                    base_ap = get_per_fold_metric(exp_B_base_df, target, metric)
                    web_ap, base_ap = np.array(web_ap), np.array(base_ap)

                    mean_web, mean_base = np.mean(web_ap), np.mean(base_ap)
                    std_web, std_base = np.std(web_ap, ddof=1), np.std(base_ap, ddof=1)
                    n_web, n_base = len(web_ap), len(base_ap)

                    # Welch's SE and df
                    se_diff = np.sqrt(std_web**2 / n_web + std_base**2 / n_base)
                    degrees_of_freedom = (std_web**2 / n_web + std_base**2 / n_base)**2 / ((std_web**2 / n_web)**2 / (n_web - 1) + (std_base**2 / n_base)**2 / (n_base - 1))

                    diff = mean_web - mean_base

                    t_stat = diff / se_diff if se_diff > 0 else 0
                    p_value = 2 * t.sf(np.abs(t_stat), degrees_of_freedom)

                    ci = {}
                    for alpha, label in zip([0.01, 0.05, 0.1], ['99', '95', '90']):
                        t_crit = t.ppf(1 - alpha / 2, degrees_of_freedom)
                        ci[f"ci_lower_{label}"] = diff - t_crit * se_diff
                        ci[f"ci_upper_{label}"] = diff + t_crit * se_diff

                    results.append({
                        'model': report_dir,
                        'metric': metric,
                        'metric_value': mean_web,
                        'target': target,
                        'mean_ap_website': mean_web,
                        'mean_ap_base': mean_base,
                        'p_value': p_value,
                        **ci,
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(MODELS_DIR / 'experiment_C' / 'individual_significance_report.csv', index=False)


def create_average_significance_report():
    results = []
    for kind in ['founding', 'current']:
        base_df = pd.read_csv(MODELS_DIR / 'experiment_C' / f'{kind}_base' / 'cv_metrics_report.csv')
        # For each target, for each metric, pool differences from all website models
        for target in TARGET_COLS:
            for metric in ["all_roc_auc", "all_pr_auc"]:
                all_diffs = []
                for report_dir in [col for col in FEATURE_CONFIGS.keys() if kind in col and col != f'{kind}_base']:
                    comp_df = pd.read_csv(MODELS_DIR / 'experiment_C' / report_dir / 'cv_metrics_report.csv')
                    web_scores = get_per_fold_metric(comp_df, target, metric)
                    base_scores = get_per_fold_metric(base_df, target, metric)
                    if web_scores is None or base_scores is None:
                        continue
                    diffs = np.array(web_scores) - np.array(base_scores)
                    all_diffs.extend(diffs)

                all_diffs = np.array(all_diffs)
                if len(all_diffs) == 0:
                    continue
                mean_diff = np.mean(all_diffs)
                mean_diff_pct = np.round(mean_diff / np.mean(base_scores) * 100, decimals=1)
                std_diff = np.std(all_diffs, ddof=1)
                n = len(all_diffs)
                se = std_diff / np.sqrt(n)

                # t-test and p-value
                t_stat, p_value = ttest_1samp(all_diffs, 0.0)

                # Confidence intervals
                ci_99 = t.ppf(0.995, n - 1) * se
                ci_95 = t.ppf(0.975, n - 1) * se
                ci_90 = t.ppf(0.95, n - 1) * se

                results.append({
                    'kind': kind,
                    'target': target,
                    'metric': metric,
                    'mean_improvement': mean_diff,
                    'mean_improvement_pct': mean_diff_pct,
                    'std': std_diff,
                    'n': n,
                    'p_value': p_value,
                    'ci_lower_99': mean_diff - ci_99,
                    'ci_upper_99': mean_diff + ci_99,
                    'ci_lower_95': mean_diff - ci_95,
                    'ci_upper_95': mean_diff + ci_95,
                    'ci_lower_90': mean_diff - ci_90,
                    'ci_upper_90': mean_diff + ci_90,
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(MODELS_DIR / 'experiment_C' / 'average_significance_report.csv', index=False)


def run_experiment():
    # Load data sample
    company_sample = pd.read_csv(RAW_DATA_DIR / 'company_sample' / 'until_2020' / '2020_sample_encoded_features.csv')

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

    # 1. Load data for experiment B
    for experiment_config, website_features in FEATURE_CONFIGS.items():
        print(f'START CONDUCTING EXPERIMENT B FOR: {experiment_config}')

        # Set model specs
        experiment_setup = {
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
                    'categorical_low_card': [f for f in XGB_LOW_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS] + website_features['low_cat'],
                    'categorical_high_card': [f for f in XGB_HIGH_CAT_FEATURES if f not in WEBSITE_FEATURE_COLS],
                    'continuous': [f for f in XGB_CONTINUOUS_FEATURES if f not in WEBSITE_FEATURE_COLS] + website_features['cont'],
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

        # 1. Load data for experiment A
        if 'current' in experiment_config:
            cols = [col for col in all_feature_df.columns if col not in FOUNDING_WEBSITE_FEATURE_COLS]
            website_df = all_feature_df[~all_feature_df['current_vp'].isna()][cols].copy()

        elif 'founding' in experiment_config:
            cols = [col for col in all_feature_df.columns if col not in CURRENT_WEBSITE_FEATURE_COLS]
            website_df = all_feature_df[~all_feature_df['founding_vp'].isna()][cols].copy()

        targets = TARGET_COLS

        # 2. Initialize model evaluation with targets and model specs
        meval = ModelEvaluation(targets, experiment_setup, random_state=RANDOM_STATE)
        meval.load_data(website_df)

        # 3. Evaluate with doc2vec scores
        out_folder = MODELS_DIR / 'experiment_C' / experiment_config
        meval.nested_cv_with_hyperparam_search(out_folder=out_folder, k_outer=10)

        # 4. Test significance
        create_individual_significance_report()
        create_average_significance_report()


if __name__ == '__main__':
    run_experiment()
