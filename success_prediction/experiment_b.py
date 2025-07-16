"""
Experiment B: Regression Analysis of Embedding Scores
"""
import pandas as pd
from pocketknife.database import connect_database, read_from_database
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from modelling.config import (
    ALL_BINARY_FEATURE_COLS, ALL_CATEGORICAL_FEATURE_COLS, ALL_CONTINUOUS_FEATURE_COLS,
    FOUNDING_WEBSITE_FEATURE_COLS, CURRENT_WEBSITE_FEATURE_COLS, TARGET_COLS
)
from modelling.regression_analysis import CoefficientAnalyser
from modelling.logreg_config import (
    LOGREG_BINARY_FEATURES, LOGREG_CONTINUOUS_FEATURES,
)
from config import PROCESSED_DATA_DIR, MODELS_DIR


# Global params
RANDOM_STATE = 42
WEBSITE_FEATURE_COLS = FOUNDING_WEBSITE_FEATURE_COLS + CURRENT_WEBSITE_FEATURE_COLS

CAT_CONTROLS = ['founding_legal_form', 'division_1_label', 'typology_9c', 'canton_id', 'bps_length_quantiles_5', 'founding_year']
OTHER_CONTROLS = [c for c in LOGREG_BINARY_FEATURES + LOGREG_CONTINUOUS_FEATURES if c not in CURRENT_WEBSITE_FEATURE_COLS + FOUNDING_WEBSITE_FEATURE_COLS]

EXPERIMENT_SETUP = [
    {
        'title': 'No FEs',
        'targets': [
            'target_inno_subsidy',
            'target_non_gov_investment',
            'target_acquisition',
            'target_inv_exit'
        ],
        'score_cols': [
            'founding_doc2vec_diff',
            ['founding_core_diff_pca', 'founding_pr_sdg_similarity'],
            ['founding_core_diff_w_pca', 'founding_pr_w_sdg_similarity'],
            ['founding_core_diff_w_red_pca', 'founding_pr_w_red_sdg_similarity'],
            'current_doc2vec_diff',
            ['current_core_diff_pca', 'current_pr_sdg_similarity'],
            ['current_core_diff_w_pca', 'current_pr_w_sdg_similarity'],
            ['current_core_diff_w_red_pca', 'current_pr_w_red_sdg_similarity']
        ],
    },
    {
        'title': 'Year FEs',
        'targets': [
            ('target_inno_subsidy', 'target_non_gov_investment', 'target_acquisition'),
            ('target_inno_subsidy', 'target_non_gov_investment'),
            'target_inno_subsidy',
            'target_non_gov_investment',
            'target_acquisition'
        ],
        'score_cols': [
            'founding_doc2vec_diff',
            ['founding_core_diff_w_red_pca', 'founding_pr_w_red_sdg_similarity'],
            'current_doc2vec_diff',
            ['current_core_diff_w_red_pca', 'current_pr_w_red_sdg_similarity']
        ],
        'cat_controls': ['founding_year'],
    },
    {
        'title': 'Year + Industry FEs',
        'targets': [
            ('target_inno_subsidy', 'target_non_gov_investment', 'target_acquisition'),
            ('target_inno_subsidy', 'target_non_gov_investment'),
            'target_inno_subsidy',
            'target_non_gov_investment',
            'target_acquisition'
        ],
        'score_cols': [
            'founding_doc2vec_diff',
            ['founding_core_diff_w_red_pca', 'founding_pr_w_red_sdg_similarity'],
            'current_doc2vec_diff',
            ['current_core_diff_w_red_pca', 'current_pr_w_red_sdg_similarity']
        ],
        'cat_controls': ['founding_year', 'division_1_label'],
    },
    {
        'title': 'Year + Industry + Canton FEs',
        'targets': [
            ('target_inno_subsidy', 'target_non_gov_investment', 'target_acquisition'),
            ('target_inno_subsidy', 'target_non_gov_investment'),
            'target_inno_subsidy',
            'target_non_gov_investment',
            'target_acquisition'
        ],
        'score_cols': [
            'founding_doc2vec_diff',
            ['founding_core_diff_w_red_pca', 'founding_pr_w_red_sdg_similarity'],
            'current_doc2vec_diff',
            ['current_core_diff_w_red_pca', 'current_pr_w_red_sdg_similarity']
        ],
        'cat_controls': ['founding_year', 'division_1_label', 'canton_id'],
    },
    {
        'title': 'Green - Year FEs',
        'targets': ['is_green'],
        'score_cols': [
            'founding_pr_sdg_similarity',
            'founding_pr_w_sdg_similarity',
            'founding_pr_w_red_sdg_similarity',
            'current_pr_sdg_similarity',
            'current_pr_w_sdg_similarity',
            'current_pr_w_red_sdg_similarity'
        ],
        'cat_controls': ['founding_year'],
    },
    {
        'title': 'Green - Year + Industry FEs',
        'targets': ['is_green'],
        'score_cols': [
            'founding_pr_sdg_similarity',
            'founding_pr_w_sdg_similarity',
            'founding_pr_w_red_sdg_similarity',
            'current_pr_sdg_similarity',
            'current_pr_w_sdg_similarity',
            'current_pr_w_red_sdg_similarity'
        ],
        'cat_controls': ['founding_year', 'division_1_label'],
    },
    {
        'title': 'Green - Year + Industry + Canton FEs',
        'targets': ['is_green'],
        'score_cols': [
            'founding_pr_sdg_similarity',
            'founding_pr_w_sdg_similarity',
            'founding_pr_w_red_sdg_similarity',
            'current_pr_sdg_similarity',
            'current_pr_w_sdg_similarity',
            'current_pr_w_red_sdg_similarity'
        ],
        'cat_controls': ['founding_year', 'division_1_label', 'canton_id'],
    },
]


def run_experiment():

    company_sample = pd.read_csv(PROCESSED_DATA_DIR / 'company_sample' / 'until_2020' / '2020_sample_encoded_features.csv')

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

    query_green = """
        SELECT * FROM zefix.green_binary WHERE is_green;
    """

    with connect_database() as con:
        df_green = read_from_database(connection=con, query=query_green)

    company_sample = company_sample[['ehraid', 'uid'] + TARGET_COLS + ALL_BINARY_FEATURE_COLS + ALL_CATEGORICAL_FEATURE_COLS + ALL_CONTINUOUS_FEATURE_COLS + ['founding_year']]
    company_sample = company_sample.merge(df_green, on='uid', how='left')
    company_sample['is_green'] = company_sample['is_green'].fillna(0).astype(int)

    company_sample.drop(columns=['ehraid', 'uid'], inplace=True)

    score_cols = [
        'founding_doc2vec_diff',
        'current_doc2vec_diff',
        'founding_pr_sdg_similarity',
        'founding_pr_w_sdg_similarity',
        'founding_pr_w_red_sdg_similarity',
        'founding_lp',
        'founding_lp_w',
        'founding_lp_w_red',
        'founding_vp',
        'founding_vp_w',
        'founding_vp_w_red',
        'current_pr_sdg_similarity',
        'current_pr_w_sdg_similarity',
        'current_pr_w_red_sdg_similarity',
        'current_lp',
        'current_lp_w',
        'current_lp_w_red',
        'current_vp',
        'current_vp_w',
        'current_vp_w_red',
        'founding_core_diff_pca',
        'current_core_diff_pca',
        'founding_core_diff_w_pca',
        'current_core_diff_w_pca',
        'founding_core_diff_w_red_pca',
        'current_core_diff_w_red_pca'
    ]

    pca = PCA(n_components=1)
    core_diff_cols = [
        (['founding_lp', 'founding_vp'], 'founding_core_diff_pca'),
        (['founding_lp_w', 'founding_vp_w'], 'founding_core_diff_w_pca'),
        (['founding_lp_w_red', 'founding_vp_w_red'], 'founding_core_diff_w_red_pca'),
        (['current_lp', 'current_vp'], 'current_core_diff_pca'),
        (['current_lp_w', 'current_vp_w'], 'current_core_diff_w_pca'),
        (['current_lp_w_red', 'current_vp_w_red'], 'current_core_diff_w_red_pca')
    ]
    for source_cols, target_col in core_diff_cols:
        mask = company_sample[source_cols].notnull().all(axis=1)
        company_sample.loc[mask, target_col] = pca.fit_transform(
            company_sample.loc[mask, source_cols]
        )

    scaler = StandardScaler()
    company_sample[score_cols + LOGREG_CONTINUOUS_FEATURES] = scaler.fit_transform(company_sample[score_cols + LOGREG_CONTINUOUS_FEATURES])

    analyser = CoefficientAnalyser(company_sample, experiment_dir=MODELS_DIR / 'experiment_B')

    for experiment in EXPERIMENT_SETUP:
        analyser.estimate(
            targets=experiment.get('targets'),
            score_cols=experiment.get('score_cols'),
            cat_controls=experiment.get('cat_controls'),
            other_controls=experiment.get('other_controls'),
            cat_interaction_terms=experiment.get('cat_interaction_terms'),
            other_interaction_terms=experiment.get('other_interaction_terms'),
            subfolder=experiment.get('title'),
            save_full_summary=True
        )


if __name__ == '__main__':
    run_experiment()
