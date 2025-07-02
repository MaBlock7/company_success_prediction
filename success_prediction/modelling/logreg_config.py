"""
LogReg Features

For logistic regression, the features listed below are included.
- High-cardinality categorical features are removed and the remaining one-hot encoded.
- Continuous features are scaled using StandardScaler().
- Binary features are used as is.
"""

LOGREG_BINARY_FEATURES = [
    'firm_name_swiss_ref',
    'firm_name_holding_ref',
    'firm_name_geog_ref',
    'firm_name_founder_match',
    'firm_name_male_match',
    'firm_name_female_match',
    'bps_geographic_term',
    'bps_male_name',
    'bps_female_name',
]

# < 30 categories and not strongly correlated
LOGREG_LOW_CAT_FEATURES = [
    'founding_legal_form',
    'section_1_label',
    'typology_9c',
    'canton_id',
    'bps_length_quantiles_5',
    'founding_dominant_language',
    'current_dominant_language',
]

LOGREG_HIGH_CAT_FEATURES = []  # No features with high cardinality

LOGREG_CONTINUOUS_FEATURES = [
    'capital_chf',
    'firm_name_length',
    'population',
    'n_firms_within_10m',
    'n_firms_within_2.5km',
    'n_founders',
    'n_inscribed_firms',
    'n_distinct_nationalities',
    'pct_female_founders',
    'pct_foreign_founders',
    'pct_dr_titles',
    'pct_founders_same_residence',
    'pct_founders_with_prior_founding',
    'n_dissolved_firms',
    'n_existing_firms',
    'bps_mean_word_length',
    'bps_lix',
    'bps_min_word_freq_norm',
    'bps_max_word_freq_norm',
    'bps_freq_ratio_norm',
    'founding_mean_text_len',
    'founding_n_internal_links_mean',
    'founding_n_external_links_mean',
    'founding_n_languages',
    'founding_pr_sdg_similarity',
    'founding_pr_w_sdg_similarity',
    'founding_pr_w_red_sdg_similarity',
    'founding_doc2vec_diff',
    'founding_lp',
    'founding_lp_w',
    'founding_lp_w_red',
    'founding_vp',
    'founding_vp_w',
    'founding_vp_w_red',
    'current_mean_text_len',
    'current_n_internal_links_mean',
    'current_n_external_links_mean',
    'current_n_languages',
    'current_pr_sdg_similarity',
    'current_pr_w_sdg_similarity',
    'current_pr_w_red_sdg_similarity',
    'current_doc2vec_diff',
    'current_lp',
    'current_lp_w',
    'current_lp_w_red',
    'current_vp',
    'current_vp_w',
    'current_vp_w_red',
    'days_of_prior_observations',
    'prediction_1_score',
]
