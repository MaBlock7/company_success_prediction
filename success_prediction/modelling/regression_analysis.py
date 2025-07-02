from pathlib import Path
from typing import Any, Union, Sequence
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold


class CoefficientAnalyser:
    """Analyser for estimating regression coefficients and model performance with cross-validation."""
    def __init__(
        self,
        df: pd.DataFrame,
        experiment_dir: str,
        maxiter: int = 1500,
        random_state: int = 42
    ) -> None:
        """
        Args:
            df: Input DataFrame containing data for regression.
            experiment_dir: Directory path to save experiment results.
            maxiter: Maximum iterations for model fitting.
            random_state: Random seed for reproducibility.
        """
        self.df = df
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True, parents=True)
        self.maxiter = maxiter
        self.random_state = random_state

    @staticmethod
    def drop_perfect_separation(
        df: pd.DataFrame,
        target: str,
        col: str
    ) -> pd.DataFrame:
        """
        Removes categories in `col` where `target` is perfectly separated (only one class).

        Args:
            df: DataFrame to filter.
            target: Target column (binary).
            col: Categorical column to check for perfect separation.

        Returns:
            DataFrame with categories having both target classes retained.
        """
        keep = df.groupby(col, observed=False)[target].nunique()
        keep = keep[keep > 1].index  # only keep categories that have both 0 and 1
        return df[df[col].isin(keep)]

    @staticmethod
    def collapse_and_drop_sparse(
        df: pd.DataFrame,
        target: str,
        cat_controls: Sequence[str],
        min_count: int = 25
    ) -> pd.DataFrame:
        """
        Collapses rare categories to 'Other' and drops groups with no variation in target.

        Args:
            df: Input DataFrame.
            target: Target column.
            cat_controls: List of categorical controls.
            min_count: Minimum count to avoid collapsing into 'Other'.

        Returns:
            Filtered DataFrame with collapsed rare categories and no-variation groups dropped.
        """
        if not cat_controls:
            return df

        for col in cat_controls:
            df[col] = df[col].astype(str)
            vc = df[col].value_counts()
            rare = vc[vc < min_count].index
            df.loc[:, col] = df[col].replace(rare, 'Other')

        # Drop no-variation groups
        group_sizes = df.groupby(cat_controls, observed=False)[target].nunique()
        valid_groups = group_sizes[group_sizes > 1].index
        mask = df.set_index(cat_controls).index.isin(valid_groups)
        return df[mask].copy()

    def _estimate_performance(
        self,
        df: pd.DataFrame,
        formula: str,
        target: str,
        score_set: Sequence[str],
        cat_controls: Sequence[str],
        k_folds: int
    ) -> dict[str, Any]:
        """
        Runs k-fold cross-validation and estimates AUC/PR-AUC scores for a logistic regression model.

        Args:
            df: Input DataFrame (no missing values in relevant columns).
            formula: Patsy formula for statsmodels logit.
            target: Target variable name.
            score_set: List of score columns used as predictors.
            cat_controls: List of categorical controls.
            k_folds: Number of cross-validation folds.

        Returns:
            Dictionary with fold-level and mean scores for AUC and PR-AUC.
        """
        df = df.copy()
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)
        X, y = df.drop(columns=[target]), df[target]
        aucs, ap_aucs = [], []
        valid_folds = 0
        for train_idx, test_idx in skf.split(X, y):
            train_df, test_df = df.iloc[train_idx].copy(), df.iloc[test_idx].copy()
            """
            # Collapse and drop rare categories inside the train fold
            train_df = self.collapse_and_drop_sparse(train_df, target, cat_controls)
            print(len(train_df))
            # Keep only test samples where categories exist in train_df
            for col in cat_controls:
                allowed_cats = set(train_df[col].unique())
                test_df = test_df[test_df[col].isin(allowed_cats)]
            if len(train_df) < 25_000 or len(test_df) < 2500:
                continue  # skip if not enough data
            """
            try:
                model = smf.logit(formula=formula, data=train_df).fit(disp=0, cov_type='HC1', maxiter=self.maxiter)
            except Exception as e:
                print(f"Exception occured during model fitting: {e}")
                continue
            y_pred = model.predict(test_df)
            y_true = test_df[target]
            aucs.append(roc_auc_score(y_true, y_pred))
            ap_aucs.append(average_precision_score(y_true, y_pred))
            valid_folds += 1

        return {
            'target': target,
            'score': '+'.join(score_set),
            'valid_folds': valid_folds,
            'roc_aucs': aucs,
            'mean_roc_auc': np.mean(aucs) if aucs else np.nan,
            'std_roc_auc': np.std(aucs) if aucs else np.nan,
            'pr_aucs': ap_aucs,
            'mean_pr_auc': np.mean(ap_aucs) if ap_aucs else np.nan,
            'std_pr_auc': np.std(ap_aucs) if ap_aucs else np.nan,
        }

    def estimate(
        self,
        targets: list[Union[str, tuple[str, str]]],
        score_cols: list[Union[str, list[str]]],
        cat_controls: list[str],
        other_controls: list[str],
        cat_interaction_terms: list[tuple[str, str]],
        other_interaction_terms: list[tuple[str, str]],
        k_folds: int = 10,
        save_full_summary: bool = True,
        subfolder: str = 'reg_results'
    ) -> None:
        """
        Runs logistic regression (with categorical controls and interactions) and saves results.

        Args:
            targets: List of target column names (str or tuple of str for "or" targets).
            score_cols: List of column names or lists of column names (as predictors).
            cat_controls: List of categorical control variable names.
            other_controls: List of other control variable names (continuous/fixed effects).
            cat_interaction_terms: List of tuples (col1, col2) for categorical interactions.
            other_interaction_terms: List of tuples (col1, col2) for numeric interactions.
            k_folds: Number of cross-validation folds.
            save_full_summary: Whether to save statsmodels' full regression summary.
            subfolder: Subfolder name for saving results.

        Returns:
            None. Saves regression summaries and cross-validation scores to CSV.
        """
        summary_rows = []
        auc_rows = []
        out_folder = self.experiment_dir / subfolder
        summary_folder = out_folder / 'summaries'
        out_folder.mkdir(exist_ok=True, parents=True)
        summary_folder.mkdir(exist_ok=True, parents=True)

        # Ensure score_cols is a list of lists
        score_cols = [col if isinstance(col, list) else [col] for col in score_cols]
        for target_col in targets:

            if isinstance(target_col, tuple):
                target = '_or_'.join(target_col)
                self.df[target] = self.df[list(target_col)].max(axis=1)
            else:
                target = target_col

            for score_set in score_cols:
                cols_needed = [target] + list(score_set)
                for col in [cat_controls, other_controls]:
                    if col:
                        cols_needed += col
                reg_df = self.df.replace([np.inf, -np.inf], np.nan)\
                                .dropna(subset=cols_needed).copy()

                reg_df[target] = reg_df[target].astype(int)

                reg_df = self.collapse_and_drop_sparse(reg_df, target, cat_controls)

                # Drop perfect separation categories
                if cat_controls:
                    for control in cat_controls:
                        reg_df = self.drop_perfect_separation(reg_df, target, control)

                # Build formula
                terms = list(score_set)
                if cat_controls:
                    terms.extend([f'C({c})' for c in cat_controls])
                if other_controls:
                    terms.extend(other_controls)
                if cat_interaction_terms:
                    terms.extend([f'C({c1}):C({c2})' for c1, c2 in cat_interaction_terms])
                    cols_needed += cat_interaction_terms
                if other_interaction_terms:
                    terms.extend([f'{c1} * {c2}' for c1, c2 in other_interaction_terms])
                    cols_needed += other_interaction_terms

                formula = f"{target} ~ {' + '.join(terms)}"
                print(f"Fitting: {formula} (n={len(reg_df)})")

                try:
                    result = smf.logit(formula=formula, data=reg_df).fit(disp=0, cov_type='HC1', maxiter=self.maxiter)
                    pseudo_r2 = 1 - result.llf / result.llnull

                    # Save all score coefs
                    for score in score_set:
                        summary_rows.append({
                            'target': target,
                            'score': '+'.join(score_set),
                            'coef_name': score,
                            'coef': result.params.get(score, np.nan),
                            'std_err': result.bse.get(score, np.nan),
                            'pval': result.pvalues.get(score, np.nan),
                            'pseudo_r2': pseudo_r2,
                            'n_obs': len(reg_df)
                        })
                    if save_full_summary:
                        fname = summary_folder / f"reg_summary_{target}_{'+'.join(score_set)}.txt"
                        with open(fname, 'w') as f:
                            f.write(result.summary().as_text())
                    print(f"Successfully fitted full data for: {formula}")

                except Exception as e:
                    print(f"Fitting error with {target}, {score_set}: {e}")

                try:
                    auc_rows.append(self._estimate_performance(
                        reg_df[list(set(cols_needed))],
                        formula,
                        target,
                        score_set,
                        cat_controls,
                        k_folds
                    ))

                except Exception as e:
                    print(f"CV error with {target}, {score_set}: {e}")

        pd.DataFrame(summary_rows).to_csv(out_folder / 'report_regression_results.csv', index=False)
        pd.DataFrame(auc_rows).to_csv(out_folder / 'report_auc_scores.csv', index=False)
        print(f"\nSaved regression summaries and AUC scores to {out_folder}")
