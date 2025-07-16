import json
import joblib
import pickle
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from optuna.integration import OptunaSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV


class ModelEvaluation:
    """Evaluation pipeline for model selection, feature selection, and hyperparameter tuning with nested CV."""

    def __init__(self, target_vars: list[str], model_specs: dict, random_state: int = 42):
        """
        Args:
            target_vars: List of target column names.
            model_specs: Dict of model specifications, including estimator, param grid, features, etc.
            random_state: Random seed for reproducibility.

        Example of model specs:
        model_specs = {
            'logreg': {'model': LogisticRegression, 'param_grid': {...}, ...},
            'rf': {'model': RandomForestClassifier, 'param_grid': {...}, ...},
            ...
        }
        """
        self.target_vars = target_vars
        self.model_specs = model_specs
        self.random_state = random_state
        self.best_params = {m: {t: None for t in target_vars} for m in model_specs}
        self.selected_features = {m: {t: None for t in target_vars} for m in model_specs}
        self.best_models = {m: {t: None for t in target_vars} for m in model_specs}
        self.production_models = {m: {t: None for t in target_vars} for m in model_specs}
        self.metrics_report = {m: {t: None for t in target_vars} for m in model_specs}

        for model_name, spec in self.model_specs.items():
            self._assert_feature_lists_mutually_exclusive(spec['features'])

    def _assert_feature_lists_mutually_exclusive(self, features_dict: dict) -> None:
        """
        Checks that feature lists for a given model spec are mutually exclusive (no overlap).

        Args:
            features_dict: Dict mapping feature set names to lists of feature names.

        Raises:
            ValueError: If duplicates are detected across feature lists.
        """
        all_features = []
        for name, feature_list in features_dict.items():
            all_features.extend(feature_list)
        if len(all_features) != len(set(all_features)):
            duplicates = [item for item, count in Counter(all_features).items() if count > 1]
            raise ValueError(f"Duplicate features detected: {duplicates}")

    def _assert_all_features_present(self, features_dict: dict) -> None:
        """
        Ensures all features specified in model specs exist in self.feature_cols.

        Args:
            features_dict: Dict mapping feature set names to lists of feature names.

        Raises:
            ValueError: If any feature is missing from self.feature_cols.
        """
        all_features = []
        for name, feature_list in features_dict.items():
            all_features.extend(feature_list)
        missing_features = [feature for feature in all_features if feature not in self.feature_cols]
        if missing_features:
            raise ValueError(f"Missing features detected: {missing_features}")

    def load_data(self, df: pd.DataFrame) -> None:
        """
        Loads the DataFrame, extracts feature columns, and validates feature specs.

        Args:
            df: Input DataFrame with feature and target columns.
        """
        self.df = df.copy()
        self.feature_cols = [col for col in df.columns if col not in self.target_vars]
        for model_name, spec in self.model_specs.items():
            self._assert_all_features_present(spec['features'])

    def load_best_params(self, file_path: Path, model_names: list[str], target_vars: list[str]) -> None:
        """
        Loads best hyperparameters from file.

        Args:
            file_path: Path to pickled hyperparameter dict.
            model_names: List of model names.
            target_vars: List of target variable names.
        """
        with open(file_path, 'rb') as f:
            all_params = pickle.load(f)
        self.best_params = {
            m: {t: all_params[m][t] for t in target_vars}
            for m in model_names
        }

    def load_best_features(
        self,
        file_path: Path,
        model_names: list[str],
        target_vars: list[str],
        additional_features: list = []
    ) -> None:
        """
        Loads best selected features from file and adds any additional features.

        Args:
            file_path: Path to pickled selected features.
            model_names: List of model names.
            target_vars: List of target variable names.
            additional_features: List of features to append to each set.
        """
        with open(file_path, 'rb') as f:
            all_features = pickle.load(f)
        self.selected_features = {
            m: {t: all_features[m][t] + additional_features for t in target_vars}
            for m in model_names
        }

    def _get_feature_importances(self, model) -> np.ndarray:
        """
        Returns feature importances or coefficients for a fitted model.

        Args:
            model: Fitted scikit-learn estimator.

        Returns:
            Array of feature importances or None if unavailable.
        """
        # Tree-based models
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        # Linear models (coefficients)
        elif hasattr(model, "coef_"):
            return np.abs(model.coef_).flatten()
        else:
            return None

    def _get_transformed_feature_names(
        self,
        preprocessor: ColumnTransformer,
        input_features: list,
        out_folder: Path,
        model_name: str,
        target: str
    ) -> None:
        """Return feature names after a ColumnTransformer fit, including passthroughs."""
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if trans == 'drop':
                continue
            if trans == 'passthrough':
                # cols: list of column names, a slice, or mask
                if isinstance(cols, (list, tuple, np.ndarray)):
                    names.extend(cols)
                elif isinstance(cols, slice):
                    names.extend(input_features[cols])
                else:
                    raise ValueError(f"Unexpected cols type: {type(cols)}")
            else:
                names.extend(trans.get_feature_names_out(cols))

        feature_names_path = out_folder / f'{model_name}_{target}_encoded_feature_names.json'
        with open(feature_names_path, 'w', encoding='utf-8') as f:
            json.dump(list(names), f, ensure_ascii=False, indent=2)
        print(f"Saved encoded feature names to {feature_names_path}")

    def _add_class_weights_to_fit_params(
        self,
        fit_params: dict,
        ModelClass: type,
        model_name: str,
        y: np.ndarray
    ) -> dict:
        """
        Adds appropriate class weight parameters to fit_params for imbalanced learning.

        Args:
            fit_params: Existing fit params.
            ModelClass: Model class.
            model_name: Name of the model.
            y: Target array.

        Returns:
            Updated fit_params with class weights, if applicable.
        """
        if not self.model_specs[model_name]['account_for_class_weights']:
            return fit_params

        if model_name == 'xgb':
            classes = np.unique(y)
            n_pos = np.sum(y == classes[1])
            n_neg = np.sum(y == classes[0])
            fit_params['scale_pos_weight'] = n_neg / n_pos if n_pos > 0 else 1.0

        elif hasattr(ModelClass(), 'class_weight'):
            fit_params['class_weight'] = 'balanced'

        return fit_params

    def _build_preprocessor(
        self,
        preprocessor_steps: list[tuple],
        model_name: str,
        target: str,
        best_features: bool = False
    ) -> ColumnTransformer:
        """
        Builds a ColumnTransformer using preprocessing steps and selected features.

        Args:
            preprocessor_steps: List of (name, transformer) steps.
            model_name: Name of the model.
            target: Target variable name.
            best_features: If True, uses only best features.

        Returns:
            ColumnTransformer for preprocessing.
        """
        steps = []
        for name, transformer in preprocessor_steps:
            features = self.model_specs[model_name]['features'][name]
            if best_features:
                features = [col for col in features if col in self.selected_features[model_name][target]]
            steps.append((name, transformer, features))
        return ColumnTransformer(steps)

    def nested_cv_with_feature_selection(
        self,
        out_folder: Path,
        k_outer: int = 5,
        k_inner: int = 3,
        min_features_to_select: int = 10,
        scoring: str = "average_precision"
    ) -> None:
        """
        Performs nested cross-validation with RFECV feature selection (no hyperparameter tuning).

        Args:
            k_outer: Number of outer CV folds.
            k_inner: Number of inner CV folds for feature selection.
            min_features_to_select: Minimum features to select.
            scoring: Scoring metric for RFECV.
        """
        print("Starting nested CV with feature selection...")
        out_folder.mkdir(parents=True, exist_ok=True)

        for model_name, spec in tqdm(self.model_specs.items(), desc="Models"):

            ModelClass = spec['model']

            for target in tqdm(self.target_vars, desc=f"{model_name} targets", leave=False):
                print(f"Model: {model_name} | Target: {target}")

                preprocessor = self._build_preprocessor(spec['preprocessor_steps'], model_name, target)

                X = self.df[self.feature_cols]
                y = self.df[target]

                outer_skf = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=self.random_state)

                selected_feature_masks = []

                for train_idx, _ in tqdm(list(outer_skf.split(X, y)), desc="Outer folds", leave=False):
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]

                    X_train_proc = preprocessor.fit_transform(X_train)

                    fit_params = self._add_class_weights_to_fit_params(spec['fit_params'], ModelClass, model_name, y_train)
                    model = ModelClass(**fit_params)

                    inner_skf = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=self.random_state)
                    rfecv = RFECV(
                        estimator=model,
                        step=1,
                        min_features_to_select=min_features_to_select,
                        cv=inner_skf,
                        scoring=scoring,
                        n_jobs=-1
                    )

                    rfecv.fit(X_train_proc, y_train)
                    selected_feature_masks.append(rfecv.support_)

                # Aggregate selected features across folds
                selected_feature_masks = np.array(selected_feature_masks)
                mean_mask = selected_feature_masks.mean(axis=0)
                threshold = 0.6  # at least 60% of folds must have selected a feature
                final_mask = mean_mask >= threshold
                final_features = np.array(self.feature_cols)[final_mask].tolist()

                self.selected_features[model_name][target] = final_features
                print(f"[{model_name}/{target}] Selected {len(final_features)} features: {final_features}")

    def nested_cv_with_hyperparam_search(
        self,
        out_folder: Path,
        k_outer: int = 5,
        k_inner: int = 3,
        best_features: bool = False,
        n_trials: int = 200,
        scoring: str = 'average_precision'
    ) -> None:
        """
        Performs nested cross-validation with hyperparameter tuning (grid or Optuna search).

        Args:
            out_folder: Output folder for saving results.
            k_outer: Number of outer CV folds.
            k_inner: Number of inner CV folds.
            best_features: Whether to use best feature subset.
            n_trials: Number of Optuna trials (if using Optuna).
            scoring: Scoring metric for model selection.
        """
        out_folder.mkdir(parents=True, exist_ok=True)

        if best_features and not self.selected_features:
            raise ValueError('To use the best features execure find_best_feature_subset first!')

        print("Starting nested CV with hyperparameter search...")

        for model_name, spec in tqdm(self.model_specs.items(), desc="Models"):

            ModelClass = spec['model']
            param_grid = spec['param_grid']

            for target in tqdm(self.target_vars, desc=f"{model_name} targets", leave=False):
                print(f"[STARTED] Model: {model_name} | Target: {target}")

                # Initialize preprocessor with the specified feature columns from the model specs
                preprocessor = self._build_preprocessor(
                    spec['preprocessor_steps'], model_name, target, best_features
                )

                X = self.df[self.feature_cols]  # Always select all features, dropping of unspecified features is handled by the preprocessor
                y = self.df[target]

                outer_skf = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=self.random_state)

                outer_metrics = {'roc_auc': [], 'pr_auc': [], 'f1_macro': []}
                inner_best_scores = []
                inner_best_params = []

                # Outer loop over k_outer folds
                for train_idx, test_idx in tqdm(list(outer_skf.split(X, y)), desc="Outer folds", leave=False):

                    # Init data of the current outer fold
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    X_train_proc = preprocessor.fit_transform(X_train)
                    X_test_proc = preprocessor.transform(X_test)

                    # Only include relevant fit_params for this model
                    fit_params = self._add_class_weights_to_fit_params(spec['fit_params'], ModelClass, model_name, y_train)
                    model = ModelClass(**fit_params)

                    # Set up inner grid search loop for k_inner folds
                    inner_skf = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=self.random_state)
                    if spec['search_type'] == 'grid':
                        searcher = GridSearchCV(
                            estimator=model,
                            param_grid=param_grid,
                            cv=inner_skf,
                            scoring=scoring,  # Because highly imbalanced data
                            n_jobs=-1
                        )
                    elif spec['search_type'] == 'optuna':
                        searcher = OptunaSearchCV(
                            estimator=model,
                            param_distributions=param_grid,
                            cv=inner_skf,
                            scoring=scoring,
                            n_trials=n_trials,
                            n_jobs=-1,
                            random_state=self.random_state,
                            verbose=0,
                        )
                    elif spec['search_type'] is None:
                        pass
                    else:
                        raise ValueError("search_type must be 'grid' or 'optuna'")

                    if spec['search_type'] is None:
                        best_params = {}
                    else:
                        # Determine best hyperparameters of this fold using only the training data and not testing
                        # training data is then again split into k_inner folds
                        searcher.fit(X_train_proc, y_train)

                        best_params = searcher.best_params_
                        # Select and store best hyperparam config determined on the training data
                        inner_best_scores.append(searcher.best_score_)
                        inner_best_params.append(best_params)

                    # Refit model with full training data to estimate auc and feature importance of the outer fold
                    temp_fit_params = {**fit_params, 'n_jobs': -1}  # For training without CV set to -1
                    params = dict(best_params, **temp_fit_params)
                    best_model = ModelClass(**params)
                    best_model.fit(X_train_proc, y_train)
                    y_pred = best_model.predict(X_test_proc)
                    y_pred_proba = best_model.predict_proba(X_test_proc)[:, 1]

                    outer_metrics['roc_auc'].append(roc_auc_score(y_test, y_pred_proba))
                    outer_metrics['pr_auc'].append(average_precision_score(y_test, y_pred_proba))
                    outer_metrics['f1_macro'].append(f1_score(y_test, y_pred, average="macro"))

                if spec['search_type'] is None:
                    overall_lambda_star = {}  # No tuned hyperparams
                else:
                    # Select hyperparameters from the inner folds that achieved the highest score
                    best_idx = np.argmax(inner_best_scores)
                    overall_lambda_star = inner_best_params[best_idx]

                # Set class weights again based on the full data
                fit_params = self._add_class_weights_to_fit_params(spec['fit_params'], ModelClass, model_name, y)

                # Retrain final model on all data
                temp_fit_params = {**fit_params, 'n_jobs': -1}  # For training without CV set to -1
                best_params = dict(overall_lambda_star, **temp_fit_params)
                production_model = ModelClass(**best_params)

                X_proc = preprocessor.fit_transform(X)
                self._get_transformed_feature_names(
                    preprocessor,
                    list(X.columns),
                    out_folder,
                    model_name,
                    target
                )
                production_model.fit(X_proc, y)

                # Store production model
                self.best_models[model_name][target] = production_model

                # Store best hyperparameters
                self.best_params[model_name][target] = overall_lambda_star

                # Store metrics report for the avg performance of the model on the current target
                self.metrics_report[model_name][target] = {
                    'mean_roc_auc': np.mean(outer_metrics['roc_auc']),
                    'std_roc_auc': np.std(outer_metrics['roc_auc']),
                    'all_roc_auc': outer_metrics['roc_auc'],
                    'mean_pr_auc': np.mean(outer_metrics['pr_auc']),
                    'std_pr_auc': np.std(outer_metrics['pr_auc']),
                    'all_pr_auc': outer_metrics['pr_auc'],
                    'mean_f1_macro': np.mean(outer_metrics['f1_macro']),
                    'std_f1_macro': np.std(outer_metrics['f1_macro']),
                    'all_f1_macro': outer_metrics['f1_macro'],
                    'best_params': overall_lambda_star,
                }
                print(f"[FINISHED] Model: {model_name} | Target: {target} | Mean ROC-AUC: {np.mean(outer_metrics['roc_auc']):.4f} | Mean PR-AUC: {np.mean(outer_metrics['pr_auc']):.4f}")
        self._save_models_and_reports(out_folder)

    def _save_models_and_reports(self, out_folder: Path) -> None:
        """
        Saves trained models, metrics, hyperparameters, and selected features to disk.

        Args:
            out_folder: Output folder path.
        """
        # Save best models stored in self.best_models[model_name][target]
        models_dir = out_folder / 'trained_models'
        models_dir.mkdir(exist_ok=True)
        for model_name, targets in self.best_models.items():
            for target, model in targets.items():
                model_path = models_dir / f'{model_name}_{target}.joblib'
                joblib.dump(model, model_path)
                print(f"Saved model for {model_name}/{target} to {model_path}")

        # Save metrics report as csv file
        summary_rows = []
        for model_name, model_results in self.metrics_report.items():
            for target, d in model_results.items():
                summary_rows.append({
                    'model': model_name,
                    'target': target,
                    'mean_roc_auc': d['mean_roc_auc'],
                    'std_roc_auc': d['std_roc_auc'],
                    'all_roc_auc': str(d['all_roc_auc']),
                    'mean_pr_auc': d['mean_pr_auc'],
                    'std_pr_auc': d['std_pr_auc'],
                    'all_pr_auc': str(d['all_pr_auc']),
                    'mean_f1_macro': d['mean_f1_macro'],
                    'std_f1_macro': d['std_f1_macro'],
                    'all_f1_macro': str(d['all_f1_macro']),
                    'best_params': str(d['best_params']),
                })
        pd.DataFrame(summary_rows).to_csv(out_folder / 'cv_metrics_report.csv', index=False)
        print(f"Saved metrics summary to {str(out_folder / 'cv_metrics_report.csv')}")

        # Save hyperparameters
        best_params_path = out_folder / 'best_hyperparameters.pkl'
        with open(best_params_path, 'wb') as f:
            pickle.dump(self.best_params, f)
        print(f"Saved best hyperparameters to {best_params_path}")

        # Save best feature set
        best_features_path = out_folder / 'best_features.pkl'
        with open(best_features_path, 'wb') as f:
            pickle.dump(self.selected_features, f)
        print(f"Saved best features to {best_features_path}")
