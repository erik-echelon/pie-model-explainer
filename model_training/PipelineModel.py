from typing import Any, Optional

import shap
from sklearn.base import (BaseEstimator, ClassifierMixin, RegressorMixin,
                          TransformerMixin)
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted


class TypeSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # Detect numeric and categorical features during fitting
        self.num_features = X.select_dtypes(include=["number"]).columns.tolist()
        self.cat_features = X.select_dtypes(exclude=["number"]).columns.tolist()

        # NOTES: if only two variables, consider as boolean
        # EXPAND THIS LOGIC
        # IMPUTER FOR BOOLEAN
        # OBJECT for pandas as well
        return self

    def transform(self, X):
        # Return X unchanged, this is just for fitting purposes
        return X


class PipelineRegressor(RegressorMixin, BaseEstimator):

    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        categorical_encoder: Optional[TransformerMixin] = None,
        imputer: Optional[TransformerMixin] = None,
        centering_scaler: Optional[TransformerMixin] = None,
        normalizing_scaler: Optional[TransformerMixin] = None,
    ) -> None:
        
        # ADD WHAT TYPE OF VARIABLE THIS
        # ADD MORE DOCSTRING AND README
        """
        Custom Estimator for preprocessing and modeling using a Pipeline.

        Parameters:
        - model (BaseEstimator, optional): The machine learning model to use. Defaults to ElasticNet if None.
        - categorical_encoder (TransformerMixin, optional): Encoder for processing categorical variables.
            Defaults to OneHotEncoder with specified parameters if None.
        - imputer (TransformerMixin, optional): Imputer for handling missing values in numeric features.
            Defaults to SimpleImputer with median strategy if None.
        - centering_scaler (TransformerMixin, optional): Scaler for centering numeric features.
            Defaults to RobustScaler without scaling if None.
        - normalizing_scaler (TransformerMixin, optional): Scaler for normalizing numeric features.
            Defaults to StandardScaler without centering if None.
        """
        self.model = model if model is not None else ElasticNet()

        if categorical_encoder is not None:
            self.categorical_encoder = categorical_encoder
        else:
            self.categorical_encoder = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10,
                sparse_output=False,
                drop="first",
            )

        if imputer is not None:
            self.imputer = imputer
        else:
            self.imputer = SimpleImputer(strategy="median")

        if centering_scaler is not None:
            self.centering_scaler = centering_scaler
        else:
            self.centering_scaler = RobustScaler(with_scaling=False)

        if normalizing_scaler is not None:
            self.normalizing_scaler = normalizing_scaler
        else:
            self.normalizing_scaler = StandardScaler(with_mean=False)

        self.type_selector = TypeSelector()

    def _create_pipeline(self) -> Pipeline:
        """
        Internal method to create a preprocessing and classification pipeline.

        Returns:
        - Pipeline: Configured scikit-learn Pipeline object.
        """

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", self.imputer),
                ("median_scaler", self.centering_scaler),
                ("std_scaler", self.normalizing_scaler),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "encoder",
                    self.categorical_encoder,
                )
            ]
        )

        # Create a MissingIndicator transformer
        missing_indicator = Pipeline(
            steps=[
                ("indicator", MissingIndicator()),
                ("imputer", self.imputer),
                ("median_scaler", self.centering_scaler),
                ("std_scaler", self.normalizing_scaler),
            ]
        )

        # Create a FeatureUnion to combine the numeric transformer and the MissingIndicator
        numeric_transformer_with_indicator = FeatureUnion(
            transformer_list=[
                ("missing_indicator", missing_indicator),
                ("numeric_features", numeric_transformer)
            ]
        )

        if len(self.type_selector.cat_features):
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer_with_indicator, self.type_selector.num_features),
                    ("cat", categorical_transformer, self.type_selector.cat_features),
                ]
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer_with_indicator, self.type_selector.num_features),
                ]
            )

        preprocessor.set_output(transform="pandas")

        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", self.model),
            ]
        )

        clf.set_output(transform="pandas")
        return clf

    def fit(self, X: Any, y: Optional[Any] = None, **fit_params) -> "PipelineRegressor":
        """
        Fit the pipeline to the data.

        Parameters:
        - X (Any): Input data to fit.
        - y (Any, optional): Target values.

        Returns:
        - PipelineRegressor: Fitted instance of itself.
        """

        self.type_selector.fit(X)
        self.pipeline_ = self._create_pipeline()

        self.pipeline_.fit(X, y, **fit_params)

        return self

    def predict(self, X: Any) -> Any:
        """
        Apply the pipeline to make predictions.

        Parameters:
        - X (Any): Input data for making predictions.

        Returns:
        - Any: Predicted values.

        """
        check_is_fitted(self, "pipeline_")  # Check if the model has been fitted
        return self.pipeline_.predict(X)

    def shap_values(self, X: Any) -> Any:
        # shap values can be used for plotting a variety of function
        # Create an explainer object
        explainer = shap.Explainer(self.model.named_steps["regressor"])

        X2 = self.model.named_steps["preprocessor"].transform(X)

        # Compute SHAP values
        shap_values = explainer(X2)

        return shap_values


class PipelineClassifier(ClassifierMixin, BaseEstimator):

    def __init__(
        self,
        model: Optional[BaseEstimator] = None,
        categorical_encoder: Optional[TransformerMixin] = None,
        imputer: Optional[TransformerMixin] = None,
        centering_scaler: Optional[TransformerMixin] = None,
        normalizing_scaler: Optional[TransformerMixin] = None,
    ) -> None:
        """
        Custom Estimator for preprocessing and modeling using a Pipeline.

        Parameters:
        - model (BaseEstimator, optional): The machine learning model to use. Defaults to ElasticNet if None.
        - categorical_encoder (TransformerMixin, optional): Encoder for processing categorical variables.
            Defaults to OneHotEncoder with specified parameters if None.
        - imputer (TransformerMixin, optional): Imputer for handling missing values in numeric features.
            Defaults to SimpleImputer with median strategy if None.
        - centering_scaler (TransformerMixin, optional): Scaler for centering numeric features.
            Defaults to RobustScaler without scaling if None.
        - normalizing_scaler (TransformerMixin, optional): Scaler for normalizing numeric features.
            Defaults to StandardScaler without centering if None.
        """
        self.model = model if model is not None else ElasticNet()

        if categorical_encoder is not None:
            self.categorical_encoder = categorical_encoder
        else:
            self.categorical_encoder = OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=10,
                sparse_output=False,
                # drop="first",
            )

        if imputer is not None:
            self.imputer = imputer
        else:
            self.imputer = SimpleImputer(strategy="median")

        if centering_scaler is not None:
            self.centering_scaler = centering_scaler
        else:
            self.centering_scaler = RobustScaler(with_scaling=False)

        if normalizing_scaler is not None:
            self.normalizing_scaler = normalizing_scaler
        else:
            self.normalizing_scaler = StandardScaler(with_mean=False)

        self.type_selector = TypeSelector()

    def _create_pipeline(self) -> Pipeline:
        """
        Internal method to create a preprocessing and classification pipeline.

        Returns:
        - Pipeline: Configured scikit-learn Pipeline object.
        """

        numeric_transformer = Pipeline(
            steps=[
                ("imputer", self.imputer),
                ("median_scaler", self.centering_scaler),
                ("std_scaler", self.normalizing_scaler),
            ]
        )

        categorical_transformer = Pipeline(
            steps=[
                (
                    "encoder",
                    self.categorical_encoder,
                )
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.type_selector.num_features),
                ("cat", categorical_transformer, self.type_selector.cat_features),
            ]
        )

        preprocessor.set_output(transform="pandas")

        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", self.model),
            ]
        )

        clf.set_output(transform="pandas")
        return clf

    def fit(self, X: Any, y: Optional[Any] = None) -> "PipelineClassifier":
        """
        Fit the pipeline to the data.

        Parameters:
        - X (Any): Input data to fit.
        - y (Any, optional): Target values.

        Returns:
        - PipelineClassifier: Fitted instance of itself.
        """

        self.type_selector.fit(X)
        self.pipeline_ = self._create_pipeline()

        self.pipeline_.fit(X, y)
        self.classes_ = self.pipeline_.named_steps["classifier"].classes_

        return self

    def predict(self, X: Any) -> Any:
        """
        Apply the pipeline to make predictions.

        Parameters:
        - X (Any): Input data for making predictions.

        Returns:
        - Any: Predicted values.

        Raises:
        - NotFittedError: If the classifier is not fitted yet.
        """
        check_is_fitted(self, "pipeline_")  # Check if the model has been fitted
        return self.pipeline_.predict(X)

    def predict_proba(self, X: Any) -> Any:
        """
        Apply the pipeline to make predictions.

        Parameters:
        - X (Any): Input data for making predictions.

        Returns:
        - Any: Predicted values.

        Raises:
        - NotFittedError: If the classifier is not fitted yet.
        """
        check_is_fitted(self, "pipeline_")  # Check if the model has been fitted
        return self.pipeline_.predict_proba(X)

    def shap_values(self, X: Any) -> Any:
        # shap values can be used for plotting a variety of function
        # Create an explainer object
        explainer = shap.Explainer(self.model.named_steps["classifier"])

        X2 = self.model.named_steps["preprocessor"].transform(X)

        # Compute SHAP values
        shap_values = explainer(X2)

        return shap_values
