import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PipelineModel import PipelineRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gini_norm(y: Union[np.ndarray, pd.Series], 
              y_pred: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate the normalized Gini coefficient between actual and predicted values.
    
    The Gini coefficient measures the inequality between values of a frequency distribution.
    In this context, it measures how well the predictions discriminate between different actual values.
    
    Args:
        y: Actual target values
        y_pred: Predicted values
        
    Returns:
        float: Normalized Gini coefficient (ranges from -1 to 1, where 1 is perfect prediction)
    
    Example:
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.8])
        >>> gini = gini_norm(y_true, y_pred)
    """
    # Convert inputs to numpy arrays for consistent handling
    y = np.array(y)
    y_pred = np.array(y_pred)
    
    # Convert to pandas Series for easier manipulation
    y_series = pd.Series(y, name='target')
    y_pred_series = pd.Series(y_pred, name='pred')
    df1 = pd.concat([y_series, y_pred_series], axis=1)
    
    # Sort by predictions to create Lorenz curve
    df_sorted = df1.sort_values(by='pred', ascending=True)
    
    # Calculate cumulative distribution
    n = len(df_sorted)
    cumulative_target = np.cumsum(df_sorted['target']) / df_sorted['target'].sum()
    
    # Calculate Gini coefficient using trapezoidal rule
    x_values = np.linspace(0, 1, n)  # Represents perfect equality line
    y_values = cumulative_target     # Represents Lorenz curve
    area_lorenz = np.trapz(y_values, x_values)
    gini_coefficient = 1 - 2 * area_lorenz
    
    return gini_coefficient

def cross_validate_model(
    X: pd.DataFrame, 
    y: pd.Series,
    weights: Optional[pd.Series] = None,
    n_splits: int = 5, 
    n_bins: int = 10,  # Number of bins for stratification
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform stratified k-fold cross validation on an ElasticNet model with pipeline preprocessing.
    The continuous target variable is binned to create strata for stratification.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        weights: Optional sample weights Series (default: None)
        n_splits: Number of folds for cross-validation (default: 5)
        n_bins: Number of bins to use for stratification (default: 10)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        Dict containing results including OOF predictions and feature importance
    """
    # Create bins for stratification
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    y_binned = kbd.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    # Log stratification information
    logger.info("\nStratification summary:")
    unique_bins, bin_counts = np.unique(y_binned, return_counts=True)
    for bin_idx, count in zip(unique_bins, bin_counts):
        bin_range = np.percentile(y, [bin_idx/n_bins*100, (bin_idx+1)/n_bins*100])
        logger.info(f"Bin {bin_idx}: {count} samples, range: [{bin_range[0]:.2f}, {bin_range[1]:.2f}]")
    
    # Initialize StratifiedKFold cross-validator
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Initialize arrays and lists for storing results
    oof_predictions = np.zeros(len(X))
    fold_gini_scores: List[float] = []
    feature_importances: List[Dict[str, float]] = []
    
    # Prepare fit parameters including weights if provided
    fit_params = {}
    if weights is not None:
        fit_params['regressor__sample_weight'] = weights
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        logger.info(f"\nTraining fold {fold + 1}/{n_splits}")
        
        # Split data into training and validation sets
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Log fold distribution
        train_bins = y_binned[train_idx]
        val_bins = y_binned[val_idx]
        logger.info("Fold distribution:")
        for bin_idx in range(n_bins):
            train_count = np.sum(train_bins == bin_idx)
            val_count = np.sum(val_bins == bin_idx)
            logger.info(f"Bin {bin_idx}: Train={train_count}, Val={val_count}")
        
        # Update fit parameters with training weights if provided
        if weights is not None:
            fit_params['regressor__sample_weight'] = weights.iloc[train_idx]
        
        # Initialize and train model for this fold
        model_builder = PipelineRegressor(
            model=ElasticNet(alpha=0.001, l1_ratio=0, max_iter=100, tol=1e-4, selection='random')
        )
        model_builder.fit(X_train, y_train, **fit_params)
        
        # Generate and store predictions for this fold
        val_predictions = model_builder.predict(X_val)
        oof_predictions[val_idx] = val_predictions
        
        # Calculate and store Gini score for this fold
        fold_gini = gini_norm(y_val, val_predictions)
        fold_gini_scores.append(fold_gini)
        
        # Extract and store feature importances for this fold
        feature_coef_pairs = [
            (feat, coef) for feat, coef in zip(
                model_builder.pipeline_.named_steps["preprocessor"].fit_transform(X_train).columns,
                model_builder.model.coef_
            ) if coef != 0
        ]
        feature_importances.append(dict(feature_coef_pairs))
        
        logger.info(f"Fold {fold + 1} Gini: {fold_gini:.4f}")
    
    # Calculate summary statistics
    mean_gini = np.mean(fold_gini_scores)
    std_gini = np.std(fold_gini_scores)
    logger.info("\nCross-validation results:")
    logger.info(f"Mean Gini: {mean_gini:.4f} ± {std_gini:.4f}")
    
    # Calculate average feature importances across all folds
    all_features = set()
    for fold_importance in feature_importances:
        all_features.update(fold_importance.keys())
    
    avg_importance: Dict[str, float] = {}
    for feature in all_features:
        coefficients = [
            fold_importance.get(feature, 0) 
            for fold_importance in feature_importances
        ]
        avg_importance[feature] = np.mean(coefficients)
    
    # Sort features by importance
    sorted_features = sorted(
        avg_importance.items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return {
        'oof_predictions': oof_predictions,
        'fold_scores': fold_gini_scores,
        'mean_gini': mean_gini,
        'std_gini': std_gini,
        'feature_importance': sorted_features,
        'stratification_bins': y_binned
    }

def train_final_model(
    X: pd.DataFrame, 
    y: pd.Series,
    weights: Optional[pd.Series] = None
) -> Tuple[PipelineRegressor, List[Tuple[str, float]]]:
    """
    Train a final model on the complete dataset and return the model and sorted coefficients.
    
    Args:
        X: Complete feature DataFrame
        y: Complete target Series
        weights: Optional sample weights Series (default: None)
    
    Returns:
        Tuple containing trained model and feature coefficients
    """
    # Initialize and train model on full dataset
    model = PipelineRegressor(
        model=ElasticNet(alpha=0.001, l1_ratio=0, max_iter=100, tol=1e-4, selection='random')
        # model=LinearRegression()
    )
    
    # Prepare fit parameters including weights if provided
    fit_params = {}
    if weights is not None:
        fit_params['regressor__sample_weight'] = weights
    
    model.fit(X, y, **fit_params)
    
    # Extract feature names and coefficients
    feature_names = model.pipeline_.named_steps["preprocessor"].fit_transform(X).columns
    coefficients = model.model.coef_
    
    # Create and sort feature-coefficient pairs
    feature_coef_pairs = [
        (feat, coef) for feat, coef in zip(feature_names, coefficients) 
        if coef != 0
    ]
    sorted_pairs = sorted(
        feature_coef_pairs, 
        key=lambda x: x[1], 
        reverse=True
    )
    
    return model, sorted_pairs

if __name__ == "__main__":
    # Load feature list and data
    csv_path = "/Users/erikallen/Downloads/dataset_117_p-all.tsv"
    selected_features = pd.read_csv('/Users/erikallen/Downloads/LH-Model.csv', header=None)
    selected_features = list(selected_features[0])
    model_target = "FTR_LOSS_RATIO_REL_MRPP_C250_ADJ"
    weight_column = "FTR_PREMIUM_MODIFIED_RATING_PURE_POST_AUDIT"
    
    # Read and prepare data
    df = pd.read_csv(csv_path, delimiter="\t")
    available_features = [f for f in selected_features if f in df.columns and f != weight_column]
    logger.info(f"Using {len(available_features)} out of {len(selected_features)} features")
    
    X = df[available_features]
    y = df[model_target]
    weights = df[weight_column]

    print(len(X))
    for col in X:
        if len(X[col].dropna()) < len(X) and len(np.unique(X[col])) > 10:
            print(col, len(np.unique(X[col])), len(X[col].dropna()))

    # Perform cross-validation
    results = cross_validate_model(X, y, weights=weights)
    
    # Display top features from cross-validation
    logger.info("\nTop 10 features by importance (from cross-validation):")
    for feature, importance in results['feature_importance'][:10]:
        logger.info(f"{feature}: {importance:.6f}")
    
    # Calculate overall out-of-fold Gini
    overall_gini = gini_norm(y, results['oof_predictions'])
    logger.info(f"\nOverall out-of-fold Gini: {overall_gini:.4f}")
    
    # Train final model on all data
    logger.info("\nTraining final model on all data...")
    final_model, final_coefficients = train_final_model(X, y, weights=weights)
    
    # Print final model coefficients
    logger.info("\nFinal model coefficients (sorted by value):")
    for feature, coef in final_coefficients:
        logger.info(f"{feature}: {coef:.6f}")
    
    # Compare to PPM 1.0.1 predictions if available
    try:
        df2 = pd.read_csv(
            '/Users/erikallen/Downloads/dataset117_preds_on_PPM1.0.1.csv', 
            low_memory=False
        )
        ppm_predictions = df2.set_index('POL_TERM_HASHKEY').loc[
            df['POL_TERM_HASHKEY'], 
            'Prediction'
        ]
        ppm_gini = gini_norm(y, ppm_predictions)
        logger.info(f"PPM 1.0.1 Gini: {ppm_gini:.4f}")
        
        # Calculate correlations between different predictions
        logger.info("\nCorrelations:")
        logger.info(
            f"OOF vs PPM: {np.corrcoef(results['oof_predictions'], ppm_predictions)[0,1]:.4f}"
        )
        logger.info(
            f"OOF vs Target: {np.corrcoef(results['oof_predictions'], y)[0,1]:.4f}"
        )
        logger.info(
            f"PPM vs Target: {np.corrcoef(ppm_predictions, y)[0,1]:.4f}"
        )
    except Exception as e:
        logger.warning(f"Could not load PPM predictions: {str(e)}")
        
    # Create output DataFrame with all features and predictions
    logger.info("\nSaving results to CSV...")
    output_df = df.copy()  # Start with all features
    
    # Add target values
    output_df['actual_target'] = y
    
    # Add OOF predictions
    output_df['oof_predictions'] = results['oof_predictions']
    
    # Add PPM predictions if available
    if 'ppm_predictions' in locals():
        output_df['ppm_predictions'] = list(ppm_predictions)
    
    # Add identifier if available
    if 'POL_TERM_HASHKEY' in df.columns:
        output_df['POL_TERM_HASHKEY'] = df['POL_TERM_HASHKEY']
    
    # Save to CSV
    output_path = 'model_predictions_comparison.csv'
    output_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")

    # Log the shape of the saved file
    logger.info(f"Saved file shape: {output_df.shape}")
    
    # Log the columns in the saved file
    logger.info("Columns in saved file:")
    logger.info(", ".join(output_df.columns))
    
    # Create and save coefficients DataFrame
    logger.info("\nSaving coefficients to CSV...")
    coef_df = pd.DataFrame(final_coefficients, columns=['feature', 'coefficient'])
    
    # Add cross-validation importance for comparison
    cv_importance_dict = dict(results['feature_importance'])
    coef_df['cv_importance'] = coef_df['feature'].map(cv_importance_dict)
    
    # Calculate absolute values for easier sorting
    coef_df['abs_coefficient'] = abs(coef_df['coefficient'])
    coef_df['abs_cv_importance'] = abs(coef_df['cv_importance'])
    
    # Sort by absolute coefficient value
    coef_df = coef_df.sort_values('coefficient', ascending=False)
    
    # Remove helper columns used for sorting
    coef_df = coef_df.drop(['abs_coefficient', 'abs_cv_importance'], axis=1)
    
    # Save coefficients to CSV
    coef_output_path = 'model_coefficients.csv'
    coef_df.to_csv(coef_output_path, index=False)
    logger.info(f"Coefficients saved to {coef_output_path}")
    
    # Log information about the coefficients file
    logger.info(f"Number of non-zero coefficients: {len(coef_df)}")
    logger.info("\nTop 10 coefficients by absolute value:")
    logger.info(coef_df.head(10).to_string())
