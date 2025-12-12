#!/usr/bin/env python3
"""
Simple SHAP Waterfall Text Explainer
====================================

A minimal script that takes a trained model and provides clean text output
of SHAP waterfall explanations for individual rows.

Usage:
    python simple_shap_explainer.py --help
    python simple_shap_explainer.py --data sample_data.json
    python simple_shap_explainer.py --csv data.csv --row 0
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import Ridge

from model_training.PipelineModel import PipelineRegressor

logging.basicConfig(level=logging.WARNING)


@dataclass
class ShapExplanation:
    """Container for all SHAP explanation outputs."""
    prediction: float
    base_value: float
    contributions_df: pd.DataFrame
    text_explanation: str
    waterfall_plot: Optional[plt.Figure] = None
    
    @property
    def risk_delta(self) -> float:
        """Difference between prediction and base value."""
        return self.prediction - self.base_value
    
    @property
    def risk_pct_change(self) -> float:
        """Percentage change from base value."""
        if self.base_value == 0:
            return 0.0
        return (self.risk_delta / self.base_value) * 100


class SimpleShapExplainer:
    """Simple SHAP explainer that outputs clean text waterfall explanations."""
    
    def __init__(
        self, 
        model_path: str = None, 
        explainer_path: str = None, 
        feature_metadata_path: str = None
    ):
        """Initialize with a model path, explainer path, or train a new model."""
        if explainer_path and Path(explainer_path).exists():
            self._load_complete_explainer(explainer_path)
        elif model_path and Path(model_path).exists():
            self.model = self._load_model(model_path)
            self._load_feature_metadata(feature_metadata_path)
            self._setup_shap_explainer()
        else:
            self.model = self._train_model()
            self._load_feature_metadata(feature_metadata_path)
            self._setup_shap_explainer()
    
    def _load_model(self, model_path: str):
        """Load a saved model."""
        import pickle
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    
    def _train_model(self):
        """Train a simple model if no model provided."""
        csv_path = "model_validation/dataset_117_p-all.tsv"
        features_path = "model_validation/LH-Model.csv"
                
        df = pd.read_csv(csv_path, delimiter="\t")
        selected_features = pd.read_csv(features_path, header=None)
        available_features = [
            f for f in selected_features[0] 
            if f in df.columns and f != "FTR_PREMIUM_MODIFIED_RATING_PURE_POST_AUDIT"
        ]
        
        X = df[available_features]
        y = df["FTR_LOSS_RATIO_REL_MRPP_C250_ADJ"]
        weights = df["FTR_PREMIUM_MODIFIED_RATING_PURE_POST_AUDIT"]
        
        model = PipelineRegressor(
            model=Ridge(alpha=0.1, max_iter=1000, tol=1e-4, solver='auto')
        )
        model.fit(X, y, regressor__sample_weight=weights)
        
        self.feature_names = available_features
        self.sample_data = X.sample(100, random_state=42)
        
        return model
    
    def _load_feature_metadata(self, metadata_path: str = None):
        """Load feature metadata from CSV file."""
        if metadata_path is None:
            metadata_path = "model_validation/PPM Explains - Sheet1.csv"
        
        self.feature_metadata = {}
        if Path(metadata_path).exists():
            try:
                df = pd.read_csv(metadata_path)
                for _, row in df.iterrows():
                    feature_name = row['Feature']
                    self.feature_metadata[feature_name] = {
                        'short_name': row.get('Short Name', feature_name),
                        'description': row.get('Description', ''),
                        'formula': row.get('formula', ''),
                        'explain': row.get('Explain', ''),
                        'direction': row.get('Direction', '')
                    }
                print(f"Loaded metadata for {len(self.feature_metadata)} features")
            except Exception as e:
                print(f"Warning: Could not load feature metadata from {metadata_path}: {e}")
                self.feature_metadata = {}
        else:
            print(f"Warning: Feature metadata file not found at {metadata_path}")
            self.feature_metadata = {}
    
    def _setup_shap_explainer(self):
        """Setup SHAP explainer with background data."""
        if hasattr(self, 'sample_data'):
            background = self.sample_data
        else:
            csv_path = "model_validation/dataset_117_p-all.tsv"
            df = pd.read_csv(csv_path, delimiter="\t", low_memory=False)
            features_path = "model_validation/LH-Model.csv"
                
            selected_features = pd.read_csv(features_path, header=None)
            self.feature_names = [
                f for f in selected_features[0] 
                if f in df.columns and f != "FTR_PREMIUM_MODIFIED_RATING_PURE_POST_AUDIT"
            ]
            background = df[self.feature_names].sample(100, random_state=42)
        
        background_transformed = self.model.pipeline_.named_steps["preprocessor"].transform(
            background
        )
        
        self.explainer = shap.Explainer(
            self._predict_function,
            background_transformed
        )
    
    def _clean_feature_name(self, feature_name: str) -> str:
        """Clean pipeline prefixes from a feature name."""
        clean_feature = feature_name
        
        prefixes_to_remove = [
            'num__missing_indicator__missingindicator_',
            'num__numeric_features__',
            'cat__categorical_features__', 
            'num__missing_indicator__',
            'cat__',
            'num__',
            'remainder__'
        ]
        
        is_missing = False
        for prefix in prefixes_to_remove:
            if clean_feature.startswith(prefix):
                clean_feature = clean_feature[len(prefix):]
                if 'missing_indicator' in prefix:
                    is_missing = True
                break
        
        if is_missing:
            clean_feature = 'MISSING_' + clean_feature
            
        return clean_feature
    
    def _resolve_display_name(self, clean_name: str, name_map: Dict[str, str] = None) -> str:
        """Resolve a display name by handling suffixes and looking up base names."""
        # 1. Direct match in name_map (e.g. "Age")
        if name_map and clean_name in name_map:
            return name_map[clean_name]

        # 2. Direct match in metadata
        if self.feature_metadata and clean_name in self.feature_metadata:
            return self.feature_metadata[clean_name].get('short_name', clean_name)

        # 3. Handle Categorical Suffixes (_True/_False)
        if clean_name.endswith(('_True', '_False')):
            suffix = '_True' if clean_name.endswith('_True') else '_False'
            base_name = clean_name[:-len(suffix)]
            val_text = "Yes" if suffix == '_True' else "No"
            
            # Resolve the base name
            base_display = base_name
            if name_map and base_name in name_map:
                base_display = name_map[base_name]
            elif self.feature_metadata and base_name in self.feature_metadata:
                base_display = self.feature_metadata[base_name].get('short_name', base_name)

            return f"{base_display} = {val_text}"

        # 4. Handle Missing Indicators
        if clean_name.startswith('MISSING_'):
            base_name = clean_name[8:]  # Remove 'MISSING_'
            
            # Resolve the base name
            base_display = base_name
            if name_map and base_name in name_map:
                base_display = name_map[base_name]
            elif self.feature_metadata and base_name in self.feature_metadata:
                base_display = self.feature_metadata[base_name].get('short_name', base_name)
                 
            return f"{base_display} ()"

        # 5. Fallback
        return clean_name
    
    def _get_display_name(self, feature_name: str) -> str:
        """Get display name for a feature, using metadata if available."""
        clean_feature = self._clean_feature_name(feature_name)
        
        if clean_feature in self.feature_metadata:
            display_name = self.feature_metadata[clean_feature]['short_name']
            if display_name and display_name.strip():
                return display_name
        
        return clean_feature
    
    def _get_display_names_list(self, feature_names: List[str]) -> List[str]:
        """Convert a list of raw feature names to display names."""
        return [self._get_display_name(f) for f in feature_names]

    def explain_full(
        self, 
        data: Union[Dict, pd.Series, pd.DataFrame], 
        max_features: int = 10, 
        include_plot: bool = True,
        name_map: Dict[str, str] = None
    ) -> ShapExplanation:
        """
        Generate complete SHAP explanation with all computed values.
        
        Args:
            data: Input data (dict, Series, or single-row DataFrame)
            max_features: Maximum number of features to show
            include_plot: Whether to generate matplotlib waterfall plot
            name_map: Optional dict mapping raw feature names to display names
            
        Returns:
            ShapExplanation dataclass containing:
                - prediction: Model prediction value
                - base_value: SHAP base value (average prediction)
                - contributions_df: DataFrame with raw_value, transformed_value, contribution
                - text_explanation: Formatted text explanation
                - waterfall_plot: matplotlib Figure (if include_plot=True)
        """
        # Prepare data
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            if len(data) != 1:
                raise ValueError("DataFrame must contain exactly one row")
            df = data.copy()
        else:
            raise ValueError("Data must be dict, Series, or single-row DataFrame")
        
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        
        df = df[self.feature_names]
        
        # Store raw values before transformation
        raw_values = df.iloc[0].to_dict()
        
        # Core computation - done once
        prediction = self.model.predict(df)[0]
        data_transformed = self.model.pipeline_.named_steps["preprocessor"].transform(df)
        shap_values = self.explainer(data_transformed)
        
        base_value = (
            shap_values.base_values[0] 
            if hasattr(shap_values.base_values, '__len__') 
            else shap_values.base_values
        )
        values = (
            shap_values.values[0] 
            if hasattr(shap_values.values[0], '__len__') 
            else shap_values.values
        )
        
        # Get preprocessed feature names
        try:
            preprocessed_features = list(
                self.model.pipeline_.named_steps["preprocessor"].get_feature_names_out()
            )
        except:
            preprocessed_features = [f"feature_{i}" for i in range(len(values))]
        
        # Get transformed values
        if isinstance(data_transformed, pd.DataFrame):
            transformed_values = data_transformed.iloc[0].to_dict()
        else:
            transformed_values = dict(zip(preprocessed_features, data_transformed[0]))
        
        # Build contributions list with both raw and transformed values
        contributions = []
        for prep_feature, shap_value in zip(preprocessed_features, values):
            if abs(shap_value) > 1e-6:
                clean_feature = self._clean_feature_name(prep_feature)
                
                # Get display name and description from metadata
                display_name = clean_feature
                description = ''
                if clean_feature in self.feature_metadata:
                    meta = self.feature_metadata[clean_feature]
                    display_name = meta['short_name'] or clean_feature
                    description = meta['explain']
                
                # Get transformed value for this preprocessed feature
                transformed_val = transformed_values.get(prep_feature, np.nan)
                
                # Determine feature type
                is_missing_indicator = (
                    'missing_indicator' in prep_feature.lower() or 
                    clean_feature.startswith('MISSING_')
                )
                is_categorical = (
                    prep_feature.startswith('cat__') or 
                    clean_feature.endswith(('_True', '_False'))
                )
                
                # Map back to raw value
                raw_val = self._get_raw_value(
                    clean_feature, raw_values, is_missing_indicator, is_categorical
                )
                
                contributions.append({
                    'feature': clean_feature,
                    'preprocessed_feature': prep_feature,
                    'display_name': display_name,
                    'description': description,
                    'raw_value': raw_val,
                    'transformed_value': transformed_val,
                    'contribution': shap_value,
                    'abs_contribution': abs(shap_value),
                    'is_missing_indicator': is_missing_indicator,
                    'is_categorical': is_categorical,
                })
        
        contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        # Build DataFrame
        df_contributions = pd.DataFrame(contributions)
        if len(df_contributions) > 0:
            df_contributions['rank'] = range(1, len(df_contributions) + 1)
            df_contributions['impact_direction'] = df_contributions['contribution'].apply(
                lambda x: 'Positive' if x > 0 else 'Negative'
            )
            # Reorder columns
            df_contributions = df_contributions[[
                'rank', 'feature', 'preprocessed_feature', 'display_name', 'description',
                'raw_value', 'transformed_value', 'contribution', 'impact_direction', 
                'abs_contribution', 'is_missing_indicator', 'is_categorical'
            ]]
        
        # Generate text explanation
        text_output = self._generate_text_explanation(
            prediction, base_value, contributions, max_features
        )
        
        # Generate plot if requested
        waterfall_fig = None
        if include_plot:
            waterfall_fig = self._create_waterfall_plot(data, max_features, name_map, raw_values, contributions)
        
        return ShapExplanation(
            prediction=prediction,
            base_value=base_value,
            contributions_df=df_contributions,
            text_explanation=text_output,
            waterfall_plot=waterfall_fig
        )
    
    def _get_raw_value(
        self, 
        clean_feature: str, 
        raw_values: Dict, 
        is_missing_indicator: bool, 
        is_categorical: bool
    ):
        """Extract the raw value for a feature."""
        if is_missing_indicator:
            # For missing indicators, show whether the base feature was missing
            base_feature = clean_feature.replace('MISSING_', '')
            original_val = raw_values.get(base_feature, np.nan)
            return pd.isna(original_val)
        elif is_categorical:
            # For categorical, find the base feature and get its raw value
            base_feature = clean_feature
            for suffix in ['_True', '_False']:
                if clean_feature.endswith(suffix):
                    base_feature = clean_feature[:-len(suffix)]
                    break
            return raw_values.get(base_feature, raw_values.get(clean_feature, np.nan))
        else:
            # Numeric feature - direct lookup
            return raw_values.get(clean_feature, np.nan)
    
    def explain_text(
        self, 
        data: Union[Dict, pd.Series, pd.DataFrame], 
        max_features: int = 10
    ) -> str:
        """Generate a clean text waterfall explanation."""
        result = self.explain_full(data, max_features, include_plot=False)
        return result.text_explanation
    
    def _generate_text_explanation(
        self, 
        prediction: float, 
        base_value: float, 
        contributions: List[Dict], 
        max_features: int
    ) -> str:
        """Generate the text explanation from prediction and contributions."""
        output = []
        output.append("=" * 60)
        output.append("SHAP WATERFALL EXPLANATION")
        output.append("=" * 60)
        output.append(f"Prediction: {prediction:.4f}")
        output.append(f"Base Value: {base_value:.4f}")
        output.append(f"Total Change: {prediction - base_value:+.4f}")
        output.append("")
        output.append("Feature Contributions (sorted by impact):")
        output.append("-" * 60)
        
        running_total = base_value
        for i, contrib in enumerate(contributions[:max_features]):
            running_total += contrib['contribution']
            direction = "↑" if contrib['contribution'] > 0 else "↓"
            
            display_name = contrib.get('display_name', contrib['feature'])[:40]
            output.append(
                f"{i+1:2d}. {display_name:<40} "
                f"{contrib['contribution']:+8.4f} {direction} "
                f"(Total: {running_total:7.4f})"
            )
        
        if len(contributions) > max_features:
            remaining = sum(c['contribution'] for c in contributions[max_features:])
            running_total += remaining
            output.append(
                f"    {'... remaining features':<40} "
                f"{remaining:+8.4f}   "
                f"(Total: {running_total:7.4f})"
            )
        
        output.append("-" * 60)
        output.append(f"Final Prediction: {prediction:.4f}")
        
        return "\n".join(output)
    
    def _prepare_shap_values(
        self, 
        data: Union[Dict, pd.Series, pd.DataFrame]
    ) -> shap.Explanation:
        """Prepare SHAP values with display names for plotting."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            if len(data) != 1:
                raise ValueError("DataFrame must contain exactly one row")
            df = data.copy()
        else:
            raise ValueError("Data must be dict, Series, or single-row DataFrame")
        
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = np.nan
        
        df = df[self.feature_names]
        data_transformed = self.model.pipeline_.named_steps["preprocessor"].transform(df)
        shap_values = self.explainer(data_transformed)
        
        # Replace feature names with display names
        if hasattr(shap_values[0], 'feature_names') and self.feature_metadata:
            display_names = self._get_display_names_list(shap_values[0].feature_names)
            shap_values[0].feature_names = display_names
        
        return shap_values
    
    def _create_waterfall_plot(
        self, 
        data: Union[Dict, pd.Series, pd.DataFrame], 
        max_features: int = 10,
        name_map: Dict[str, str] = None,
        raw_values: Dict = None,
        contributions: List[Dict] = None
    ) -> plt.Figure:
        """Create SHAP's native waterfall plot with display names and raw values."""
        shap_values = self._prepare_shap_values(data)
        
        # Get preprocessed feature names from the pipeline
        try:
            preprocessed_names = list(
                self.model.pipeline_.named_steps["preprocessor"].get_feature_names_out()
            )
        except:
            preprocessed_names = list(shap_values[0].feature_names)
        
        # Build final display names using the resolver
        final_names = []
        for prep_name in preprocessed_names:
            clean_name = self._clean_feature_name(prep_name)
            display_name = self._resolve_display_name(clean_name, name_map)
            final_names.append(display_name)
        
        # Create raw data array for the plot using the original raw values
        raw_data_array = None
        if raw_values and contributions:
            # We need to reconstruct raw values for ALL preprocessed features, not just contributing ones
            # First, convert the input data properly
            if isinstance(data, dict):
                input_df = pd.DataFrame([data])
            elif isinstance(data, pd.Series):
                input_df = pd.DataFrame([data])
            else:
                input_df = data.copy()
            
            # Ensure all required columns are present
            for col in self.feature_names:
                if col not in input_df.columns:
                    input_df[col] = np.nan
            input_df = input_df[self.feature_names]
            input_raw_values = input_df.iloc[0].to_dict()
            
            # Build raw data array matching the preprocessed feature order
            raw_data_array = []
            for prep_name in preprocessed_names:
                clean_name = self._clean_feature_name(prep_name)
                
                # Determine feature type
                is_missing_indicator = (
                    'missing_indicator' in prep_name.lower() or 
                    clean_name.startswith('MISSING_')
                )
                is_categorical = (
                    prep_name.startswith('cat__') or 
                    clean_name.endswith(('_True', '_False'))
                )
                
                # Get raw value for this preprocessed feature
                raw_val = self._get_raw_value(
                    clean_name, input_raw_values, is_missing_indicator, is_categorical
                )
                raw_data_array.append(raw_val)
            
            raw_data_array = np.array(raw_data_array)
        
        # Create a new Explanation object with display names and raw data
        final_data = raw_data_array if raw_data_array is not None else shap_values[0].data
        
        explanation = shap.Explanation(
            values=shap_values[0].values,
            base_values=shap_values[0].base_values,
            data=final_data,
            feature_names=final_names
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.plots.waterfall(explanation, max_display=max_features, show=False)
        
        return plt.gcf()
    
    def create_force_plot(
        self, 
        data: Union[Dict, pd.Series, pd.DataFrame],
        matplotlib: bool = True,
        name_map: Dict[str, str] = None,
        max_features: int = None
    ) -> Optional[plt.Figure]:
        """Create SHAP force plot with display names and raw values."""
        shap_values = self._prepare_shap_values(data)
        
        # Get preprocessed feature names from the pipeline
        try:
            preprocessed_names = list(
                self.model.pipeline_.named_steps["preprocessor"].get_feature_names_out()
            )
        except:
            preprocessed_names = list(shap_values[0].feature_names)
        
        # Build final display names using the resolver
        final_names = []
        for prep_name in preprocessed_names:
            clean_name = self._clean_feature_name(prep_name)
            display_name = self._resolve_display_name(clean_name, name_map)
            final_names.append(display_name)
        
        values = shap_values[0].values
        base_value = shap_values[0].base_values
        
        # Create raw data array for the force plot (same logic as waterfall plot)
        if isinstance(data, dict):
            input_df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            input_df = pd.DataFrame([data])
        else:
            input_df = data.copy()
        
        # Ensure all required columns are present
        for col in self.feature_names:
            if col not in input_df.columns:
                input_df[col] = np.nan
        input_df = input_df[self.feature_names]
        input_raw_values = input_df.iloc[0].to_dict()
        
        # Build raw data array matching the preprocessed feature order
        raw_data_array = []
        for prep_name in preprocessed_names:
            clean_name = self._clean_feature_name(prep_name)
            
            # Determine feature type
            is_missing_indicator = (
                'missing_indicator' in prep_name.lower() or 
                clean_name.startswith('MISSING_')
            )
            is_categorical = (
                prep_name.startswith('cat__') or 
                clean_name.endswith(('_True', '_False'))
            )
            
            # Get raw value for this preprocessed feature
            raw_val = self._get_raw_value(
                clean_name, input_raw_values, is_missing_indicator, is_categorical
            )
            raw_data_array.append(raw_val)
        
        # For force plots, use SHAP contributions as both values and data labels
        data_values = values.copy()
        
        # Filter to top N features if max_features is set with aggregation
        if max_features is not None and max_features < len(values):
            # Reserve one slot for aggregation if there are remaining features
            top_n = max_features - 1 if max_features < len(values) else max_features
            top_indices = np.argsort(np.abs(values))[-top_n:]
            top_indices = top_indices[np.argsort(values[top_indices])]
            
            # Calculate remaining features aggregation
            remaining_indices = np.setdiff1d(np.arange(len(values)), top_indices)
            if len(remaining_indices) > 0:
                remaining_sum = np.sum(values[remaining_indices])
                remaining_count = len(remaining_indices)
                
                # Add aggregated remaining features:
                # - values: SHAP contributions (bar length/color) 
                # - data: SHAP contributions (displayed as labels)
                aggregated_values = np.append(values[top_indices], remaining_sum)
                aggregated_data = np.append(values[top_indices], remaining_sum)  # Use contributions as labels
                aggregated_names = [final_names[i] for i in top_indices] + [f"{remaining_count} other features"]
                
                values = aggregated_values
                data_values = aggregated_data
                final_names = aggregated_names
            else:
                values = values[top_indices]
                data_values = values[top_indices]  # Use contributions as labels
                final_names = [final_names[i] for i in top_indices]
        else:
            # No filtering needed, use all features
            pass
        
        values = np.round(values, 2)
        base_value = np.round(base_value, 2)
        data_values = np.round(data_values, 2)
        
        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=data_values,
            feature_names=final_names
        )
        
        if matplotlib:
            shap.plots.force(explanation, matplotlib=True, show=False, text_rotation=30)
            fig = plt.gcf()
            fig.set_size_inches(14, 3)
            return fig
        else:
            return shap.plots.force(explanation)
    
    def save_explainer(self, filepath: str):
        """Save the complete explainer to disk."""
        import pickle
        
        explainer_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'explainer': self.explainer,
            'feature_metadata': getattr(self, 'feature_metadata', {}),
            'version': '1.3'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(explainer_data, f)
        
        print(f"Complete explainer saved to {filepath}")
    
    def _load_complete_explainer(self, filepath: str):
        """Load a complete explainer from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            explainer_data = pickle.load(f)
        
        self.model = explainer_data['model']
        self.feature_names = explainer_data['feature_names']
        self.explainer = explainer_data['explainer']
        self.feature_metadata = explainer_data.get('feature_metadata', {})
        
        version = explainer_data.get('version', '1.0')
        metadata_count = len(self.feature_metadata)
        print(
            f"Complete explainer loaded from {filepath} "
            f"(v{version}, {metadata_count} feature descriptions)"
        )
    
    def _predict_function(self, x):
        """Prediction function for SHAP explainer."""
        return self.model.model.predict(x)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Generate SHAP waterfall explanations as text"
    )
    parser.add_argument("--model", type=str, help="Path to saved model file")
    parser.add_argument("--explainer", type=str, help="Path to saved complete explainer file")
    parser.add_argument("--data", type=str, help="JSON file with feature data")
    parser.add_argument("--csv", type=str, help="CSV file with data")
    parser.add_argument("--row", type=int, default=0, help="Row number to explain (if using CSV)")
    parser.add_argument("--features", type=int, default=10, help="Max features to show")
    parser.add_argument("--save-explainer", type=str, help="Save complete explainer to this path")
    
    args = parser.parse_args()
    
    print("Initializing SHAP explainer...")
    explainer = SimpleShapExplainer(args.model, args.explainer)
    
    if args.save_explainer:
        explainer.save_explainer(args.save_explainer)
        return
    
    if args.data:
        with open(args.data, 'r') as f:
            data = json.load(f)
    elif args.csv:
        df = pd.read_csv(args.csv)
        data = df.iloc[args.row].to_dict()
    else:
        print("No data provided, using sample from training data...")
        csv_path = "model_validation/dataset_117_p-all.tsv"
        df = pd.read_csv(csv_path, delimiter="\t", low_memory=False)
        data = df[explainer.feature_names].iloc[0].to_dict()
    
    print("Generating SHAP explanation...\n")
    result = explainer.explain_full(data, max_features=args.features)
    print(result.text_explanation)
    
    if result.waterfall_plot:
        plt.show()
    
    if len(result.contributions_df) > 0:
        print("\n" + "=" * 80)
        print("FEATURE CONTRIBUTIONS TABLE")
        print("=" * 80)
        print(result.contributions_df.to_string(index=False, float_format='%.4f'))
        print("=" * 80)


if __name__ == "__main__":
    main()