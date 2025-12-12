#!/usr/bin/env python3
"""
Pie Pricing Model Explanation App
==================================

A Streamlit application that provides natural language explanations of 
Workers Compensation pricing model predictions using SHAP and Claude AI.

Usage:
    streamlit run model_explainer_app.py
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional

import anthropic
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from model_training.simple_shap_explainer import SimpleShapExplainer, ShapExplanation


# Configuration
EXPLAINER_PATH = "fresh_explainer_with_metadata.pkl"
DATA_DICTIONARY_PATH = "feature_dictionary.csv"
POLICIES_PATH = "/Users/erikallen/Downloads/dataset_130_all.tsv"


class FeatureDictionary:
    """Helper class to manage feature metadata and lookups."""
    
    def __init__(self, csv_path: str):
        """Load and process the feature dictionary."""
        self.df = pd.read_csv(csv_path, low_memory=False)
        self._build_lookup_maps()
    
    def _build_lookup_maps(self):
        """Build lookup maps for quick feature access."""
        self.feature_map = {}
        
        for _, row in self.df.iterrows():
            base_feature = row['Feature']
            self.feature_map[base_feature] = {
                'description': row['Description'],
                'explain': row['Explain'],
                'impact_direction': row['Direction'],
                'short_name': row['Short Name'],
                'formula': row.get('formula', '')
            }
    
    def parse_preprocessed_feature(self, feature_name: str) -> Dict:
        """Parse a preprocessed feature name to extract the base feature and transformation type."""
        clean_name = feature_name
        
        is_missing = False
        if clean_name.startswith('MISSING_'):
            is_missing = True
            clean_name = clean_name.replace('MISSING_', '', 1)
        
        if 'missing_indicator' in clean_name or 'missingindicator' in clean_name:
            is_missing = True
            clean_name = re.sub(r'num__missing_indicator__missingindicator_', '', clean_name)
        
        clean_name = re.sub(r'num__numeric_features__', '', clean_name)
        
        is_categorical = False
        cat_value = None
        if clean_name.startswith('cat__'):
            is_categorical = True
            clean_name = re.sub(r'^cat__', '', clean_name)
            parts = clean_name.rsplit('_', 1)
            if len(parts) == 2:
                clean_name, cat_value = parts
        elif clean_name.endswith('_True') or clean_name.endswith('_False'):
            is_categorical = True
            if clean_name.endswith('_True'):
                cat_value = 'True'
                clean_name = clean_name[:-5]
            else:
                cat_value = 'False'
                clean_name = clean_name[:-6]
        
        return {
            'base_feature': clean_name,
            'is_missing': is_missing,
            'is_categorical': is_categorical,
            'cat_value': cat_value,
            'original': feature_name
        }
    
    def get_feature_info(self, feature_name: str) -> Dict:
        """Get comprehensive information about a feature."""
        parsed = self.parse_preprocessed_feature(feature_name)
        base_feature = parsed['base_feature']
        
        if base_feature in self.feature_map:
            info = self.feature_map[base_feature].copy()
            info.update(parsed)
            return info
        else:
            return {
                'base_feature': base_feature,
                'description': base_feature,
                'explain': base_feature,
                'impact_direction': 'Unknown',
                'short_name': base_feature,
                'formula': '',
                **parsed
            }
    
    def format_feature_for_display(self, feature_name: str) -> str:
        """Format a feature name for user-friendly display."""
        info = self.get_feature_info(feature_name)
        
        if info['is_missing']:
            return f"{info['short_name']} ()"
        elif info['is_categorical']:
            value_display = (
                "Yes" if info['cat_value'] == 'True' 
                else "No" if info['cat_value'] == 'False' 
                else info['cat_value']
            )
            return f"{info['short_name']} = {value_display}"
        else:
            return info['short_name']
    
    def get_feature_explanation(self, feature_name: str) -> str:
        """Get a detailed explanation of what the feature represents."""
        info = self.get_feature_info(feature_name)
        
        explanation = info['explain']
        if info['is_missing']:
            explanation = f"Missing data indicator for {explanation}"
        elif info['is_categorical']:
            explanation = f"{explanation} (value: {info['cat_value']})"
        
        return explanation
    
    def to_prompt_text(self) -> str:
        """Generate formatted text for Claude prompt."""
        lines = ["Feature Data Dictionary:\n"]
        
        for _, row in self.df.iterrows():
            lines.append(f"**{row['Short Name']}** ({row['Feature']})")
            lines.append(f"  - Description: {row['Description']}")
            lines.append(f"  - Business Meaning: {row['Explain']}")
            direction_text = 'increase' if row['Direction'] == 'Positive' else 'decrease'
            lines.append(f"  - Expected Direction: {row['Direction']} (higher values = {direction_text} risk)")
            if pd.notna(row.get('formula')):
                lines.append(f"  - Formula: {row['formula']}")
            lines.append("")
        
        lines.append("\nNote: Features may appear with transformations:")
        lines.append("  - 'Missing Indicator' suffix: Flag indicating this data was missing")
        lines.append("  - Categorical values: Specific category assignments (e.g., 'True' or 'False')")
        lines.append("  - All numeric features are standardized/normalized")
        
        return "\n".join(lines)


@st.cache_resource
def load_explainer():
    """Load the pre-trained SHAP explainer."""
    if Path(EXPLAINER_PATH).exists():
        return SimpleShapExplainer(explainer_path=EXPLAINER_PATH)
    else:
        st.error(f"Explainer not found at {EXPLAINER_PATH}. Please train and save explainer first.")
        return None


@st.cache_data
def load_policies() -> Optional[pd.DataFrame]:
    """Load the policies CSV file at startup."""
    try:
        if not Path(POLICIES_PATH).exists():
            st.error(f"Policies file not found at {POLICIES_PATH}")
            return None
            
        df = pd.read_csv(POLICIES_PATH, delimiter="\t", low_memory=False)
        df = df.loc[df["EX_IS_VALEN"] == 0].reset_index(drop=True)
        df = df.loc[df['POL_BOUND_POLICY_TERM_NUMBER'].isin(["WC 63370 02", "WC PI 533242 00", "WC 122131 00"])]
        
        if 'POL_BOUND_POLICY_TERM_NUMBER' not in df.columns:
            st.warning("No 'policy_number' column found. Using index as policy ID.")
        return df
    except Exception as e:
        st.error(f"Error loading policies: {e}")
        return None


@st.cache_resource
def load_data_dictionary() -> Optional[FeatureDictionary]:
    """Load and process the data dictionary."""
    try:
        return FeatureDictionary(DATA_DICTIONARY_PATH)
    except Exception as e:
        st.warning(f"Could not load data dictionary: {e}")
        return None


def explain_with_claude(
    explanation: ShapExplanation,
    contributions_df: pd.DataFrame,
    feature_dict: Optional[FeatureDictionary],
    policy_data: pd.Series,
    top_n: int = 10
) -> str:
    """Use Claude API to generate a natural language explanation of the model prediction."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
        
    if not api_key:
        return "⚠️ Claude API key not configured. Please add ANTHROPIC_API_KEY to your .env file."
    
    top_features = contributions_df.head(top_n).copy()
    
    if feature_dict:
        top_features['display_name'] = top_features['feature'].apply(
            lambda x: feature_dict.format_feature_for_display(x)
        )
    
    feature_details = []
    for _, row in top_features.head(5).iterrows():
        direction = "increases" if row['contribution'] > 0 else "decreases"
        feature_details.append(
            f"- {row['display_name']}: {direction} risk by {abs(row['contribution']):.4f}"
        )
    
    feature_list = "\n".join(feature_details)
    
    # Get E-Mod value
    emod_value = None
    if 'EXPERIENCE_MOD_FACTOR' in policy_data.index and pd.notna(policy_data['EXPERIENCE_MOD_FACTOR']):
        emod_value = policy_data['EXPERIENCE_MOD_FACTOR']
    
    emod_display = f"{emod_value:.3f}" if emod_value is not None else "N/A"
    
    prompt = f"""Explain this Workers Compensation insurance pricing prediction in 4-6 sentences for underwriters.

Model Output:
- Raw Model Score: {explanation.prediction:.4f}
- Baseline Loss Ratio: {explanation.base_value:.4f}
- Difference: {explanation.risk_delta:+.4f} ({explanation.risk_pct_change:+.1f}%)

Policy E-Mod: {emod_display} (1.0 = average; >1.0 = worse than average claims history; <1.0 = better than average)

Top Contributing Factors (driving the MODEL SCORE, not E-Mod):
{feature_list}

CRITICAL CONTEXT - READ FIRST:
The E-Mod and Model Score are TWO COMPLETELY INDEPENDENT risk assessments:

1. E-MOD reflects ONLY the insured's past claims history compared to similar businesses. It is calculated by NCCI/rating bureaus, not by Pie. A high E-Mod means they've had more/worse claims than average.

2. MODEL SCORE captures EVERYTHING ELSE about risk - business characteristics, class codes, territory, operations, payroll patterns, etc. It does NOT incorporate claims history at all. The model score is Pie's proprietary assessment.

These two scores are MULTIPLIED TOGETHER in pricing. They are not related to each other and will often point in different directions. For example:
- A business with terrible claims history (high E-Mod) might still have favorable operational characteristics (low model score)
- A business with great claims history (low E-Mod) might operate in a risky territory or class (high model score)

Write a 4-6 sentence explanation that:
1. States if the MODEL SCORE indicates higher/lower risk than average and by how much (the percentage shown above is ONLY from the model score)
2. Identifies the 2-3 most important factors from the list above driving the model score
3. Briefly note the E-Mod value as separate context about their claims history
4. If claims-related features don't appear in the top factors, note that claims history is not a concern FOR THE MODEL SCORE (it's handled separately by E-Mod)

CRITICAL WRITING RULES:
- The percentage difference shown ({explanation.risk_pct_change:+.1f}%) is ENTIRELY from the model score. Do not attribute any part of it to E-Mod.
- Do NOT write sentences that combine E-Mod and model score effects (e.g., "Both contribute to the X% improvement" is WRONG)
- Do NOT imply they "add up" or "work together" for any combined effect
- E-Mod is mentioned ONLY as independent context, not as part of explaining the model output
- Keep E-Mod discussion to one brief sentence, separate from the model score discussion

Use conversational business language. Avoid technical jargon. Be direct and concise."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=400,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text
        
    except Exception as e:
        return f"⚠️ Error calling Claude API: {str(e)}"


def build_name_map(feature_dict: Optional[FeatureDictionary], feature_names: list) -> Dict[str, str]:
    """Build a mapping from raw feature names to display names."""
    name_map = {}
    if not feature_dict:
        return name_map
    
    for raw_name in feature_names:
        display = feature_dict.format_feature_for_display(raw_name)
        
        # Fallback logic for common patterns if format returned raw name
        if display == raw_name:
            if raw_name.startswith('MISSING_'):
                base = raw_name.replace('MISSING_', '')
                display = f"{base} ()"
            elif raw_name.endswith('_True'):
                base = raw_name[:-5]
                display = f"{base} = Yes"
            elif raw_name.endswith('_False'):
                base = raw_name[:-6]
                display = f"{base} = No"
        
        name_map[raw_name] = display
    
    return name_map


def safe_format_display(feature_name: str, feature_dict: Optional[FeatureDictionary]) -> str:
    """Safely format a feature name for display."""
    if not feature_dict:
        return feature_name
    
    try:
        display = feature_dict.format_feature_for_display(feature_name)
        if display == feature_name:
            if feature_name.startswith('MISSING_'):
                return f"{feature_name.replace('MISSING_', '')} ()"
            elif '_True' in feature_name:
                return f"{feature_name.replace('_True', '')} = Yes"
            elif '_False' in feature_name:
                return f"{feature_name.replace('_False', '')} = No"
        return display
    except:
        return feature_name


def format_raw_value(value, is_missing: bool, is_categorical: bool) -> str:
    """Format a raw value for display."""
    if is_missing:
        return "Yes" if value else "No"
    elif pd.isna(value):
        return "Missing"
    elif is_categorical:
        return str(value)
    elif isinstance(value, (int, np.integer)):
        return f"{value:,}"
    elif isinstance(value, (float, np.floating)):
        if abs(value) >= 1000:
            return f"{value:,.2f}"
        else:
            return f"{value:.4f}"
    else:
        return str(value)


def display_policy_details(policy_data: pd.Series):
    """Display policy/submission details in a formatted grid."""
    # Define the fields to display: (column_name, display_name, format_func)
    fields = [
        ('PREMIUM_MODIFIED_RATING_PURE_INITIAL', 'Premium', lambda x: f"${x:,.2f}"),
        ('GOVERNING_STATE_CODE_INITIAL', 'State', lambda x: str(x)),
        ('GOVERNING_HAZARD_GROUP_INITIAL', 'Hazard Group', lambda x: f"{x:.0f}"),
        ('GOVERNING_CLASS_CODE_INITIAL', 'Gov Class', lambda x: str(int(x)) if pd.notna(x) else "N/A"),
        ('COUNT_CLASS_NONZERO_PAYROLL_INITIAL', 'Class Count', lambda x: f"{x:.0f}"),
        ('EXPERIENCE_MOD_FACTOR', 'E-Mod', lambda x: f"{x:.3f}"),
        ('CLAIM_COUNT_NON_ZERO_UPTO_3', 'Claim Count (past 3 years)', lambda x: f"{x:.0f}"),
    ]
    
    # Create two rows of metrics
    row1_fields = fields[:4]
    row2_fields = fields[4:]
    
    # First row
    cols = st.columns(len(row1_fields))
    for col, (col_name, display_name, fmt_func) in zip(cols, row1_fields):
        with col:
            if col_name in policy_data.index and pd.notna(policy_data[col_name]):
                st.metric(display_name, fmt_func(policy_data[col_name]))
            else:
                st.metric(display_name, "N/A")
    
    # Second row
    cols = st.columns(len(row2_fields))
    for col, (col_name, display_name, fmt_func) in zip(cols, row2_fields):
        with col:
            if col_name in policy_data.index and pd.notna(policy_data[col_name]):
                st.metric(display_name, fmt_func(policy_data[col_name]))
            else:
                st.metric(display_name, "N/A")


def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Pie Pricing Model Explainer",
        page_icon="🥧",
        layout="wide"
    )
    
    # Initialize session state for caching Claude explanations
    if 'claude_explanation_cache' not in st.session_state:
        st.session_state.claude_explanation_cache = {}
    
    st.title("🥧 Pie Pricing Model Explanation")
    st.markdown("Understanding Workers Compensation Risk Predictions")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        st.subheader("Display Settings")
        top_n_features = st.slider(
            "Number of top features to explain",
            min_value=5,
            max_value=20,
            value=10,
            help="How many features to show in the explanation"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool explains pricing model predictions using:
        - **SHAP values** for feature importance
        - **Claude AI** for natural language explanations
        
        The model predicts Loss Ratio (losses/premium) for Workers Compensation policies.
        """)
    
    # Load data
    policies_df = load_policies()
    if policies_df is None:
        st.error("Failed to load policies data. Please check the POLICIES_PATH configuration.")
        return
    
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Data Info")
        st.metric("Total Policies", len(policies_df))
    
    explainer = load_explainer()
    if explainer is None:
        return
    
    feature_dict = load_data_dictionary()
    if feature_dict is None:
        st.warning("Feature dictionary not loaded. Explanations will use raw feature names.")
    
    # Policy selection
    st.header("Select a Policy or Submission ID")
    
    policy_col = 'POL_BOUND_POLICY_TERM_NUMBER'
    if policy_col not in policies_df.columns:
        policy_col = policies_df.columns[0]
    
    policy_numbers = policies_df[policy_col].astype(str).tolist()
    selected_policy = st.selectbox(
        "Policy Number or Submission ID",
        options=policy_numbers,
        help="Start typing to search for a policy or Submission ID"
    )
    
    if not selected_policy:
        return
    
    policy_data = policies_df[policies_df[policy_col].astype(str) == selected_policy].iloc[0]
    
    # Policy details expander
    with st.expander("📋 Policy or Submission Details", expanded=False):
        display_policy_details(policy_data)
    
    # Generate explanation
    with st.spinner("Analyzing policy/submission and generating explanation..."):
        try:
            name_map = build_name_map(feature_dict, explainer.feature_names)
            
            # Single call to get everything
            explanation = explainer.explain_full(
                policy_data,
                max_features=10,
                name_map=name_map
            )
            
            contributions_df = explanation.contributions_df.copy()
            
            # Add display names
            contributions_df['display_name'] = contributions_df['feature'].apply(
                lambda x: safe_format_display(x, feature_dict)
            )
            
            # Tab selection using radio buttons (persists across rerenders)
            tab_labels = ["🔍 Business Explanation", "💰 Premium Breakdown", "🔬 Data Scientist View"]
            selected_tab = st.radio(
                "View",
                tab_labels,
                horizontal=True,
                key="tab_selector",
                label_visibility="collapsed"
            )
            
            st.markdown("---")
            
            # Business Explanation Tab
            if selected_tab == tab_labels[0]:
                st.header("Model Prediction Explanation")
                
                col1, col2 = st.columns(2)
                with col1:
                    delta_color = "inverse" if explanation.prediction > explanation.base_value else "normal"
                    st.metric(
                        "Raw Model Score",
                        f"{explanation.prediction:.4f}",
                        delta=f"{explanation.risk_delta:+.4f}",
                        delta_color=delta_color
                    )
                with col2:
                    st.metric("vs. Average", f"{explanation.risk_pct_change:+.1f}%")
                
                st.markdown("---")
                
                # Cached Claude explanation
                cache_key = f"{selected_policy}_{top_n_features}"
                
                if cache_key not in st.session_state.claude_explanation_cache:
                    with st.spinner("Generating explanation with Claude AI..."):
                        claude_explanation = explain_with_claude(
                            explanation=explanation,
                            contributions_df=contributions_df,
                            feature_dict=feature_dict,
                            policy_data=policy_data,
                            top_n=top_n_features
                        )
                        st.session_state.claude_explanation_cache[cache_key] = claude_explanation
                else:
                    claude_explanation = st.session_state.claude_explanation_cache[cache_key]
                
                st.markdown(claude_explanation)
                
                st.markdown("---")
                st.subheader("Key Contributing Factors")
                
                display_df = contributions_df.head(top_n_features)[[
                    'rank', 'display_name', 'contribution', 'impact_direction'
                ]].copy()
                display_df.columns = ['Rank', 'Factor', 'Impact on Risk', 'Effect']
                
                st.dataframe(
                    display_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Impact on Risk": st.column_config.NumberColumn(format="%.4f")
                    }
                )
            
            # Premium Breakdown Tab
            elif selected_tab == tab_labels[1]:
                st.header("Premium Calculation Walkthrough")
                st.markdown("Understanding how your premium is built, step by step.")
                
                # Extract values
                exposure = policy_data.get('EXPOSURE_AMOUNT_INITIAL', None)
                loss_cost = policy_data.get('COMBINED_WEIGHTED_LOSS_COST_INITIAL', None)
                manual_premium = policy_data.get('PREMIUM_RATING_PURE_INITIAL', None)
                emod = policy_data.get('EXPERIENCE_MOD_FACTOR', None)
                mrpp = policy_data.get('PREMIUM_MODIFIED_RATING_PURE_INITIAL', None)
                written_premium = policy_data.get('PREMIUM_MODIFIED_INITIAL', None)
                model_score = explanation.prediction
                
                # Step 1: Rating Pure Premium
                st.markdown("---")
                st.subheader("Step 1: Rating Pure Premium")
                
                col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
                
                with col1:
                    st.metric("Exposure", f"{exposure:,.0f}" if pd.notna(exposure) else "N/A")
                    st.caption("Payroll / $100")
                
                with col2:
                    st.markdown("<div style='text-align: center; font-size: 2rem; padding-top: 1rem;'>×</div>", unsafe_allow_html=True)
                
                with col3:
                    st.metric("Combined Weighted Loss Cost", f"${loss_cost:.4f}" if pd.notna(loss_cost) else "N/A")
                    st.caption("Rate per $100 of payroll")
                
                with col4:
                    st.markdown("<div style='text-align: center; font-size: 2rem; padding-top: 1rem;'>=</div>", unsafe_allow_html=True)
                
                with col5:
                    calculated_manual = exposure * loss_cost if pd.notna(exposure) and pd.notna(loss_cost) else None
                    st.metric("Rating Pure Premium", f"${manual_premium:,.2f}" if pd.notna(manual_premium) else "N/A")
                    if calculated_manual and manual_premium:
                        if abs(calculated_manual - manual_premium) < 1:
                            st.caption("✓ Calculation verified")
                        else:
                            st.caption(f"Calculated: ${calculated_manual:,.2f}")
                
                # Step 2: MRPP
                st.markdown("---")
                st.subheader("Step 2: Modified Rating Pure Premium (MRPP)")
                
                col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
                
                with col1:
                    st.metric("Rating Pure Premium", f"${manual_premium:,.2f}" if pd.notna(manual_premium) else "N/A")
                
                with col2:
                    st.markdown("<div style='text-align: center; font-size: 2rem; padding-top: 1rem;'>×</div>", unsafe_allow_html=True)
                
                with col3:
                    emod_display = f"{emod:.3f}" if pd.notna(emod) else "N/A"
                    if pd.notna(emod):
                        if emod > 1.0:
                            emod_label = "E-Mod (Above Average)"
                            delta = f"+{(emod - 1) * 100:.1f}%"
                            delta_color = "inverse"
                        elif emod < 1.0:
                            emod_label = "E-Mod (Below Average)"
                            delta = f"{(emod - 1) * 100:.1f}%"
                            delta_color = "normal"
                        else:
                            emod_label = "E-Mod (Average)"
                            delta = None
                            delta_color = "off"
                        st.metric(emod_label, emod_display, delta=delta, delta_color=delta_color)
                    else:
                        st.metric("E-Mod", "N/A")
                    st.caption("Experience modification factor")
                
                with col4:
                    st.markdown("<div style='text-align: center; font-size: 2rem; padding-top: 1rem;'>=</div>", unsafe_allow_html=True)
                
                with col5:
                    calculated_mrpp = manual_premium * emod if pd.notna(manual_premium) and pd.notna(emod) else None
                    st.metric("MRPP", f"${mrpp:,.2f}" if pd.notna(mrpp) else "N/A")
                    if calculated_mrpp and mrpp:
                        if abs(calculated_mrpp - mrpp) < 1:
                            st.caption("✓ Calculation verified")
                        else:
                            st.caption(f"Calculated: ${calculated_mrpp:,.2f}")
                
                # Step 3: Written Premium
                st.markdown("---")
                st.subheader("Step 3: Written Premium")
                
                col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
                
                with col1:
                    st.metric("MRPP", f"${mrpp:,.2f}" if pd.notna(mrpp) else "N/A")
                
                with col2:
                    st.markdown("<div style='text-align: center; font-size: 2rem; padding-top: 1rem;'>×</div>", unsafe_allow_html=True)
                
                with col3:
                    if model_score > 1.0:
                        score_label = "Model Score (Higher Risk)"
                        delta = f"+{(model_score - 1) * 100:.1f}%"
                        delta_color = "inverse"
                    elif model_score < 1.0:
                        score_label = "Model Score (Lower Risk)"
                        delta = f"{(model_score - 1) * 100:.1f}%"
                        delta_color = "normal"
                    else:
                        score_label = "Model Score (Average)"
                        delta = None
                        delta_color = "off"
                    st.metric(score_label, f"{model_score:.4f}", delta=delta, delta_color=delta_color)
                    st.caption("Pie's predictive model adjustment")
                
                with col4:
                    st.markdown("<div style='text-align: center; font-size: 2rem; padding-top: 1rem;'>≈</div>", unsafe_allow_html=True)
                
                with col5:
                    st.metric("Written Premium", f"${written_premium:,.2f}" if pd.notna(written_premium) else "N/A")
                    if pd.notna(mrpp) and pd.notna(written_premium):
                        calculated_written = mrpp * model_score
                        st.caption(f"Estimated: ${calculated_written:,.2f}")
                        st.caption("*(May differ due to model transforms)*")
                
                # Summary box
                st.markdown("---")
                st.subheader("📊 Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown("**The Formula:**")
                    st.code("Written Premium ≈ Exposure × Loss Cost × E-Mod × Model Score", language=None)
                    
                with summary_col2:
                    st.markdown("**Your Numbers:**")
                    if all(pd.notna(x) for x in [exposure, loss_cost, emod, model_score]):
                        st.code(f"${written_premium:,.2f} ≈ {exposure:,.0f} × ${loss_cost:.4f} × {emod:.3f} × {model_score:.4f}", language=None)
                    else:
                        st.code("Some values unavailable", language=None)
                
                # Explanation
                with st.expander("ℹ️ Understanding the Components", expanded=False):
                    st.markdown("""
                    **Exposure** — Your total payroll divided by $100. This is the base unit for calculating workers' comp premiums.
                    
                    **Governing Class Loss Cost** — The expected loss rate for your primary class code, set by rating bureaus based on industry-wide claims data.
                    
                    **E-Mod (Experience Modification Factor)** — Reflects your company's actual claims history compared to similar businesses. 
                    - E-Mod = 1.00 means average claims experience
                    - E-Mod > 1.00 means worse than average (increases premium)
                    - E-Mod < 1.00 means better than average (decreases premium)
                    
                    **Model Score** — Pie's proprietary risk assessment based on many factors beyond just claims history. This is applied on top of the E-Mod to capture additional risk signals.
                    
                    **Note:** The final written premium may not exactly match the calculation because the model score undergoes additional transformations before being applied.
                    """)
            
            # Data Scientist View Tab
            elif selected_tab == tab_labels[2]:
                st.header("SHAP Technical Analysis")
                
                with st.expander("📊 Text Summary", expanded=False):
                    st.text(explanation.text_explanation)
                
                st.markdown("---")
                
                # SHAP Visualizations
                st.subheader("SHAP Visualizations")
                
                # Force Plot
                st.markdown("#### Force Plot")
                st.markdown("*Shows how each feature pushes the prediction from the base value.*")
                try:
                    force_plot = explainer.create_force_plot(
                        policy_data, matplotlib=True, name_map=name_map, max_features=5
                    )
                    if force_plot is not None:
                        st.pyplot(force_plot)
                    else:
                        st.info("💡 Force plot not available.")
                except Exception as e:
                    st.warning(f"Could not generate force plot: {e}")
                
                st.markdown("---")
                
                # Waterfall Plot
                st.markdown("#### Waterfall Plot")
                st.markdown("*Shows the cumulative contribution of each feature.*")
                if explanation.waterfall_plot is not None:
                    st.pyplot(explanation.waterfall_plot)
                else:
                    st.info("💡 Waterfall plot not available.")
                
                st.markdown("---")
                
                # Full contributions table with raw and transformed values
                st.subheader("All Feature Contributions")
                
                # Format raw values for display
                contributions_df['raw_value_display'] = contributions_df.apply(
                    lambda row: format_raw_value(
                        row['raw_value'], 
                        row['is_missing_indicator'], 
                        row['is_categorical']
                    ),
                    axis=1
                )
                
                technical_df = contributions_df[[
                    'rank', 'display_name', 'feature', 
                    'raw_value_display', 'transformed_value', 
                    'contribution', 'impact_direction'
                ]].copy()
                technical_df.columns = [
                    'Rank', 'Display Name', 'Technical Name', 
                    'Raw Value', 'Transformed', 
                    'SHAP Impact', 'Direction'
                ]
                
                st.dataframe(
                    technical_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Transformed": st.column_config.NumberColumn(format="%.4f"),
                        "SHAP Impact": st.column_config.NumberColumn(format="%.6f"),
                    }
                )
                
                st.markdown("---")
                
                # Feature dictionary lookup - FIXED VERSION
                st.subheader("Feature Dictionary Lookup")
                if feature_dict:
                    # Build display name to raw name mapping
                    display_to_raw = {
                        safe_format_display(f, feature_dict): f 
                        for f in contributions_df['feature'].tolist()
                    }
                    display_names = list(display_to_raw.keys())
                    
                    selected_display = st.selectbox(
                        "Select a feature to see details",
                        options=display_names,
                        key="feature_lookup_selectbox"
                    )
                    
                    if selected_display:
                        # Map back to raw feature name
                        selected_feature = display_to_raw[selected_display]
                        info = feature_dict.get_feature_info(selected_feature)
                        
                        # Get the row for this feature
                        feature_row = contributions_df[
                            contributions_df['feature'] == selected_feature
                        ].iloc[0]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Display Name:** {selected_display}")
                            st.markdown(f"**Technical Name:** `{selected_feature}`")
                            st.markdown(f"**Base Feature:** {info['base_feature']}")
                            st.markdown(f"**Raw Value:** {feature_row['raw_value_display']}")
                            st.markdown(f"**Transformed Value:** {feature_row['transformed_value']:.4f}")
                        with col2:
                            st.markdown(f"**Expected Direction:** {info['impact_direction']}")
                            st.markdown(f"**Missing Indicator:** {info['is_missing']}")
                            st.markdown(f"**Categorical:** {info['is_categorical']}")
                            st.markdown(f"**SHAP Impact:** {feature_row['contribution']:.6f}")
                        
                        st.markdown(f"**Description:** {info['description']}")
                        st.markdown(f"**Business Meaning:** {info['explain']}")
                        if info['formula']:
                            st.markdown(f"**Formula:** `{info['formula']}`")
                
        except Exception as e:
            st.error(f"Error generating explanation: {str(e)}")
            with st.expander("Show error details"):
                st.exception(e)


if __name__ == "__main__":
    main()