#!/usr/bin/env python3
"""
Create a fresh SHAP explainer with properly trained model and feature metadata.
"""

import pandas as pd
from model_training.PipelineModel import PipelineRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from model_training.simple_shap_explainer import SimpleShapExplainer

def main():
    print("Creating fresh SHAP explainer with properly trained model...")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv('./model_validation/dataset_117_p-all.tsv', sep='\t', low_memory=False)
    
    # Select features (using the same logic as in ppm_example.py)
    features_path = './model_validation/LH-Model.csv'
    selected_features = pd.read_csv(features_path, header=None, low_memory=False)
    feature_names = [f for f in selected_features[0] 
                    if f in df.columns and f != "FTR_PREMIUM_MODIFIED_RATING_PURE_POST_AUDIT"]
    
    # Prepare data
    X = df[feature_names]
    y = df['FTR_LOSS_RATIO_REL_MRPP_C250_ADJ']
    
    print(f"Training model with {len(feature_names)} features...")
    
    # Train model (same as in ppm_example.py)
    model = PipelineRegressor(
        model=Ridge(alpha=0.1, max_iter=1000, tol=1e-4, solver='auto')
    )
    model.fit(X, y)
    
    print("Model training complete. Saving model...")
    
    # Save the trained model
    import pickle
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Creating SHAP explainer...")
    
    # Create explainer with trained model
    explainer = SimpleShapExplainer(model_path='trained_model.pkl')
    
    # Save the complete explainer
    explainer.save_explainer('fresh_explainer_with_metadata.pkl')
    
    print("Fresh explainer created and saved successfully!")
    
    # Test it with a sample
    print("\nTesting explainer with sample data...")
    sample_data = X.iloc[0:1]
    text, df_contrib, plot = explainer.explain_full(sample_data, max_features=5)
    
    print("\nSample explanation (first 15 lines):")
    print('\n'.join(text.split('\n')[:15]))
    
    print(f"\nMetadata loaded for {len(explainer.feature_metadata)} features")
    print("Sample feature descriptions:")
    for i, (feat, meta) in enumerate(list(explainer.feature_metadata.items())[:3]):
        print(f"  {feat}: {meta['short_name']} - {meta['explain'][:50]}...")
    
    print("\nFresh explainer is ready to use!")

if __name__ == "__main__":
    main()