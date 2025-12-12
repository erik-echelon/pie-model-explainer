import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('./data/sample_transform_details.csv')
    return df

def create_waterfall_chart(df, submission_id=None):
    """
    Create a waterfall chart showing how transforms affect the score
    """
    if submission_id:
        df = df[df['SUBMISSION_ID'] == submission_id].copy()
    
    # Sort by transform index to ensure proper order
    df = df.sort_values('TRANSFORM_INDEX')
    
    # Calculate the delta for each transform
    df['DELTA'] = df['SCORE_RESULT'] - df['SCORE_INPUT']
    
    # Prepare data for waterfall chart
    transforms = ['Initial'] + df['TRANSFORM_TYPE'].fillna('Initial').tolist()
    values = [df.iloc[0]['SCORE_INPUT']] + df['DELTA'].tolist()
    
    # Create measures list (relative for changes, total for final)
    measures = ['absolute'] + ['relative'] * (len(df))
    
    # Create the waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Score Transform",
        orientation="v",
        measure=measures,
        x=transforms,
        textposition="outside",
        text=[f"{v:.4f}" for v in [df.iloc[0]['SCORE_INPUT']] + df['SCORE_RESULT'].tolist()],
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#EF553B"}},
        increasing={"marker": {"color": "#00CC96"}},
        totals={"marker": {"color": "#636EFA"}}
    ))
    
    fig.update_layout(
        title="Score Transformation Journey",
        xaxis_title="Transform Step",
        yaxis_title="Score Value",
        showlegend=False,
        height=600,
        xaxis={'tickangle': -45}
    )
    
    return fig

def create_line_chart(df, submission_id=None):
    """
    Create a line chart showing score progression through transforms
    """
    if submission_id:
        df = df[df['SUBMISSION_ID'] == submission_id].copy()
    
    # Sort by transform index
    df = df.sort_values('TRANSFORM_INDEX')
    
    # Create step labels
    df['Step'] = df['TRANSFORM_INDEX'].astype(str) + ': ' + df['TRANSFORM_TYPE'].fillna('Initial')
    
    fig = go.Figure()
    
    # Add the line trace
    fig.add_trace(go.Scatter(
        x=df['TRANSFORM_INDEX'],
        y=df['SCORE_RESULT'],
        mode='lines+markers',
        name='Score',
        line=dict(color='#636EFA', width=3),
        marker=dict(size=10, color='#636EFA'),
        customdata=df[['TRANSFORM_TYPE', 'SCORE_INPUT', 'SCORE_RESULT']],
        hovertemplate='<b>Step %{x}: %{customdata[0]}</b><br>' +
                      'Input: %{customdata[1]:.4f}<br>' +
                      'Output: %{customdata[2]:.4f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add annotations for significant changes
    for idx, row in df.iterrows():
        delta = row['SCORE_RESULT'] - row['SCORE_INPUT']
        if abs(delta) > 0.01:  # Only annotate significant changes
            fig.add_annotation(
                x=row['TRANSFORM_INDEX'],
                y=row['SCORE_RESULT'],
                text=f"{delta:+.4f}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="green" if delta > 0 else "red",
                ax=0,
                ay=-40 if delta > 0 else 40,
                font=dict(size=10, color="green" if delta > 0 else "red")
            )
    
    fig.update_layout(
        title="Score Evolution Through Transforms",
        xaxis_title="Transform Index",
        yaxis_title="Score Value",
        height=600,
        hovermode='closest',
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        )
    )
    
    return fig

def create_bar_chart(df, submission_id=None):
    """
    Create a bar chart showing the impact of each transform
    """
    if submission_id:
        df = df[df['SUBMISSION_ID'] == submission_id].copy()
    
    # Sort by transform index
    df = df.sort_values('TRANSFORM_INDEX')
    
    # Calculate the delta for each transform
    df['DELTA'] = df['SCORE_RESULT'] - df['SCORE_INPUT']
    df['Impact'] = df['DELTA'].abs()
    df['Direction'] = df['DELTA'].apply(lambda x: 'Increase' if x > 0 else ('Decrease' if x < 0 else 'No Change'))
    
    # Filter out transforms with no impact
    df_impact = df[df['Impact'] > 0].copy()
    
    fig = go.Figure()
    
    # Add bars colored by direction
    colors = df_impact['DELTA'].apply(lambda x: '#00CC96' if x > 0 else '#EF553B')
    
    fig.add_trace(go.Bar(
        x=df_impact['TRANSFORM_TYPE'],
        y=df_impact['DELTA'],
        marker_color=colors,
        text=[f"{v:+.4f}" for v in df_impact['DELTA']],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>' +
                      'Impact: %{text}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="Transform Impact Analysis",
        xaxis_title="Transform Type",
        yaxis_title="Score Change",
        height=600,
        xaxis={'tickangle': -45},
        showlegend=False
    )
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def main():
    st.set_page_config(page_title="Score Transform Visualization", layout="wide")
    
    st.title("📊 Score Transformation Story")
    st.markdown("### Understanding How Transforms Impact Your Score")
    
    # Load data directly from file
    try:
        df = load_data()
        
        # Sidebar for filtering
        st.sidebar.header("Filters")
        
        # Select submission ID if multiple exist
        submission_ids = df['SUBMISSION_ID'].unique()
        if len(submission_ids) > 1:
            selected_submission = st.sidebar.selectbox(
                "Select Submission ID",
                submission_ids
            )
        else:
            selected_submission = submission_ids[0]
        
        # Filter data
        df_filtered = df[df['SUBMISSION_ID'] == selected_submission].copy()
        df_filtered = df_filtered.sort_values('TRANSFORM_INDEX')
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        initial_score = df_filtered.iloc[0]['SCORE_INPUT']
        final_score = df_filtered.iloc[-1]['SCORE_RESULT']
        total_change = final_score - initial_score
        num_transforms = len(df_filtered)
        
        col1.metric("Initial Score", f"{initial_score:.4f}")
        col2.metric("Final Score", f"{final_score:.4f}")
        col3.metric("Total Change", f"{total_change:+.4f}", delta=f"{(total_change/initial_score)*100:+.2f}%")
        col4.metric("Transforms Applied", num_transforms)
        
        # Visualization type selector
        viz_type = st.radio(
            "Choose Visualization Style",
            ["Waterfall Chart", "Line Chart", "Impact Analysis"],
            horizontal=True
        )
        
        st.markdown("---")
        
        # Display selected visualization
        if viz_type == "Waterfall Chart":
            st.markdown("**Waterfall charts** show the cumulative effect of each transform, making it easy to see which transforms increase or decrease the score.")
            fig = create_waterfall_chart(df, selected_submission)
            st.plotly_chart(fig, use_container_width=True)
            
        elif viz_type == "Line Chart":
            st.markdown("**Line charts** display the score progression through each transform step, with annotations highlighting significant changes.")
            fig = create_line_chart(df, selected_submission)
            st.plotly_chart(fig, use_container_width=True)
            
        else:  # Impact Analysis
            st.markdown("**Impact analysis** focuses on transforms that actually changed the score, showing their relative magnitude.")
            fig = create_bar_chart(df, selected_submission)
            st.plotly_chart(fig, use_container_width=True)
        
        # Show detailed table
        with st.expander("📋 View Detailed Transform Data"):
            display_df = df_filtered[['TRANSFORM_INDEX', 'TRANSFORM_TYPE', 'SCORE_INPUT', 'SCORE_RESULT']].copy()
            display_df['DELTA'] = display_df['SCORE_RESULT'] - display_df['SCORE_INPUT']
            display_df['PERCENT_CHANGE'] = (display_df['DELTA'] / display_df['SCORE_INPUT'] * 100).round(2)
            
            st.dataframe(
                display_df.style.format({
                    'SCORE_INPUT': '{:.6f}',
                    'SCORE_RESULT': '{:.6f}',
                    'DELTA': '{:+.6f}',
                    'PERCENT_CHANGE': '{:+.2f}%'
                }),
                use_container_width=True
            )
    
    except FileNotFoundError:
        st.error("❌ Could not find './data/sample_transform_details.csv'. Please ensure the file exists in the correct location.")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")

if __name__ == "__main__":
    main()