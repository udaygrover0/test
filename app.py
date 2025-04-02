import streamlit as st
import pandas as pd
import plotly.express as px
import re
import json
import numpy as np
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI

def detect_column_types(df):
    """Dynamically detect column types."""
    column_types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]) or any(
            isinstance(x, str) and re.search(r'\d{4}-\d{2}-\d{2}', str(x)) for x in df[col].dropna()
        ):
            df[col] = pd.to_datetime(df[col], errors='coerce')
            column_types[col] = "datetime"
        else:
            column_types[col] = "categorical"
    return column_types

def preprocess_data(df):
    """Handle missing values and convert data types dynamically."""
    df = df.fillna(0)
    column_types = detect_column_types(df)
    return df, column_types

def main():
    st.set_page_config(layout="wide")
    st.title("📊 Marketing Data Visualizer with AI Insights")
    
    # Sidebar for input selections
    st.sidebar.header("1️⃣ Upload & Configure")
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    
    llm_choice = st.sidebar.selectbox("Select AI Model", ["OpenAI (GPT-4)"])  # Using OpenAI here
    api_key = st.sidebar.text_input("Enter API Key", type="password")
    
    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)
        
        st.sidebar.success("✅ Dataset uploaded successfully!")
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        chart_type = st.sidebar.selectbox("Select Chart Type", ["bar", "line", "scatter", "histogram", "pie", "box"])
        user_prompt = st.sidebar.text_area("💬 Custom AI Prompt (Optional)", "Analyze the provided data and chart to generate actionable marketing insights.")
        generate_button = st.sidebar.button("🚀 Generate Visualization & Insights")
        
        if generate_button:
            st.subheader("📈 Visualization")
            try:
                fig = getattr(px, chart_type)(df, x=x_axis, y=y_axis, title=f'{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                return
            
            # Initialize LLM explicitly using OpenAI
            llm = OpenAI(model="gpt-4", api_key=api_key)
            
            # Construct AI prompt with clear instructions and a final directive to output the final answer.
            ai_prompt = f"""
You are an AI specialized in marketing analytics. Based on the visualization of the data (chart with x-axis: {x_axis} and y-axis: {y_axis}), provide clear, actionable insights. 
Analyze trends, anomalies, and key performance indicators.
Finally, provide your final answer.
{user_prompt}
"""
            # Increase max iterations to give the agent more room if needed (adjust as supported by your version)
            agent = ReActAgent.from_tools([], llm=llm, verbose=True, max_iterations=10)
            
            try:
                response = agent.chat(ai_prompt)
                insights_text = response.response if response.response else "No insights provided by AI."
            except ValueError as err:
                insights_text = f"Error: {err}"
            
            st.subheader("💡 AI-Generated Insights")
            st.write(insights_text)
            
            if st.button("📜 Show Python Code"):
                python_code = f"""
import plotly.express as px
fig = px.{chart_type}(df, x='{x_axis}', y='{y_axis}', title='{chart_type.capitalize()} Visualization')
fig.update_layout(xaxis_title='{x_axis}', yaxis_title='{y_axis}')
fig.show()
"""
                st.code(python_code, language='python')
    else:
        st.info("Upload a dataset and enter an API key to proceed.")

if __name__ == "__main__":
    main()
