import streamlit as st
import pandas as pd
import plotly.express as px
import re
import json
import numpy as np
import base64
from io import BytesIO
from llama_index.core import PromptTemplate
from llama_index.core.agent import ReActAgent
from llama_index.llms.groq import Groq
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
    
    llm_choice = st.sidebar.selectbox("Select AI Model", ["Groq (Llama3-70B)", "OpenAI (GPT-4o)"])
    api_key = st.sidebar.text_input("Enter API Key", type="password")
    
    if uploaded_file is not None and api_key:
        df = pd.read_csv(uploaded_file)
        df, column_types = preprocess_data(df)
        
        st.sidebar.success("✅ Dataset uploaded successfully!")
        x_axis = st.sidebar.selectbox("Select X-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select Y-axis", df.columns)
        chart_type = st.sidebar.selectbox("Select Chart Type", ["bar", "line", "scatter", "histogram", "pie", "box"])
        user_prompt = st.sidebar.text_area("💬 Custom AI Prompt (Optional)", "Analyze the provided data and chart to generate insights.")
        generate_button = st.sidebar.button("🚀 Generate Visualization & Insights")
        
        if generate_button:
            st.subheader("📈 Visualization")
            try:
                # Create the Plotly figure based on the selected chart type
                fig = px.__getattribute__(chart_type)(df, x=x_axis, y=y_axis, title=f'{chart_type.capitalize()} Visualization')
                fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error generating chart: {e}")
                return
            
            # Convert the figure to a PNG image and encode it as a Base64 string
            try:
                # Obtain image bytes from the figure
                img_bytes = fig.to_image(format="png")
                # Encode the image bytes to Base64
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
            except Exception as e:
                st.error(f"Error converting chart to image: {e}")
                return

            # Initialize LLM based on selection
            llm = Groq(model="llama3-70b-8192", api_key=api_key) if "Groq" in llm_choice else OpenAI(model="gpt-4o-mini", api_key=api_key)
            
            # Append the Base64 encoded image to the AI prompt
            ai_prompt = f"""
                You are an AI specialized in marketing analytics. Given the dataset and the generated visualization:
                - Identify key trends in '{x_axis}' and '{y_axis}'.
                - Provide actionable marketing insights based on the chart.
                - Analyze anomalies, patterns, seasonal variations, and customer behavior.
                - Ensure insights are specific to the provided dataset and visualization.
                {user_prompt}
                The visualization image (Base64 encoded PNG): {img_b64}
            """
            
            agent = ReActAgent.from_tools([], llm=llm, verbose=True)
            response = agent.chat(ai_prompt)
            
            # Extract insights
            insights_text = response.response if response.response else "No insights provided by AI."
            
            st.subheader("💡 AI-Generated Insights")
            st.write(insights_text)
            
            # Show Python Code Button
            if st.button("📜 Show Python Code"):
                python_code = f'''
                import plotly.express as px
                import base64
                # Generate the chart
                fig = px.{chart_type}(df, x="{x_axis}", y="{y_axis}", title="{chart_type.capitalize()} Visualization")
                fig.update_layout(xaxis_title="{x_axis}", yaxis_title="{y_axis}")
                fig.show()
                # Convert the chart to a Base64 encoded image
                img_bytes = fig.to_image(format="png")
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                print("Base64 Encoded Chart Image:", img_b64)
                '''
                st.code(python_code, language='python')
    else:
        st.info("Upload a dataset and enter an API key to proceed.")

if __name__ == "__main__":
    main()
