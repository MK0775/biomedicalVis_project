import streamlit as st
import plotly.express as px
import pandas as pd

def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["FactValueNumeric"] = pd.to_numeric(df["FactValueNumeric"], errors="coerce")  # fix value column[file:16]
    
    # Filter latest + select YOUR columns[file:16]
    df_clean = df[df["IsLatestYear"] == "true"].dropna(subset=["FactValueNumeric"])
    df_clean = df_clean[["ParentLocation", "Location", "Dim1", "FactValueNumeric"]].copy()
    df_clean.columns = ["category", "sub_category", "stack_col", "value"]  # rename for plot
    return df_clean


st.set_page_config(page_title="Stacked Bar Example", layout="wide")

st.title("Interactive Stacked Bar Chart")

# Sidebar file input
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
  try:
    # Clean data
    df = load_and_clean_data(uploaded_file)

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())

    # Optional: allow user to pick columns (with defaults for your data)
    category_col = st.sidebar.selectbox(
        "Category (x-axis)", options=df.columns, index=df.columns.get_loc("category")
    )
    subcategory_col = st.sidebar.selectbox(
        "Stack (color)", options=df.columns, index=df.columns.get_loc("stack_col")
    )
    value_col = st.sidebar.selectbox(
        "Value (height)", options=df.columns[df.columns.str.contains("value")], index=0
    )

    # Stacked bar chart
    fig = px.bar(
        df,
        x=category_col,
        y=value_col,
        color=subcategory_col,
        barmode="stack", 
        hover_data=df.columns, 
        title="Stacked Bar Chart (ParentLocation vs Location/Dim1)",
    )

    # Better layout
    fig.update_layout(
        xaxis_title=category_col,
        yaxis_title=value_col,
        legend_title=subcategory_col,
        height=600
    )

    # Display interactive chart in browser
    st.subheader("Stacked Bar Visualization")
    st.plotly_chart(fig, use_container_width=True)

  except Exception as e:
    st.error(f"Error: {e}")
else:
    st.info("Upload Data_water.csv in the sidebar to see the chart (uses ParentLocation as category, Dim1 as stack by default).")