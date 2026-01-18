import streamlit as st
import plotly.express as px
import pandas as pd

def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df["FactValueNumeric"] = pd.to_numeric(df["FactValueNumeric"], errors="coerce")
    
    df["IsLatestYear"] = df["IsLatestYear"].astype(str).str.strip().str.lower()
    df_clean = df[df["IsLatestYear"] == "true"].dropna(subset=["FactValueNumeric"])
    

    return df_clean


st.set_page_config(page_title="Stacked Bar Example", layout="wide")

st.title("Interactive Charts")

# Sidebar file input
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
  try:
    # Clean data
    df = load_and_clean_data(uploaded_file)

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())


    tab1, tab2, tab3, tab4 = st.tabs(["Stacked bar", "Heatmap Matrix", "Waterdrop", "About"])
    with tab1:
    # Stacked bar 
        cat1 = st.selectbox("X-axis", df.columns.tolist)
        stack1 = st.selectbox("Color", df.columns.tolist())
        val1 = st.selectbox("Y-value", df.select_dtypes(include=['number']).columns.tolist())

        fig = px.bar(
            df,
            x=cat1,
            y=val1,
            color=stack1,
            barmode="stack", 
            hover_data=df.columns, 
            title=f"Plotting: {cat1} vs {stack1} / {val1}"        
            )

        # Better layout
        fig.update_layout(
            xaxis_title="regions",
            yaxis_title="value in %",
            legend_title="colour legend",
            height=600,
            yaxis=dict(range=[0, 100], tickmode='linear', dtick=10),  
            barnorm='percent'  
        )

        # Display interactive chart in browser
        st.subheader("Stacked Bar Visualization")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Cleaned Data")
        st.dataframe(df)

    with tab3:
        st.subheader("Cleaned Data")
        st.dataframe(df)
    with tab4:
        st.subheader("Cleaned Data")
        st.dataframe(df)
  except Exception as e:
    st.error(f"Error: {e}")
else:
    st.info("Upload Data_water.csv in the sidebar to see the chart (uses ParentLocation as category, Dim1 as stack by default).")