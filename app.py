import streamlit as st
import plotly.express as px
import pandas as pd

def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.dropna(axis=1, how='all')
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
        cat1 = st.selectbox("X-axis", df.columns.tolist())
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
        st.info("Heatmap to be implemented.")
        cat2 = st.selectbox("X-axis", df.columns.tolist(), key="tab2_cat")
        stack2 = st.selectbox("Color", df.columns.tolist(), key="tab2_stack")
        val2 = st.selectbox("Y-value", df.select_dtypes(include=['number']).columns.tolist(), key="tab2_val")


    with tab3:
        st.info("Waterdrop chart to be implemented.")
        cat3 = st.selectbox("X-axis", df.columns.tolist(), key="tab3_cat")
        stack3 = st.selectbox("Color", df.columns.tolist(), key="tab3_stack")
        val3 = st.selectbox("Y-value", df.select_dtypes(include=['number']).columns.tolist(), key="tab3_val")


    with tab4:
        years= sorted(df['Period'].unique())
        if len(years) > 1:
            selected_year = st.slider("Select Year", min_value=years[0], max_value=years[-1],step=1 )
        else: 
           selected_year = years[0]

        df_year = df[df['Period'] == selected_year].copy()

        df_mapped = df_year.groupby('SpatialDimValueCode')['FactValueNumeric'].mean().reset_index()
        df_mapped.columns = ['SpatialDimValueCode', 'FactValueNumeric']

        fig_map = px.choropleth(
            df_mapped,
            locations="SpatialDimValueCode",
            color="FactValueNumeric",
            title=f"Water accessibilty for Year {selected_year}",
            hover_name="SpatialDimValueCode",
            color_continuous_scale="RdYlGn",
            labels={"FactValueNumeric": "Access (%)"}
        )

        st.subheader("Choropleth Map")
        st.plotly_chart(fig_map, use_container_width=True)

        st.write("Countries for map:", df_mapped[:56])  # Debugging


  except Exception as e:
    st.error(f"Error: {e}")
else:
    st.info("Upload Data_water.csv in the sidebar to see the chart (uses ParentLocation as category, Dim1 as stack by default).")