
import streamlit as st
import plotly.express as px
import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath

 
alt.data_transformers.disable_max_rows()

def load_and_clean_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.dropna(axis=1, how='all')

    # Normalisierung der Ländernamen zu "Title Case" (z.B. AUSTRIA -> Austria)
    if 'Location' in df.columns:
        df['Location'] = df['Location'].astype(str).str.strip().str.title()
    # numeric data cleaning
    df["FactValueNumeric"] = pd.to_numeric(df["FactValueNumeric"], errors="coerce")
    df["IsLatestYear"] = df["IsLatestYear"].astype(str).str.strip().str.lower()
    df["IsLatestYear"] = df["IsLatestYear"].astype(str).str.strip().str.lower()
    return df


st.set_page_config(page_title="Interactive Visualisation WS26", layout="wide")

st.title("Interactive Charts")

# Sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
  try:

    df = load_and_clean_data(uploaded_file)

    #GDP data
    df_gdp_all=pd.read_csv('gdp_worldwide.csv', skiprows=3)
    df_gdp_all=df_gdp_all[['Country Name','2023']].rename(
            columns={'Country Name': 'Location', '2023': 'GDP_2023'})
    df_gdp_all['Location'] = df_gdp_all['Location'].astype(str).str.strip().str.title()

    df_gdp_all=df_gdp_all.reset_index()
    print(df_gdp_all)

    # Global merge for all tabs
    df_merged = pd.merge(df, df_gdp_all, on='Location', how='inner')
    res_col = 'Dim1' if 'Dim1' in df_merged.columns else 'Location type'

    st.subheader("Cleaned Data Preview")
    st.dataframe(df.head())


    tab1, tab2, tab3, tab4 = st.tabs(["Stacked bar", "Heatmap Matrix", "Waterdrop", "Chloropeth"])
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


        fig.update_layout(
            xaxis_title="regions",
            yaxis_title="value in %",
            legend_title="colour legend",
            height=600,
            yaxis=dict(range=[0, 100], tickmode='linear', dtick=10),
            barnorm='percent'
        )

        # Display in browser
        st.subheader("Stacked Bar Visualization")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.info("Choose the starting year for the heatmap")
        df_heat = df_merged[df_merged[res_col] == 'Total'].copy()

        if not df_heat.empty:
            min_yr = int(df_heat['Period'].min())
            max_yr = int(df_heat['Period'].max())

            start_year = st.slider("Starting at year:", min_yr, max_yr, min_yr)
            df_heat_filtered = df_heat[df_heat['Period'] >= start_year]

            heatmap = alt.Chart(df_heat_filtered).mark_rect().encode(
                x=alt.X('Period:O', title='Year'),
                y=alt.Y('Location:N', title='Country', sort='ascending'),
                color=alt.Color('Value:Q', scale=alt.Scale(scheme='yellowgreenblue'),
                                legend=alt.Legend(title="Access %")),
                tooltip=['Location', 'Period', 'Value', 'GDP_2023']
            ).properties(width=700, height=alt.Step(20))

            st.altair_chart(heatmap, use_container_width=True)
        else:
            st.warning("No 'Total' data found for heatmap.")

    with tab3:
        st.info("Select a country and type to see the corresponding data in 2022")
        df_2022 = df_merged[df_merged['Period'] == 2022].copy()
        #st.write("2022 data shape:", df_2022.shape)  # Debug

        def create_drop_path():
            Path = mpath.Path
            path_data = [
                (Path.MOVETO, (0, 1.5)),
                (Path.CURVE4, (0.1, 1.0)), (Path.CURVE4, (1.1, 0.4)), (Path.CURVE4, (1.1, -0.5)),
                (Path.CURVE4, (1.1, -1.4)), (Path.CURVE4, (-1.1, -1.4)), (Path.CURVE4, (-1.1, -0.5)),
                (Path.CURVE4, (-1.1, 0.4)), (Path.CURVE4, (-0.1, 1.0)), (Path.CURVE4, (0, 1.5)),
                (Path.CLOSEPOLY, (0, 1.5)),
            ]
            codes, verts = zip(*path_data)
            return Path(verts, codes)

        def draw_water_drop(country, res_type):
            subset = df_2022[(df_2022['Location'] == country) & (df_2022[res_col] == res_type)]
            if subset.empty:
                st.warning(f"No data for {country} ({res_type})")
                return None

            percent = subset['Value'].iloc[0]
            gdp = subset['GDP_2023'].iloc[0] if 'GDP_2023' in subset else 0

            fig, ax = plt.subplots(figsize=(5, 7))
            drop_path = create_drop_path()

            ax.add_patch(mpatches.PathPatch(drop_path, facecolor='#f0f0f0', edgecolor='#2c3e50', lw=3, alpha=0.3))
            fill_drop = mpatches.PathPatch(drop_path, facecolor='#3498db', edgecolor='none')
            ax.add_patch(fill_drop)

            total_h = 2.8
            y_min = -1.3
            clip_rect = plt.Rectangle((-1.5, y_min), 3, (percent / 100) * total_h, transform=ax.transData)
            fill_drop.set_clip_path(clip_rect)

            ax.text(0, -0.3, f"{percent:.1f}%", fontsize=35, fontweight='bold', ha='center', color='#2c3e50')
            ax.text(0, -1.6, f"BIP 2023: ${gdp:,.0f}", fontsize=11, ha='center', color='gray')
            ax.set_title(f"{country} ({res_type})", fontsize=16, pad=20)
            ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.8, 1.8); ax.set_aspect('equal'); ax.axis('off')

            plt.tight_layout()
            return fig

        # Streamlit widgets

        types = sorted(df_2022[res_col].unique())

        col1, col2 = st.columns(2)
        with col1:
            countries = sorted(df_2022['Location'].unique())
            country = st.selectbox("Country:", countries,index=countries.index('Austria') if 'Austria' in countries else 0)


        with col2:
            res_type = st.selectbox("Type:", types, index=types.index('Total') if 'Total' in types else 0)
        col_empty1, col_chart, col_empty2 = st.columns([2, 2, 2])
        with col_chart:
            fig = draw_water_drop(country, res_type)
            if fig:
                st.pyplot(fig)
            else:
                st.warning(f"No data for {country} in 2022.")

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
    st.info("Upload Data_water.csv in the sidebar to see the chart.")