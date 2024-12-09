import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix

# Page Configuration
st.set_page_config(
    page_title="Dashboard Interactif des Articles Politiques",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(file_path):
    return pd.read_csv(file_path)

DATA_FILE = "output_data/data.csv"
data = load_data(DATA_FILE)

# Sidebar: Navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Choisissez une section :",
    ["Visualisations Simples", "Analyse Géographique", "Corrélation des Mots-Clés"]
)

# Section 1: Simple Visualizations
if section == "Visualisations Simples":
    st.title("Visualisations Simples")
    st.sidebar.header("Filtres")
    selected_year = st.sidebar.multiselect("Année", sorted(data["year"].unique()), default=data["year"].unique())
    selected_month = st.sidebar.multiselect("Mois", sorted(data["month"].unique()), default=data["month"].unique())

    # Filter data based on user selection
    filtered_data = data[
        (data["year"].isin(selected_year)) &
        (data["month"].isin(selected_month))
    ]

    # Temporal Overview
    st.subheader("1. Nombre d'articles publiés par jour")
    daily_counts = filtered_data.groupby(["year", "month", "day"]).size().reset_index(name="count")
    fig = px.bar(daily_counts, x="day", y="count", color="month", title="Nombre d'articles publiés par jour")
    st.plotly_chart(fig)

    # Top Keywords
    st.subheader("2. Mots-clés les plus fréquents")
    if "kws" in filtered_data.columns:
        keyword_freq = filtered_data["kws"].str.split(",").explode().value_counts().reset_index()
        keyword_freq.columns = ["Mot-clé", "Occurrences"]
        fig = px.bar(keyword_freq.head(10), x="Occurrences", y="Mot-clé", orientation="h", title="Top 10 des mots-clés")
        st.plotly_chart(fig)

# Section 2: Geographic Analysis
elif section == "Analyse Géographique":
    st.title("Analyse Géographique")

    def load_geocoded_data():
        df_2022 = pd.read_csv("metadata/geocoded_locations_2022.csv")
        df_2023 = pd.read_csv("metadata/geocoded_locations_2023.csv")
        df_2022['dataset_year'] = 2022
        df_2023['dataset_year'] = 2023
        return pd.concat([df_2022, df_2023], ignore_index=True)

    geocoded_data = load_geocoded_data()

    # Sidebar options
    st.sidebar.title("Filtres Géographiques")
    dataset_years = sorted(geocoded_data['dataset_year'].unique())
    selected_dataset_year = st.sidebar.selectbox("Select Year", dataset_years)
    filtered_geocoded = geocoded_data[geocoded_data['dataset_year'] == selected_dataset_year]
    years = sorted(filtered_geocoded['year'].unique())
    color_themes = ['Viridis', 'Cividis', 'Inferno', 'Plasma', 'Magma', 'Turbo']
    selected_color_theme = st.sidebar.selectbox("Select Color Theme", color_themes)

    # Aggregate data: sum values for each country
    country_summary = filtered_geocoded.groupby('country').agg(
        total_value=('value', 'sum'),  # Sum the 'value' column
        latitude=('latitude', 'first'),  # Use the first latitude value for each country
        longitude=('longitude', 'first')  # Use the first longitude value for each country
    ).reset_index()

    def make_choropleth(summary_df, color_theme):
        min_value = summary_df['total_value'].min()
        max_value = summary_df['total_value'].max()
        midpoint_value = summary_df['total_value'].median()
        fig = px.choropleth(
            summary_df,
            locations="country",
            locationmode="country names",
            color="total_value",
            hover_name="country",
            hover_data={"total_value": True, "latitude": False, "longitude": False},
            title="Geocoded Locations Map (Aggregated Values by Country)",
            color_continuous_scale=color_theme,
            range_color=(min_value, max_value),
            color_continuous_midpoint=midpoint_value
        )
        fig.update_layout(
            geo=dict(
                showland=True,
                landcolor="lightgray",
                showocean=True,
                oceancolor="lightblue",
                projection_type="natural earth"
            ),
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        return fig

        # Display the map in the middle
    st.markdown(f"#### Carte des Pays les Plus Cités ({selected_dataset_year})")
    choropleth = make_choropleth(country_summary, selected_color_theme)
    st.plotly_chart(choropleth, use_container_width=True)

    # Display table and pie chart side by side
    st.markdown("### Top 10 des Pays les Plus Cités")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Tableau des Pays")
        top_countries = country_summary.nlargest(10, 'total_value')  # Get top 10 countries by total_value
        top_countries = top_countries.rename(columns={'country': 'Pays', 'total_value': 'Nombre de fois cité'})  # Rename columns for display
        st.dataframe(top_countries[['Pays', 'Nombre de fois cité']], use_container_width=True, hide_index=True)

    with col2:
        st.markdown("#### Répartition des Pays les Plus Cités")
        pie_chart = px.pie(
            top_countries,
            names='Pays',
            values='Nombre de fois cité',
            hole=0.4  # To create a donut chart style
        )
        st.plotly_chart(pie_chart, use_container_width=True)





# Section 3: Keyword Correlation
# elif section == "Corrélation des Mots-Clés":
#     st.title("Corrélation des Mots-Clés")

#     # Preprocess Keywords
#     @st.cache
#     def preprocess_keywords(df):
#         df["kws_list"] = df["kws"].str.split(",")
#         return df

#     data = preprocess_keywords(data)

#     # Build Keyword Network
#     def build_network_graph(df):
#         G = nx.Graph()
#         for keywords in df["kws_list"].dropna():
#             pairs = combinations(sorted(set(keywords)), 2)
#             for k1, k2 in pairs:
#                 if G.has_edge(k1, k2):
#                     G[k1][k2]["weight"] += 1
#                 else:
#                     G.add_edge(k1, k2, weight=1)
#         return G

#     G = build_network_graph(data)

#     # Display Network Graph
#     st.subheader("Graphique des Corrélations")
#     fig, ax = plt.subplots(figsize=(12, 12))
#     pos = nx.spring_layout(G, k=0.3)
#     nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
#     nx.draw_networkx_edges(G, pos, width=[d["weight"] for (_, _, d) in G.edges(data=True)])
#     nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")
#     plt.title("Keyword Co-occurrence Network")
#     st.pyplot(fig)

# Footer
st.sidebar.markdown("Ikram IDDOUCH - Khadija Zaroil")
st.sidebar.markdown("**Dashboard créé avec Streamlit**")
