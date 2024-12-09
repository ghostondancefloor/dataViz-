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

# Load Main Data

def load_data(file_path):
    return pd.read_csv(file_path)

# Load Data Files
DATA_FILE = "output_data/data.csv"
data = load_data(DATA_FILE)


def load_yearly_data():
    file1_path = 'metadata\year_data_2022.csv'
    file2_path = 'metadata\year_data_2023.csv'
    data1 = pd.read_csv(file1_path)
    data2 = pd.read_csv(file2_path)
    return pd.concat([data1, data2], ignore_index=True)

combined_data = load_yearly_data()

def load_geocoded_data():
    df_2022 = pd.read_csv('metadata\geocoded_locations_2022.csv')
    df_2023 = pd.read_csv('metadata\geocoded_locations_2023.csv')
    df_2022['dataset_year'] = 2022
    df_2023['dataset_year'] = 2023
    return pd.concat([df_2022, df_2023], ignore_index=True)

geocoded_data = load_geocoded_data()

# Sidebar Navigation
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

    # Filter data
    filtered_data = data[(data["year"].isin(selected_year)) & (data["month"].isin(selected_month))]

    # Keyword Trend by Month with Month Filter
    st.subheader("1. Fréquence des mots-clés par mois")
    selected_keyword = st.sidebar.text_input("Entrer un mot-clé pour filtrer les articles:", "")

    if selected_keyword:
        keyword_filtered_data = data[data['kws'].str.contains(selected_keyword, case=False, na=False)]

        if keyword_filtered_data.empty:
            st.warning(f"Aucun article trouvé mentionnant le mot-clé '{selected_keyword}'.")
        else:
        # Create a datetime column for grouping
            keyword_filtered_data['publish_date'] = pd.to_datetime(
                keyword_filtered_data[['year', 'month', 'day']],
                errors='coerce'
            )
            keyword_filtered_data = keyword_filtered_data.dropna(subset=['publish_date'])

        # Apply the month filter
            keyword_filtered_data = keyword_filtered_data[keyword_filtered_data['month'].isin(selected_month)]

        # Check if data is empty after applying month filter
            if keyword_filtered_data.empty:
                st.warning("Aucun article trouvé pour les mois sélectionnés avec le mot-clé donné.")
            else:
            # Group data by month and count the number of articles
                keyword_filtered_data['month'] = keyword_filtered_data['publish_date'].dt.to_period('M').astype(str)
                monthly_counts = keyword_filtered_data.groupby('month').size().reset_index(name='article_count')

            # Create a histogram using Plotly Express
                fig = px.bar(
                    monthly_counts,
                    x='month',
                    y='article_count',
                    labels={'month': 'Mois', 'article_count': 'Nombre d articles'},
                    title=f"Articles qui mentionnent '{selected_keyword}' par mois"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
    else:
        st.info("Entrer un mot-clé pour générer l'histogramme.")

    # Yearly and Category-Based Word Analysis
    st.subheader("2. Analyse basée sur les catégories ou les années")

# Category-specific histogram section
    selected_category = st.text_input("Saisir la catégorie", "")

# Check if 'category' column exists
    if 'category' in combined_data.columns:
    # Apply category filter only for the histogram
        if selected_category:
            histogram_data = combined_data[combined_data['category'].str.contains(selected_category, case=False, na=False)]
        else:
            histogram_data = combined_data

        if histogram_data.empty:
            st.warning("Aucune donnée disponible pour la catégorie sélectionnée.")
        else:
        # Check for the 'kws' column
            if 'key' in histogram_data.columns:
                aggregated_data = histogram_data.groupby('key', as_index=False)['value'].sum()

            # Get the top 10 most common words
                top_words = aggregated_data.nlargest(10, 'value')

            # Create a bar chart using Plotly Express
                fig = px.bar(
                    top_words,
                    x='key',
                    y='value',
                    labels={'kws': 'Mots', 'value': 'Fréquence'},
                    title='Top 10 Mots'
                )
                fig.update_layout(xaxis_tickangle=-45)

            # Display the bar chart
                st.plotly_chart(fig)
            else:
                st.warning("The column 'key' does not exist in the dataset. Please check your data structure.")
    else:
        st.warning("The column 'category' does not exist in the dataset. Please check your data structure.")


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
