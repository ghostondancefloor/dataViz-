import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import coo_matrix
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Dashboard Interactif des Articles Politiques",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_data(file_path):
    return pd.read_csv(file_path)

def load_yearly_data():
    file1_path = 'metadata\year_data_2022.csv'
    file2_path = 'metadata\year_data_2023.csv'
    data1 = pd.read_csv(file1_path)
    data2 = pd.read_csv(file2_path)
    return pd.concat([data1, data2], ignore_index=True)

combined_data = load_yearly_data()

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

    # Filter data based on year and month
    filtered_data = data[(data["year"].isin(selected_year)) & (data["month"].isin(selected_month))]

    # Keyword Trend by Month with Month Filter
    st.subheader("1. Fréquence des mots-clés par mois")
    selected_keyword = st.text_input("Entrer un mot-clé pour filtrer les articles:", "")

    if selected_keyword:
        keyword_filtered_data = filtered_data[filtered_data['kws'].str.contains(selected_keyword, case=False, na=False)]

        if keyword_filtered_data.empty:
            st.warning(f"Aucun article trouvé mentionnant le mot-clé '{selected_keyword}'.")
        else:
            # Create a datetime column for grouping
            keyword_filtered_data['publish_date'] = pd.to_datetime(
                keyword_filtered_data[['year', 'month', 'day']],
                errors='coerce'
            )
            keyword_filtered_data = keyword_filtered_data.dropna(subset=['publish_date'])

            # Apply the month filter (already applied in filtered_data)
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
                    title=f"Articles qui mentionnent '{selected_keyword}' par mois",
                    color_discrete_sequence=['#FF4B4B'] 
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
    else:
        st.info("Entrer un mot-clé pour générer l'histogramme.")

    # Yearly and Category-Based Word Analysis
    st.subheader("2. Analyse basée sur les catégories ou les années")
    selected_category = st.text_input("Saisir la catégorie", "")

    filtered_data2 = combined_data[(combined_data["year"].isin(selected_year))]


    # Check if 'category' column exists
    if 'category' in combined_data.columns:
        # Filter data based on category
        if selected_category:
            histogram_data = filtered_data2[filtered_data2['category'].str.contains(selected_category, case=False, na=False)]
        else:
            histogram_data = filtered_data2

        if histogram_data.empty:
            st.warning("Aucune donnée disponible pour la catégorie sélectionnée.")
        else:
            # Check for the 'key' column
            if 'key' in histogram_data.columns:
                aggregated_data = histogram_data.groupby('key', as_index=False)['value'].sum()

                # Get the top 10 most common words
                top_words = aggregated_data.nlargest(10, 'value')

                # Create a bar chart using Plotly Express
                fig = px.bar(
                    top_words,
                    x='key',
                    y='value',
                    labels={'key': 'Mots', 'value': 'Fréquence'},
                    title='Top 10 Mots',
                    color_discrete_sequence=['#FF4B4B'] 
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





# Section 3: 
# Section 3: Keyword Correlation Network
elif section == "Corrélation des Mots-Clés":
    st.title("Relation entre les différents organisations")

    # Load the keyword relationships
    relationships_file = 'output_data/org_relationships.csv'  # Replace with your file path
    keyword_relationships = pd.read_csv(relationships_file)


    st.sidebar.header("Filtres")
    min_count = st.sidebar.slider("Minimum Count", min_value=10, max_value=100, value=1)
    filtered_relationships = keyword_relationships[keyword_relationships['count'] > min_count]

    # Create a graph from the filtered relationships
    G = nx.Graph()
    for _, row in filtered_relationships.iterrows():
        G.add_edge(row['keyword'], row['related_keyword'], weight=row['count'])

    # Create the graph visualization
    def plot_network(G):
        # Generate layout
        pos = nx.spring_layout(G, k=0.5)  # Force-directed layout

        # Prepare edge data
        edge_x = []
        edge_y = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

        # Prepare node data
        node_x = []
        node_y = []
        node_size = []
        node_labels = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_size.append(sum([G[node][neighbor]['weight'] for neighbor in G.neighbors(node)]) / 10)
            node_labels.append(f"{node}")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=node_labels,
            marker=dict(
                size=node_size,
                color="lightblue",
                line=dict(width=1, color="darkblue"),
            )
        )

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            title="Keyword Correlation Network",
                            titlefont_size=16,
                            showlegend=False,
                            hovermode="closest",
                            margin=dict(b=0, l=0, r=0, t=40),
                            xaxis=dict(showgrid=False, zeroline=False),
                            yaxis=dict(showgrid=False, zeroline=False),
                        ))
        return fig

    # Display the graph
    st.markdown("### Réseau de Corrélations")
    if not filtered_relationships.empty:
        fig = plot_network(G)
        st.plotly_chart(fig)
    else:
        st.warning("Aucune relation à afficher pour le seuil de count sélectionné.")


# Footer
st.sidebar.markdown("Ikram IDDOUCH - Khadija Zaroil")
st.sidebar.markdown("**Dashboard créé avec Streamlit**")
