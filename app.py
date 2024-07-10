import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.features import RadViz
from sklearn.metrics import silhouette_score

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('./fin_score_p.csv').drop(columns=['Unnamed: 0']).rename(columns={'contribution_s': 'contribution', 'stability_s': 'stability', 'popularity_s':'popularity', 'commission_s': 'commission', 'period_s': 'period'})

data = load_data()

st.title("Validator Analysis with RadViz")

# Extract unique chain names for the dropdown menu
chain_list = data['chain'].unique().tolist()

# Initialize session state
if 'chain_name' not in st.session_state:
    st.session_state.chain_name = chain_list[0]
if 'elbow_plot' not in st.session_state:
    st.session_state.elbow_plot = None
if 'silhouette_plot' not in st.session_state:
    st.session_state.silhouette_plot = None
if 'cluster_num' not in st.session_state:
    st.session_state.cluster_num = 3
if 'weights' not in st.session_state:
    st.session_state.weights = {
        'contribution': 1,
        'stability': 1,
        'popularity': 1,
        'commission': 1,
        'period': 1
    }
if 'show_correlation' not in st.session_state:
    st.session_state.show_correlation = False
if 'optimal_k' not in st.session_state:
    st.session_state.optimal_k = 3

# Sidebar: Select chain name and settings
with st.sidebar:
    st.header("Custom Input")
    chain_name = st.selectbox("Select chain name:", chain_list, index=chain_list.index(st.session_state.chain_name))
    if chain_name != st.session_state.chain_name:
        st.session_state.show_correlation = False
    st.session_state.chain_name = chain_name

    # Sidebar: Show correlation heatmap
    if st.button("Show Correlation"):
        st.session_state.show_correlation = True

    if st.session_state.show_correlation:
        afin = data[data['period'].isna() == False].reset_index(drop=True)
        akash = afin[afin['chain'] == st.session_state.chain_name].drop(columns=['chain'])

        # Correlation heatmap
        corr = akash[['contribution', 'stability', 'popularity', 'commission', 'period']].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='Oranges', ax=ax)
        ax.set_title('Correlation Heatmap')
        st.pyplot(fig)

    # Sidebar: Set weights for evaluation indicators
    st.write("### Set Weights for Scores")
    weights = {}
    for indicator in ['contribution', 'stability', 'popularity', 'commission', 'period']:
        weights[indicator] = st.slider(f"{indicator}", 0, 5, st.session_state.weights[indicator])
        st.session_state.weights[indicator] = weights[indicator]

    # Sidebar: Evaluate KMeans
    if st.button("KMeans Evaluate"):
        afin = data[data['period'].isna() == False].reset_index(drop=True)
        akash = afin[afin['chain'] == st.session_state.chain_name].drop(columns=['chain'])

        # Apply weights and exclude columns with weight 0
        for indicator in ['contribution', 'stability', 'popularity', 'commission', 'period']:
            if st.session_state.weights[indicator] == 0:
                akash = akash.drop(columns=[indicator])
            else:
                akash[indicator] *= st.session_state.weights[indicator]

        # Recalculate scores to make the sum of scores 100
        akash_k = akash.drop(columns=['voter'])
        akash_k = akash_k.div(akash_k.sum(axis=1), axis=0) * 100

        # Elbow method plot and silhouette scores
        inertia = []
        silhouette_scores = []
        K = range(2, 11)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(akash_k)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(akash_k, kmeans.labels_))

        # Find the optimal k with the highest silhouette score
        optimal_k = K[np.argmax(silhouette_scores)]
        st.session_state.optimal_k = optimal_k

        fig, ax = plt.subplots()
        ax.plot(K, inertia, 'bo-', markersize=8)
        ax.set_xlabel('Number of clusters, k')
        ax.set_ylabel('Inertia')
        ax.set_title('Elbow Method For Optimal k')
        st.session_state.elbow_plot = fig

        # Silhouette coefficient plot for the optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=0)
        kmeans.fit(akash_k)
        fig, ax = plt.subplots()
        visualizer = SilhouetteVisualizer(kmeans, ax=ax)
        visualizer.fit(akash_k)
        visualizer.finalize()
        st.session_state.silhouette_plot = fig

    # Display the elbow plot and silhouette coefficient plot if they exist in session state
    if st.session_state.elbow_plot is not None and st.session_state.silhouette_plot is not None:
        st.write("### KMeans Evaluation")
        st.pyplot(st.session_state.elbow_plot)
        st.pyplot(st.session_state.silhouette_plot)

# Main: Set number of clusters and run analysis
cluster_num = st.number_input("Select number of clusters:", min_value=2, max_value=10, value=st.session_state.optimal_k, step=1)

def run_analysis():
    st.session_state.cluster_num = cluster_num
    afin = data[data['period'].isna() == False].reset_index(drop=True)
    akash = afin[afin['chain'] == st.session_state.chain_name].drop(columns=['chain'])

    # Apply weights and exclude columns with weight 0
    for indicator in ['contribution', 'stability', 'popularity', 'commission', 'period']:
        if st.session_state.weights[indicator] == 0:
            akash = akash.drop(columns=[indicator])
        else:
            akash[indicator] *= st.session_state.weights[indicator]

    # Recalculate scores to make the sum of scores 100
    akash_k = akash.drop(columns=['voter'])
    akash_k = akash_k.div(akash_k.sum(axis=1), axis=0) * 100

    kmeans = KMeans(n_clusters=st.session_state.cluster_num, init='k-means++', max_iter=300, random_state=0, n_init=10)
    kmeans.fit_transform(akash_k)
    akash['kmeanscluster'] = kmeans.labels_

    # Radviz
    fig, ax = plt.subplots()
    classs = [i for i in range(st.session_state.cluster_num)]
    visualizer = RadViz(classes=classs, ax=ax)
    visualizer.fit(akash_k, kmeans.labels_)
    visualizer.transform(akash_k)
    visualizer.show()
    fig.savefig("radviz_plot_all.png")
    st.image("radviz_plot_all.png")

    # Display cluster statistics
    st.write("### Cluster Statistics")
    akash = akash.rename(columns={'kmeanscluster': 'cluster'})
    
    cluster_counts = akash[['cluster', 'voter']].groupby(by='cluster').count()
    cluster_stats = akash.drop(columns=['voter']).groupby('cluster').mean()
    cluster_stats = cluster_stats.rename(columns={
        'contribution': 'contribution_mean',
        'stability': 'stability_mean',
        'popularity': 'popularity_mean',
        'commission': 'commission_mean',
        'period': 'period_mean'
    })
    cluster_counts = cluster_counts.rename(columns={'voter': 'count'})
    cluster_counts = pd.concat([cluster_counts, cluster_stats], axis=1)
    st.write(cluster_counts)

    # Top 10 voters
    st.write("### Top 10 Voters List")
    akash['total_score'] = akash.drop(columns=['voter']).sum(axis=1)
    top_10_voters = akash.sort_values(by='total_score', ascending=False)[:10].reset_index(drop=True)
    st.write(top_10_voters)

# Run the analysis whenever weights change
run_analysis()
