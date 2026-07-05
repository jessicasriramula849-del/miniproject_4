import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Amazon Music Clustering", page_icon="🎵", layout="wide"
)

FILE_PATH = r"C:\Users\pc\Desktop\miniproject4\single_genre_artists_dbscan_cleaned.xlsx"

@st.cache_data
def load_and_cluster_data(path):
    if not os.path.exists(path):
        return None

    # Load original dataset
    df = pd.read_excel(path)

    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo', 'duration_ms'
    ]
    target_cols = [col for col in audio_features if col in df.columns]

    X_audio = df[target_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_audio)

    pca = PCA(n_components=4, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42, n_init=10)
    df["Music_Cluster"] = kmeans.fit_predict(X_pca)

    df["PCA_1"] = X_pca[:, 0]
    df["PCA_2"] = X_pca[:, 1]

    return df

df_clean = load_and_cluster_data(FILE_PATH)

if df_clean is None:
    st.error(f"Could not find the dataset at: `{FILE_PATH}`")
    st.info(
        "Please check your path or ensure the script is running on the correct machine."
    )
    st.stop()

cluster_identities = {
    0: {"name": "⚡ Energy Tracks Music", "color": "#ff4b4b"},
    1: {"name": "🍃 Chill Acoustic Music", "color": "#00a86b"},
    2: {"name": "💃 Party Music", "color": "#1f77b4"},
    3: {"name": "🎻 Deep Ambient Music", "color": "#9b59b6"} 
}

audio_features = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]
available_features = [col for col in audio_features if col in df_clean.columns]

st.sidebar.title("🎵 Navigation Panel")
app_mode = st.sidebar.radio(
    "Go to:",
    ["Home", "Song Discovery"],
)

st.sidebar.markdown("---")

if app_mode == "Home":
    st.title("Amazon Music Clustering 🎵.")
    st.markdown(
        "Welcome to my fourth mini project, in this project, I have grouped the songs provided in the amazon music dataset based on danceability, valence, tempo, energy, loudness, speechiness, instrumentalness, liveness, acousticness and duration_ms into four clusters using PCA and K-Means methods. Here below is a graphical representation of the four clusters of the songs."
    )
    st.image("music.jpg", caption="Jessica Sriramula")

elif app_mode == "Song Discovery":
    st.title("🔍 Search for the Song that You Like")
    st.markdown(
        "Here we can filter from the entire amazon music dataset by cluster group(energy hyped, chill acoustic, groove rhythmic and melodic deep ambient) to find and recommend song tracks based on specific musical moods."
    )

    cluster_choice = st.selectbox(
        "Choose a target sound profile category to explore:",
        options=[0, 1, 2, 3],
        format_func=lambda x: cluster_identities[x]["name"],
    )

    display_fields = ["name_song", "name_artists", "popularity_songs", "genres"]
    available_display = [col for col in display_fields if col in df_clean.columns]

    filtered_df = df_clean[df_clean["Music_Cluster"] == cluster_choice]

    st.markdown(f"Showing sample inventory matches for **{cluster_identities[cluster_choice]['name']}**")
    
    sort_by_popularity = st.checkbox("Sort results to display highest popularity first", value=True)
    if sort_by_popularity and "popularity_songs" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by="popularity_songs", ascending=False)

    search_query = st.text_input("🔍 Search for a specific song title or artist name within this cluster:")
    if search_query and ("name_song" in filtered_df.columns and "name_artists" in filtered_df.columns):
        mask = filtered_df["name_song"].str.contains(search_query, case=False, na=False) | \
               filtered_df["name_artists"].str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]

    st.dataframe(
        filtered_df[available_display].head(100),
        use_container_width=True,
        hide_index=True,
    )