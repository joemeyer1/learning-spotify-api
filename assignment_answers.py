#!usr/bin/env python3

import pandas as pd

from typing import Dict

from data_manager import DataManager

from clustering_helpers import cluster_tracks, write_clusters_to_csv, get_artist_spreads_over_clusters, plot_tracks_with_clusters
from music_info import ArtistInfo


def get_top_led_zep_tracks_and_feats() -> pd.DataFrame:
    """Fetches Led Zeppelin top tracks and audio features, and writes data to df."""

    data_manager = DataManager()
    return data_manager.fetch_top_tracks_df(
        artists=[ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")],
        filename='led_zep_tracks.csv',
    )


def get_similar_artist_tracks_and_feats() -> pd.DataFrame:
    """Fetches top tracks and their audio features for artists related to Led Zeppelin."""

    data_manager = DataManager()
    led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
    artists = data_manager.fetch_similar_artists(led_zep_info.id)
    print(f"led zep -ish artists: {[artist.name for artist in artists]}\n\n")
    return data_manager.fetch_top_tracks_df(artists=artists, filename='tracks.csv')


def get_led_zep_ish_tracks_and_feats() -> pd.DataFrame:
    """Fetches top tracks and their audio features for Led Zeppelin and related artists."""

    data_manager = DataManager()
    led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
    artists = [led_zep_info] + data_manager.fetch_similar_artists(led_zep_info.id)
    print(f"led zep -ish artists: {[artist.name for artist in artists[1:]]}\n\n")

    return data_manager.fetch_top_tracks_df(artists=artists, filename='tracks.csv')


def cluster_led_zep_ish_songs() -> pd.DataFrame:
    """Clusters songs by Led Zeppelin and related artists, and writes clusters to csv."""

    data_manager = DataManager()
    led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
    artists = [led_zep_info] + data_manager.fetch_similar_artists(led_zep_info.id)
    print(f"led zep -ish artists: {[artist.name for artist in artists[1:]]}\n\n")

    led_zep_ish_tracks_and_feats = data_manager.fetch_top_tracks(artists=artists)

    # I'm using HDBSCAN, which builds MST then deletes weakest edges until clusters are separated.
    # HDBSCAN can handle weirdly shaped clusters of different sizes, and requires minimal hyper-parameter tuning.
    # In practice I find HDBSCAN often aligns better with my intuition compared with other clustering algorithms.

    # I set min_cluster_size to 4 because intuitively a set of 4 songs is about the minimum required to observe/assess meaningful patterns.
    # Given min_cluster_size = 4, the precise number of clusters yielded by HDBSCAN may vary. A recent run yielded 24 clusters (after secondary step - see below).
    # After initial HDBSCAN clustering, outliers are lumped together in their own cluster (labelled '-1').
    # I run one more HDBSCAN iteration to cluster these outliers.
    # After second HDBSCAN iteration, an outlier cluster may still persist,
    # but in practice this is often ok - if it's not, then lingering outliers may be clustered by another algorithm.
    labels = cluster_tracks(led_zep_ish_tracks_and_feats, min_cluster_size=4)

    return write_clusters_to_csv(led_zep_ish_tracks_and_feats, labels)


def get_num_clusters_per_artist() -> Dict[str, int]:
    """Returns the number of clusters each artist appears in."""

    data_manager = DataManager()
    led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
    artists = [led_zep_info] + data_manager.fetch_similar_artists(led_zep_info.id)
    top_tracks = data_manager.fetch_top_tracks(artists=artists)

    labels = cluster_tracks(tracks=top_tracks, min_cluster_size=4)

    plot_tracks_with_clusters(top_tracks, labels)
    write_clusters_to_csv(top_tracks, labels)

    artist_spreads = get_artist_spreads_over_clusters(top_tracks, labels)
    print(f"Artist to Cluster Distribution Table:\n{artist_spreads}")
    num_clusters_per_artist = {artist: len(cluster_spread) for artist, cluster_spread in artist_spreads.items()}
    print(f"Artist to Number of Clusters Map:\n{num_clusters_per_artist}")
    return num_clusters_per_artist










