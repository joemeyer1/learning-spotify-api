#!usr/bin/env python3


from typing import Dict, List

import numpy as np

from utilities.clustering_helpers import cluster_tracks, write_clusters_to_csv, read_clusters, get_artist_spreads_over_clusters, plot_tracks_with_clusters
from utilities.data_manager import DataManager
from utilities.music_info import ArtistInfo


def cluster_artist_and_co_tracks(seed_artist_info: ArtistInfo = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")):
    """Clusters top tracks for seed artist and similar artists, and writes clustering to csv."""

    data_manager = DataManager()

    artists = [seed_artist_info] + data_manager.fetch_similar_artists(seed_artist_info.id)
    top_tracks = data_manager.fetch_top_tracks(artists=artists)
    # convert_tracks_to_df(top_tracks, 'top_tracks.csv')  # write to df

    labels = cluster_tracks(top_tracks, min_cluster_size=4)

    plot_tracks_with_clusters(top_tracks, labels)
    write_clusters_to_csv(top_tracks, labels)

    # get_artist_spreads_over_clusters(top_tracks, labels)


def _read_old_clusters(
        clusters_filename: str = 'saved_clusters.csv',
        seed_artist_info: ArtistInfo = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp"),
) -> np.ndarray:
    data_manager = DataManager()

    artists = [seed_artist_info] + data_manager.fetch_similar_artists(seed_artist_info.id)
    top_tracks = data_manager.fetch_top_tracks(artists=artists)
    # convert_tracks_to_df(top_tracks, 'top_tracks.csv')  # write to df

    labels = read_clusters(clusters_filename=clusters_filename, tracks=top_tracks)

    return labels


def _get_artist_spread_old_clusters(
        clusters_filename: str = 'saved_clusters.csv',
        seed_artist_info: ArtistInfo = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp"),
) -> Dict[str, List[int]]:
    """Maps each artist to the distribution of their tracks over clusters."""

    data_manager = DataManager()

    artists = [seed_artist_info] + data_manager.fetch_similar_artists(seed_artist_info.id)
    top_tracks = data_manager.fetch_top_tracks(artists=artists)
    # convert_tracks_to_df(top_tracks, 'top_tracks.csv')  # write to df

    labels = read_clusters(clusters_filename=clusters_filename, tracks=top_tracks)

    artist_spreads = get_artist_spreads_over_clusters(top_tracks, labels)
    return artist_spreads


def _get_num_clusters_per_artist_old_clusters(
        clusters_filename: str = 'saved_clusters.csv',
        seed_artist_info: ArtistInfo = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp"),
) -> Dict[str, int]:
    """Returns a map of the number of clusters each artist belongs to."""

    artist_spreads = _get_artist_spread_old_clusters(clusters_filename=clusters_filename, seed_artist_info=seed_artist_info)
    num_clusters_per_artist = {artist: len(cluster_spread) for artist, cluster_spread in artist_spreads.items()}
    return num_clusters_per_artist


if __name__ == '__main__':
    cluster_artist_and_co_tracks()
