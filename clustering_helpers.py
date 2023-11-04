import os
from collections import defaultdict, Counter
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from main import get_tracks_feats
from music_info import TrackInfo


def cluster_tracks(tracks: List[TrackInfo], min_cluster_size: int = 4) -> np.ndarray:
    """Clusters tracks according to their features.

    Runs HDBSCAN clustering in 2 stages:
        1. Initial clustering
        2. Outliers clustering

    Normalizes column means but not standard deviations.
    """

    def _cluster_and_get_labels(tracks: List[TrackInfo], min_cluster_size: int = 4) -> np.ndarray:
        """Clusters tracks with HDBSCAN.

        Returns:
            An array of computed cluster indices in order corresponding to input tracks.
        """

        tracks_feats = get_tracks_feats(tracks)
        return HDBSCAN(min_cluster_size=min_cluster_size).fit(tracks_feats).labels_

    def _secondary_cluster(tracks: List[TrackInfo], labels: np.ndarray, min_cluster_size: int = 4) -> np.ndarray:
        """Runs another clustering iteration on outliers that weren't clustered initially."""

        labels = deepcopy(labels)  # avoid unexpected global variable modification
        poor_cluster_ixs = np.where(labels == -1)[0]
        if len(poor_cluster_ixs) >= (min_cluster_size * 2):  # if there's not enough tracks to split into 2 clusters, don't bother
            poor_cluster_tracks = [tracks[ix] for ix in poor_cluster_ixs]
            poor_cluster_labels = _cluster_and_get_labels(poor_cluster_tracks, min_cluster_size=min_cluster_size)
            poor_cluster_labels[np.where(poor_cluster_labels >= 0)[0]] += max(labels) + 1
            labels[poor_cluster_ixs] = poor_cluster_labels
        return labels

    raw_labels = _cluster_and_get_labels(tracks=tracks, min_cluster_size=min_cluster_size)
    labels = _secondary_cluster(tracks=tracks, labels=raw_labels, min_cluster_size=min_cluster_size)
    return labels


def write_artist_clusters(top_tracks, labels, clusters_filename='clusters.csv'):

    def _map_label_to_tracks():
        label_map = defaultdict(list)
        for i, track in enumerate(top_tracks):
            label = labels[i]
            label_map[label].append((track.name, track.artist))
        return dict(label_map)

    label_to_tracks = _map_label_to_tracks()
    for label in label_to_tracks:
        print(f"{pd.DataFrame(label_to_tracks[label], columns=['track', 'artist'])}\n\n")

    artist_clusters_df = pd.DataFrame.from_dict(label_to_tracks, orient='index')
    artist_clusters_df.to_csv(clusters_filename)
    os.system(f"open {clusters_filename}")
    return artist_clusters_df


def get_artist_spreads_over_clusters(tracks: List[TrackInfo], labels: np.ndarray):

    artist_to_clusters = defaultdict(list)

    for track, label in zip(tracks, labels):
        artist_to_clusters[track.artist].append(label)
    return {artist: Counter(distinct_clusters).values() for artist, distinct_clusters in artist_to_clusters.items()}