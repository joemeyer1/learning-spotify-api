#!usr/bin/env python3


import os
from collections import defaultdict, Counter
from copy import deepcopy
from typing import List

import numpy as np
import pandas as pd
from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20
from bokeh.plotting import figure
from sklearn.cluster import HDBSCAN

from music_info import TrackInfo
from target_feature_types import target_feature_types


def cluster_tracks(tracks: List[TrackInfo], min_cluster_size: int = 4) -> np.ndarray:
    """Clusters tracks according to their features.

    Runs HDBSCAN clustering in 2 stages:
        1. Initial clustering
        2. Outliers clustering

    Normalizes column means but not standard deviations.


    sklearn HDBSCAN Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    sklearn HDBSCAN User Guide: https://scikit-learn.org/stable/modules/clustering.html#hdbscan
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


def write_clusters_to_csv(top_tracks, labels, clusters_filename='clusters.csv'):

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


def get_tracks_feats(tracks: List[TrackInfo]) -> np.ndarray:
    return normalize_data(np.array([track.get_feats(target_feature_types) for track in tracks]))


def normalize_data(data: np.ndarray) -> np.ndarray:
    for col_i in range(data.shape[1]):
        # normalize cols to have mean=0
        data[:, col_i] -= np.mean(data[:, col_i])

        # std is meaningful here (e.g. larger spreads matter), so don't normalize it
        # data[:, col_i] /= np.std(data[:, col_i])

    return data


def plot_tracks_with_clusters(top_tracks, labels):
    """Plots tracks by energy and danceability, and color-codes by cluster label."""

    def assign_colors(labels):
        unique_labels = set(labels)
        available_colors = Category20[20]
        colors = {}
        for i, unique_label in enumerate(unique_labels):
            colors[unique_label] = available_colors[i % len(available_colors)]
        return colors

    def get_colors(labels):
        colors = assign_colors(labels)
        return [colors[label] for label in labels]

    data = {
            'energy': [track.feats['energy'] for track in top_tracks],
            'danceability': [track.feats['danceability'] for track in top_tracks],
            'track': [track.name for track in top_tracks],
            'artist': [track.artist for track in top_tracks],
            'label': labels,
            'color': get_colors(labels)
    }

    source = ColumnDataSource(data=data)

    # TOOLS="hover,zoom_in,zoom_out,box_zoom,undo,redo,reset,examine,help"

    p = figure()
    p.circle(x='energy', y='danceability', source=source, size=10, fill_color='color')
    p.add_tools(HoverTool(tooltips=[
        ("track", "@track"),
        ("artist", "@artist"),
        ("cluster_ix", "@label"),
        ("energy", "@energy"),
        ("danceability", "@danceability"),
    ]))
    show(p)