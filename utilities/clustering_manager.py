#!usr/bin/env python3


import os
from collections import defaultdict, Counter
from copy import deepcopy
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from bokeh.io import show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20
from bokeh.plotting import figure
from sklearn.cluster import HDBSCAN

from utilities.music_info import TrackInfo
from utilities.target_feature_types import target_feature_types


class ClusteringManager:
    """Clusters tracks, and analyzes clustering."""

    tracks: List[TrackInfo]
    labels: Optional[np.ndarray]

    def __init__(self, tracks: List[TrackInfo]):
        self.tracks = tracks
        self.labels = None  # we'll initialize this upon "cluster_tracks()" or "read_clusters()" step

    def cluster_tracks(self, min_cluster_size: int = 4) -> np.ndarray:
        """Clusters tracks according to their features.

        Runs HDBSCAN clustering in 2 stages:
            1. Initial clustering
            2. Outliers clustering

        Normalizes column means but not standard deviations.


        sklearn HDBSCAN Docs: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
        sklearn HDBSCAN User Guide: https://scikit-learn.org/stable/modules/clustering.html#hdbscan
        """

        def _cluster_and_get_labels(tracks: List[TrackInfo]) -> np.ndarray:
            """Clusters tracks with HDBSCAN.

            Returns:
                An array of computed cluster indices in order corresponding to input tracks.
            """

            tracks_feats = self.get_tracks_feats(tracks)
            return HDBSCAN(min_cluster_size=min_cluster_size).fit(tracks_feats).labels_

        def _secondary_cluster(tracks: List[TrackInfo], labels: np.ndarray) -> np.ndarray:
            """Runs another clustering iteration on outliers that weren't clustered initially."""

            labels = deepcopy(labels)  # avoid unexpected global variable modification
            poor_cluster_ixs = np.where(labels == -1)[0]
            if len(poor_cluster_ixs) >= (min_cluster_size * 2):  # if there's not enough tracks to split into 2 clusters, don't bother
                poor_cluster_tracks = [tracks[ix] for ix in poor_cluster_ixs]
                poor_cluster_labels = _cluster_and_get_labels(poor_cluster_tracks)
                poor_cluster_labels[np.where(poor_cluster_labels >= 0)[0]] += max(labels) + 1
                labels[poor_cluster_ixs] = poor_cluster_labels
            return labels

        raw_labels = _cluster_and_get_labels(tracks=self.tracks)
        self.labels = _secondary_cluster(tracks=self.tracks, labels=raw_labels)
        return self.labels

    def read_clusters(
            self,
            clusters_filename: str = 'saved_clusters.csv',
    ) -> np.ndarray:
        """Fetches tracks and reads clusters."""

        def _read_clusters_from_tracks() -> np.ndarray:
            track_name_to_ix = {track.name: ix for ix, track in enumerate(self.tracks)}
            clusters_df = pd.read_csv(clusters_filename, index_col='Unnamed: 0')
            cluster_labels = np.zeros(len(self.tracks), dtype=int)
            for cluster_i in clusters_df.index:
                for track in clusters_df.loc[cluster_i]:
                    if type(track) is str:
                        track_name = track.split(", '")[0][2: -1]
                        track_ix = track_name_to_ix.get(track_name, None)
                        assert track_ix is not None
                        cluster_labels[track_ix] = int(cluster_i)
            return cluster_labels

        self.labels = _read_clusters_from_tracks()
        return self.labels

    def write_clusters_to_csv(self, clusters_filename: str = 'clusters.csv') -> pd.DataFrame:
        """Writes clusters to csv and returns clusters as df."""

        assert self.labels is not None

        def _map_label_to_tracks():
            label_map = defaultdict(list)
            for i, track in enumerate(self.tracks):
                label = self.labels[i]
                label_map[label].append((track.name, track.artist))
            return dict(label_map)

        label_to_tracks = _map_label_to_tracks()
        for label in label_to_tracks:
            print(f"{pd.DataFrame(label_to_tracks[label], columns=['track', 'artist'])}\n\n")

        artist_clusters_df = pd.DataFrame.from_dict(label_to_tracks, orient='index')
        artist_clusters_df.to_csv(clusters_filename)
        os.system(f"open {clusters_filename}")
        return artist_clusters_df

    def plot_tracks_with_clusters(self):
        """Plots tracks by energy and danceability, and color-codes by cluster label."""

        assert self.labels is not None

        def assign_colors():
            unique_labels = set(self.labels)
            available_colors = Category20[20]
            colors = {}
            for i, unique_label in enumerate(unique_labels):
                colors[unique_label] = available_colors[i % len(available_colors)]
            return colors

        def get_colors(labels: np.ndarray) -> List[str]:
            colors = assign_colors()
            return [colors[label] for label in labels]

        data = {
                'energy': [track.feats['energy'] for track in self.tracks],
                'danceability': [track.feats['danceability'] for track in self.tracks],
                'track': [track.name for track in self.tracks],
                'artist': [track.artist for track in self.tracks],
                'label': self.labels,
                'color': get_colors(self.labels)
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

    def get_artist_spreads_over_clusters(self) -> Dict[str, List[int]]:
        """Maps each artist to the distribution of their tracks over clusters."""

        assert self.labels is not None

        artist_to_clusters = defaultdict(list)

        for track, label in zip(self.tracks, self.labels):
            artist_to_clusters[track.artist].append(label)
        return {artist: list(Counter(distinct_clusters).values()) for artist, distinct_clusters in artist_to_clusters.items()}

    def get_tracks_feats(self, tracks: List[TrackInfo]) -> np.ndarray:
        return self.normalize_data(np.array([track.get_feats(target_feature_types) for track in tracks]))

    @staticmethod
    def normalize_data(data: np.ndarray) -> np.ndarray:
        for col_i in range(data.shape[1]):
            # normalize cols to have mean=0
            data[:, col_i] -= np.mean(data[:, col_i])

            # std is meaningful here (e.g. larger spreads matter), so don't normalize it
            # data[:, col_i] /= np.std(data[:, col_i])

        return data

    def _get_artist_spread_old_clusters(
            self,
            clusters_filename: str = 'saved_clusters.csv',
    ) -> Dict[str, List[int]]:
        """Reads clusters, then maps each artist to the distribution of their tracks over clusters."""

        self.read_clusters(clusters_filename)
        artist_spreads = self.get_artist_spreads_over_clusters()
        return artist_spreads

    def _get_num_clusters_per_artist_old_clusters(
            self,
            clusters_filename: str = 'saved_clusters.csv',
    ) -> Dict[str, int]:
        """Returns a map of the number of clusters each artist belongs to."""

        artist_spreads = self._get_artist_spread_old_clusters(clusters_filename=clusters_filename)
        num_clusters_per_artist = {artist: len(cluster_spread) for artist, cluster_spread in artist_spreads.items()}
        return num_clusters_per_artist
