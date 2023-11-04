#!usr/bin/env python3

from typing import List, Optional

from clustering_helpers import cluster_tracks, write_artist_clusters, get_artist_spreads_over_clusters
from music_info import ArtistInfo, TrackInfo
from target_feature_types import target_feature_types

import numpy as np

from bokeh.models import ColumnDataSource, HoverTool

import pandas as pd


from bokeh.plotting import figure, show
from bokeh.palettes import Category20

from data_manager import DataManager


def normalize_data(data: np.ndarray) -> np.ndarray:
    for col_i in range(data.shape[1]):
        # normalize cols to have mean=0
        data[:, col_i] -= np.mean(data[:, col_i])

        # std is meaningful here (e.g. larger spreads matter), so don't normalize it
        # data[:, col_i] /= np.std(data[:, col_i])

    return data


def get_tracks_feats(tracks: List[TrackInfo]) -> np.ndarray:
    return normalize_data(np.array([track.get_feats(target_feature_types) for track in tracks]))


def plot_data(top_tracks, labels):

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


def convert_tracks_to_df(tracks: List[TrackInfo], filename: Optional[str] = None) -> pd.DataFrame:
    """Returns a df of tracks, and writes to csv if filename is passed."""

    data = []
    for track in tracks:
        row = [track.name, track.artist] + track.get_feats(target_feature_types)
        data.append(row)
    df = pd.DataFrame(data, columns=['track', 'artist'] + target_feature_types)
    if filename is not None:
        df.to_csv(filename)
    return df


def main():
    """Clusters top tracks for Led Zepellin and similar artists."""

    led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")

    data_manager = DataManager()

    artists = [led_zep_info] + data_manager.get_similar_artists(led_zep_info.id)
    top_tracks = data_manager.get_top_tracks(artists=artists)
    convert_tracks_to_df(top_tracks, 'tracks.csv')

    labels = cluster_tracks(top_tracks, min_cluster_size=4)

    plot_data(top_tracks, labels)
    write_artist_clusters(top_tracks, labels)

    get_artist_spreads_over_clusters(top_tracks, labels)


if __name__ == '__main__':
    main()
