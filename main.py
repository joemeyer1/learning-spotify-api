#!usr/bin/env python3

import sklearn

import os
import json
from typing import Dict, Any, List, Optional

from dataclasses import dataclass

from target_feature_types import target_feature_types

import numpy as np

from sklearn import metrics
from sklearn.cluster import HDBSCAN

from bokeh.models import ColumnDataSource, HoverTool

from collections import defaultdict
import pandas as pd


from bokeh.plotting import figure, show
from bokeh.palettes import Category20



@dataclass
class ArtistInfo:
    name: str
    id: str



@dataclass
class TrackInfo:
    name: str
    id: str
    feats: Optional[Dict[str, float]] = None
    artist: Optional[str] = None

    def get_feats(self, target_feature_types) -> List[float]:
        if self.feats is None:
            return []
        else:
            assert type(self.feats) == dict
            return [self.feats[feat_type] for feat_type in target_feature_types]



def normalize_data(data: np.ndarray) -> np.ndarray:
    for col_i in range(data.shape[1]):
        data[col_i] -= np.mean(data[col_i])
        data[col_i] /= np.std(data[col_i])
    return data


def get_top_tracks_info(artist: ArtistInfo, access_token: str) -> List[TrackInfo]:

    top_tracks = _get_data(
        output_filename=f"{artist.id}_top_tracks.json",
        access_token=access_token,
        data_args=f"artists/{artist.id}/top-tracks?market=US",
    )

    def _get_feats(track_ids: str) -> List[Dict[str, Any]]:
        tracks_feats = _get_data(
            output_filename=f"track_{track_ids}feats",
            access_token=access_token,
            data_args=f"audio-features?ids={track_ids}",
        )["audio_features"]
        print(tracks_feats)
        target_feats = [{feat_type: track_feats[feat_type] for feat_type in target_feature_types} for track_feats in tracks_feats]
        return target_feats

    tracks_infos = []
    track_ids = ''
    for track in top_tracks['tracks']:
        track_info = TrackInfo(
            name=track['name'],
            id=track['id'],
            artist=artist.name,
        )
        tracks_infos.append(track_info)
        track_ids += f"{track['id']},"
    tracks_feats = _get_feats(track_ids)
    for i, track_info in enumerate(tracks_infos):
        tracks_infos[i].feats = tracks_feats[i]
    return tracks_infos


def read_json(filename):
    with open(filename, 'r') as f:
        return json.loads(f.read())


# generate access token
def generate_access_token(access_token_filename: str = "access_token.json") -> str:
    os.system(f'curl -X POST "https://accounts.spotify.com/api/token" \
         -H "Content-Type: application/x-www-form-urlencoded" \
         -d "grant_type=client_credentials&client_id=41e74624046646428afaa5c8bca7173f&client_secret=d8d74e43da5e475c961090e14d2ecea7" > {access_token_filename}')

    access_token = read_json('access_token.json')['access_token']
    return access_token

def _get_data(output_filename: str, access_token: str, data_args: str = '') -> Dict[str, Any]:
    """Fetches and returns data."""

    os.system(f"curl --request GET \
      --url https://api.spotify.com/v1/{data_args} \
      --header 'Authorization: Bearer {access_token}' \
              > {output_filename}")

    data = read_json(output_filename)
    os.system(f"rm {output_filename}")
    return data



def get_similar_artists(artist_id: str, access_token: str) -> List[ArtistInfo]:
    similar_artists = _get_data(output_filename='related_artists', access_token=access_token, data_args=f'artists/{artist_id}/related-artists')
    similar_artists_info = [
        ArtistInfo(
            name=similar_artist['name'],
            id=similar_artist['id'],
        ) for similar_artist in similar_artists['artists']
    ]
    return similar_artists_info




def get_top_tracks(artists: List[ArtistInfo], access_token) -> List[TrackInfo]:
    top_tracks_infos = []
    for artist in artists:
        top_tracks_info = get_top_tracks_info(artist=artist, access_token=access_token)
        top_tracks_infos += top_tracks_info
    return top_tracks_infos

def get_tracks_feats(tracks: List[TrackInfo]) -> List[List[float]]:
    return normalize_data(np.array([track.get_feats(target_feature_types) for track in tracks]))


def fit_cluster_hdbscan(tracks: List[TrackInfo], min_cluster_size=5):
    tracks_feats = get_tracks_feats(tracks)
    return HDBSCAN(min_cluster_size=min_cluster_size).fit(tracks_feats)

def get_cluster_labels(tracks: List[TrackInfo]):
    return fit_cluster_hdbscan(tracks).labels_


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
            'x': [track.feats['energy'] for track in top_tracks],
            'y': [track.feats['danceability'] for track in top_tracks],
            'track': [track.name for track in top_tracks],
            'artist': [track.artist for track in top_tracks],
            'colors': get_colors(labels)
    }


    # create a ColumnDataSource by passing the dict
    source = ColumnDataSource(data=data)

    # create a plot using the ColumnDataSource's two columns

    # TOOLS="hover,zoom_in,zoom_out,box_zoom,undo,redo,reset,examine,help"

    p = figure()
    p.circle(x='x', y='y', source=source, size=10, fill_color='colors')
    p.add_tools(HoverTool(tooltips=[("track", "@track"), ("artist", "@artist")]))
    show(p)


def print_artist_clusters(top_tracks, labels, clusters_filename='clusters.csv'):

    def _map_label_to_tracks():
        label_map = defaultdict(list)
        for i, track in enumerate(top_tracks):
            label = labels[i]
            label_map[label].append((track.name, track.artist))
        return dict(label_map)

    label_to_tracks = _map_label_to_tracks()
    for label in label_to_tracks:
        print(f"{pd.DataFrame(label_to_tracks[label], columns=['track', 'artist'])}\n\n")

    pd.DataFrame.from_dict(label_to_tracks, orient='index').to_csv(clusters_filename)

def main():
    access_token = generate_access_token()
    led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
    artists = [led_zep_info] + get_similar_artists(led_zep_info.id, access_token)
    top_tracks = get_top_tracks(artists=artists, access_token=access_token)
    labels = get_cluster_labels(top_tracks)
    plot_data(top_tracks, labels)
    print_artist_clusters(top_tracks,labels)









