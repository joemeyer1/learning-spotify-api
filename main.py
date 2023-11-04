#!usr/bin/env python3

import sklearn

import os
import json
from typing import Dict, Any, List, Optional

from dataclasses import dataclass

from target_feature_types import target_feature_types


@dataclass
class ArtistInfo:
    name: str
    id: str


@dataclass
class TrackFeatures:



@dataclass
class TrackInfo:
    name: str
    id: str
    feats: Optional[Dict[str, float]] = None



# def get_top_tracks_info(artist_id: str, access_token: str, output_filename: str) -> List[TrackInfo]:
#
#     top_tracks = _get_data(
#         output_filename=output_filename,
#         access_token=access_token,
#         data_args=f"artists/{artist_id}/top-tracks?market=US",
#     )
#
#     def _get_feats(track_ids: List[str]) -> Dict[str, Any]:
#         feats = _get_data(
#             output_filename="track_feats.json",
#             access_token=access_token,
#             data_args=f"audio-features/{','.join(track_ids)}",
#         )["audio_features"]
#         target_feats = {feat_type: feats[feat_type] for feat_type in target_feature_types}
#         return target_feats
#
#     track_id_to_name = {track['id']: track['name'] for track in top_tracks['tracks']}
#     top_tracks_feats = _get_feats(list(track_id_to_name.keys()))
#     tracks_infos = []
#     for track_id in track_id_to_name.keys():
#         track_info = TrackInfo(
#             name=track_id_to_name['track_id'],
#             id=track_id,
#             feats=_get_feats(track['id']),
#         )
#         tracks_infos.append(track_info)
#     return tracks_infos



def get_top_tracks_info(artist_id: str, access_token: str) -> List[TrackInfo]:
    print(f"artist_id: {artist_id}")

    top_tracks = _get_data(
        output_filename=f"{artist_id}_top_tracks.json",
        access_token=access_token,
        data_args=f"artists/{artist_id}/top-tracks?market=US",
    )

#     def _get_feats(track_ids: str) -> Dict[str, Any]:
# #         print(f"track_id: {track_id}")
#         feats = _get_data(
#             output_filename=f"{track_id}_track_feats.json",
#             access_token=access_token,
#             data_args=f"audio-features?ids={track_ids}",
#         )  # ["audio_features"]
#         target_feats = {feat_type: feats[feat_type] for feat_type in target_feature_types}
#         return target_feats

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
        )
        tracks_infos.append(track_info)
        track_ids += f"{track['id']},"
    tracks_feats = _get_feats(track_ids)
    for i, track_info in enumerate(tracks_infos):
        tracks_infos[i].feats = tracks_feats[i]
    return tracks_infos

#
# def get_top_tracks_info(artist_id: str, access_token: str) -> List[TrackInfo]:
#
#     top_tracks = _get_data(
#         output_filename=f"{artist_id}_top_tracks.json",
#         access_token=access_token,
#         data_args=f"artists/{artist_id}/top-tracks?market=US",
#     )
#
#     def _get_feats(track_id: str) -> Dict[str, Any]:
#         feats = _get_data(
#             output_filename=f"{track_id}_track_feats.json",
#             access_token=access_token,
#             data_args=f"audio-features/{track_id}",
#         )  # ["audio_features"]
#         target_feats = {feat_type: feats[feat_type] for feat_type in target_feature_types}
#         return target_feats
#
#     tracks_infos = []
#     for track in top_tracks['tracks']:
#         track_info = TrackInfo(
#             name=track['name'],
#             id=track['id'],
#             feats=_get_feats(track['id']),
#         )
#         tracks_infos.append(track_info)
#     return tracks_infos


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
    """Fetches, writes, and returns data."""

    os.system(f"curl --request GET \
      --url https://api.spotify.com/v1/{data_args} \
      --header 'Authorization: Bearer {access_token}' \
              > {output_filename}")

    return read_json(output_filename)



def get_similar_artists(artist_id: str, access_token: str) -> List[ArtistInfo]:
    similar_artists = _get_data(output_filename='related_artists', access_token=access_token, data_args=f'artists/{artist_id}/related-artists')
    similar_artists_info = [
        ArtistInfo(
            name=similar_artist['name'],
            id=similar_artist['id'],
        ) for similar_artist in similar_artists['artists']
    ]
    return similar_artists_info




def get_top_tracks(artist_ids, access_token) -> List[TrackInfo]:
    top_tracks_infos = []
    for artist_id in artist_ids:
        top_tracks_info = get_top_tracks_info(artist_id=artist_id, access_token=access_token)
        top_tracks_infos += top_tracks_info
    return top_tracks_infos


def main():
    access_token = generate_access_token()
    led_zep_id = "36QJpDe2go2KgaRleHCDTp"
    artist_ids = [led_zep_id]
    similar_artists = get_similar_artists(led_zep_id, access_token)
    for artist in similar_artists:
        artist_ids.append(artist.id)
    top_tracks = get_top_tracks(artist_ids=artist_ids, access_token=access_token)










