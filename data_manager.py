#!usr/bin/env python3


import json
import os
from typing import Dict, Any, List, Optional

import pandas as pd

from music_info import ArtistInfo, TrackInfo
from target_feature_types import target_feature_types


class DataManager:
    """Manages data fetching through Spotify API."""

    access_token: str  # key to access Spotify API

    def __init__(self):
        self.generate_access_token()

    def generate_access_token(self, access_token_filename: str = "access_token.json") -> str:
        """Generates, saves, and returns an access_token for Spotify's API."""

        os.system(f'curl -X POST "https://accounts.spotify.com/api/token" \
             -H "Content-Type: application/x-www-form-urlencoded" \
             -d "grant_type=client_credentials&client_id=41e74624046646428afaa5c8bca7173f&client_secret=d8d74e43da5e475c961090e14d2ecea7" > {access_token_filename}')

        self.access_token = self._read_json('access_token.json')['access_token']
        return self.access_token

    def fetch_top_tracks_df(self, artists: List[ArtistInfo], filename: Optional[str] = None) -> pd.DataFrame:
        top_tracks = self.fetch_top_tracks(artists)
        return self._convert_tracks_to_df(tracks=top_tracks, filename=filename)

    def fetch_top_tracks(self, artists: List[ArtistInfo]) -> List[TrackInfo]:
        """Returns artists' top tracks."""

        def _fetch_top_tracks_for_artist(artist: ArtistInfo) -> List[TrackInfo]:
            """Returns top tracks for single artist."""

            top_tracks = self._fetch_data(
                output_filename=f"{artist.id}_top_tracks.json",
                data_args=f"artists/{artist.id}/top-tracks?market=US",
            )

            def _fetch_feats(track_ids: str) -> List[Dict[str, Any]]:
                tracks_feats = self._fetch_data(
                    output_filename=f"track_{track_ids}feats",
                    data_args=f"audio-features?ids={track_ids}",
                )["audio_features"]
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
            tracks_feats = _fetch_feats(track_ids)
            for i, track_info in enumerate(tracks_infos):
                tracks_infos[i].feats = tracks_feats[i]
            return tracks_infos

        top_tracks_infos = []
        for artist in artists:
            top_tracks_info = _fetch_top_tracks_for_artist(artist=artist)
            top_tracks_infos += top_tracks_info
        return top_tracks_infos

    def fetch_similar_artists(self, artist_id: str) -> List[ArtistInfo]:
        similar_artists = self._fetch_data(output_filename='related_artists', data_args=f'artists/{artist_id}/related-artists')
        similar_artists_info = [
            ArtistInfo(
                name=similar_artist['name'],
                id=similar_artist['id'],
            ) for similar_artist in similar_artists['artists']
        ]
        return similar_artists_info

    def _fetch_data(self, output_filename: str, data_args: str = '') -> Dict[str, Any]:
        """Fetches and returns data."""

        os.system(f"curl --request GET \
          --url https://api.spotify.com/v1/{data_args} \
          --header 'Authorization: Bearer {self.access_token}' \
                  > {output_filename}")

        data = self._read_json(output_filename)
        os.system(f"rm {output_filename}")
        return data

    @staticmethod
    def _read_json(filename):
        with open(filename, 'r') as f:
            return json.loads(f.read())

    @staticmethod
    def _convert_tracks_to_df(tracks: List[TrackInfo], filename: Optional[str] = None) -> pd.DataFrame:
        """Returns a df of tracks, and writes to csv if filename is passed."""

        data = []
        for track in tracks:
            row = [track.name, track.artist] + track.get_feats(target_feature_types)
            data.append(row)
        df = pd.DataFrame(data, columns=['track', 'artist'] + target_feature_types)
        if filename is not None:
            df.to_csv(filename)
        return df
