#!usr/bin/env python3


import unittest

import numpy as np

from main import normalize_data, generate_access_token, ArtistInfo, TrackInfo, get_top_tracks, convert_tracks_to_df, get_similar_artists, cluster_tracks, plot_data, write_artist_clusters, get_artist_spreads
from target_feature_types import target_feature_types



class TestSpotifyProj(unittest.TestCase):
    def test_normalization(self):
        x = np.array([[1, 2, 3],
                      [4, 5, 6.]])
        norm_x = normalize_data(x)
        # currently we normalize mean to 0 but we don't normalize std
        self.assertTrue(np.all(norm_x == np.array([[-1.5, -1.5, -1.5],
                                                   [1.5, 1.5, 1.5]])))

    def test_fetch_top_tracks(self):
        access_token = generate_access_token()
        artists = [ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")]
        top_tracks = get_top_tracks(artists=artists, access_token=access_token)
        self.assertGreater(len(top_tracks), 0)
        self.assertEqual(type(top_tracks[0]), TrackInfo)
        self.assertEqual(type(top_tracks[0].name), str)
        self.assertEqual(type(top_tracks[0].id), str)
        self.assertEqual(type(top_tracks[0].artist), str)
        self.assertEqual(type(top_tracks[0].feats[target_feature_types[0]]), float)

    def test_fetch_similar_artists(self):
        access_token = generate_access_token()
        led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
        similar_artists = get_similar_artists(led_zep_info.id, access_token)
        self.assertGreater(len(similar_artists), 0)
        self.assertEqual(type(similar_artists[0]), ArtistInfo)
        self.assertEqual(type(similar_artists[0].name), str)
        self.assertEqual(type(similar_artists[0].id), str)

    def test_convert_tracks_to_df(self):
        access_token = generate_access_token()
        artists = [ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")]
        top_tracks = get_top_tracks(artists=artists, access_token=access_token)
        df = convert_tracks_to_df(top_tracks, 'tracks.csv')
        self.assertGreater(len(df), 0)
        self.assertTrue(np.all(df.columns == ['track', 'artist'] + target_feature_types))

    def test_clustering(self):
        access_token = generate_access_token()
        led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
        artists = [led_zep_info] + get_similar_artists(led_zep_info.id, access_token)
        top_tracks = get_top_tracks(artists=artists, access_token=access_token)
        convert_tracks_to_df(top_tracks, 'tracks.csv')

        labels = cluster_tracks(top_tracks, min_cluster_size=4)
        self.assertEqual(len(labels), len(top_tracks))

    def test_cluster_writing(self):
        access_token = generate_access_token()
        led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
        artists = [led_zep_info] + get_similar_artists(led_zep_info.id, access_token)
        top_tracks = get_top_tracks(artists=artists, access_token=access_token)
        convert_tracks_to_df(top_tracks, 'tracks.csv')

        labels = cluster_tracks(top_tracks, min_cluster_size=4)

        artist_clusters_df = write_artist_clusters(top_tracks, labels)
        self.assertEqual(len(artist_clusters_df), len(set(labels)))


    def test_artist_spreads(self):
        access_token = generate_access_token()
        led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
        artists = [led_zep_info] + get_similar_artists(led_zep_info.id, access_token)
        top_tracks = get_top_tracks(artists=artists, access_token=access_token)

        labels = cluster_tracks(tracks=top_tracks, min_cluster_size=4)

        plot_data(top_tracks, labels)
        write_artist_clusters(top_tracks, labels)

        artist_spreads = get_artist_spreads(top_tracks, labels)
        num_clusters_per_artist = {artist: len(cluster_spread) for artist, cluster_spread in artist_spreads.items()}

        self.assertLess(max(num_clusters_per_artist.values()), len(set(labels)))


if __name__ == '__main__':
    unittest.main()
