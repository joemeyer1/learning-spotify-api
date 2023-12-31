#!usr/bin/env python3


import os
import unittest

import numpy as np

from utilities.clustering_manager import ClusteringManager
from utilities.data_manager import DataManager
from utilities.music_info import ArtistInfo, TrackInfo
from utilities.target_feature_types import target_feature_types


class TestSpotifyProj(unittest.TestCase):

    def test_fetch_top_tracks(self):
        data_manager = DataManager()
        artists = [ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")]
        top_tracks = data_manager.fetch_top_tracks(artists=artists)
        self.assertGreater(len(top_tracks), 0)
        self.assertEqual(type(top_tracks[0]), TrackInfo)
        self.assertEqual(type(top_tracks[0].name), str)
        self.assertEqual(type(top_tracks[0].id), str)
        self.assertEqual(type(top_tracks[0].artist), str)
        self.assertEqual(type(top_tracks[0].feats[target_feature_types[0]]), float)

    def test_convert_tracks_to_df(self):
        data_manager = DataManager()
        artists = [ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")]
        top_tracks_df = data_manager.fetch_top_tracks_df(artists=artists, filename='tracks.csv')
        self.assertGreater(len(top_tracks_df), 0)
        self.assertTrue(np.all(top_tracks_df.columns == ['track', 'artist'] + target_feature_types))

    def test_fetch_similar_artists(self):
        data_manager = DataManager()
        led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
        similar_artists = data_manager.fetch_similar_artists(led_zep_info.id)
        self.assertGreater(len(similar_artists), 0)
        self.assertEqual(type(similar_artists[0]), ArtistInfo)
        self.assertEqual(type(similar_artists[0].name), str)
        self.assertEqual(type(similar_artists[0].id), str)

    def test_normalization(self):
        x = np.array([[1, 2, 3],
                      [4, 5, 6.]])
        norm_x = ClusteringManager.normalize_data(x)
        # currently we normalize mean to 0 but we don't normalize std
        self.assertTrue(np.all(norm_x == np.array([[-1.5, -1.5, -1.5],
                                                   [1.5, 1.5, 1.5]])))

    def test_clustering(self):
        data_manager = DataManager()
        led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
        artists = [led_zep_info] + data_manager.fetch_similar_artists(led_zep_info.id)
        top_tracks = data_manager.fetch_top_tracks(artists=artists)

        clustering_manager = ClusteringManager(tracks=top_tracks)
        labels = clustering_manager.cluster_tracks(min_cluster_size=4)
        self.assertEqual(len(labels), len(top_tracks))

    def test_cluster_writing(self):
        data_manager = DataManager()
        led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
        artists = [led_zep_info] + data_manager.fetch_similar_artists(led_zep_info.id)
        top_tracks = data_manager.fetch_top_tracks(artists=artists)

        clustering_manager = ClusteringManager(tracks=top_tracks)
        labels = clustering_manager.cluster_tracks(min_cluster_size=4)

        artist_clusters_df = clustering_manager.write_clusters_to_csv()
        self.assertEqual(len(artist_clusters_df), len(set(labels)))

    def test_clustering_scatterplot(self):
        data_manager = DataManager()
        led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
        artists = [led_zep_info] + data_manager.fetch_similar_artists(led_zep_info.id)
        top_tracks = data_manager.fetch_top_tracks(artists=artists)

        clustering_manager = ClusteringManager(tracks=top_tracks)
        clustering_manager.cluster_tracks(min_cluster_size=4)

        clustering_manager.plot_tracks_with_clusters()
        self.assertTrue(os.path.exists('tests.html'))

    def test_artist_spreads(self):
        data_manager = DataManager()
        led_zep_info = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")
        artists = [led_zep_info] + data_manager.fetch_similar_artists(led_zep_info.id)
        top_tracks = data_manager.fetch_top_tracks(artists=artists)

        clustering_manager = ClusteringManager(tracks=top_tracks)
        labels = clustering_manager.cluster_tracks(min_cluster_size=4)

        artist_spreads = clustering_manager.get_artist_spreads_over_clusters()
        num_clusters_per_artist = {artist: len(cluster_spread) for artist, cluster_spread in artist_spreads.items()}

        self.assertLess(max(num_clusters_per_artist.values()), len(set(labels)))


if __name__ == '__main__':
    unittest.main()
