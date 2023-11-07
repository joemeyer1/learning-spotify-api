#!usr/bin/env python3


from utilities.clustering_manager import ClusteringManager
from utilities.data_manager import DataManager
from utilities.music_info import ArtistInfo


def cluster_artist_and_co_tracks(seed_artist_info: ArtistInfo = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")):
    """Clusters top tracks for seed artist and similar artists, and writes clustering to csv."""

    data_manager = DataManager()

    artists = [seed_artist_info] + data_manager.fetch_similar_artists(seed_artist_info.id)
    top_tracks = data_manager.fetch_top_tracks(artists=artists)

    clustering_manager = ClusteringManager(tracks=top_tracks)
    clustering_manager.cluster_tracks(min_cluster_size=4)

    clustering_manager.plot_tracks_with_clusters()
    clustering_manager.write_clusters_to_csv()


if __name__ == '__main__':
    cluster_artist_and_co_tracks()
