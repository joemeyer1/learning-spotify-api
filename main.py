#!usr/bin/env python3


from utilities.clustering_helpers import cluster_tracks, write_clusters_to_csv, plot_tracks_with_clusters
from utilities.data_manager import DataManager
from utilities.music_info import ArtistInfo


def cluster_artist_and_co_tracks(seed_artist_info: ArtistInfo = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")):
    """Clusters top tracks for seed artist and similar artists, and writes clustering to csv."""

    data_manager = DataManager()

    artists = [seed_artist_info] + data_manager.fetch_similar_artists(seed_artist_info.id)
    top_tracks = data_manager.fetch_top_tracks(artists=artists)

    labels = cluster_tracks(top_tracks, min_cluster_size=4)

    plot_tracks_with_clusters(top_tracks, labels)
    write_clusters_to_csv(top_tracks, labels)


if __name__ == '__main__':
    cluster_artist_and_co_tracks()
