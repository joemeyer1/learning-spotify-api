#!usr/bin/env python3

from clustering_helpers import cluster_tracks, write_artist_clusters
from data_manager import DataManager
from music_info import ArtistInfo


def main(seed_artist_info: ArtistInfo = ArtistInfo(name='The Beatles', id="3WrFJ7ztbogyGnTHbHJFl2")):
    """Clusters top tracks for seed artist and similar artists, and writes clustering to csv."""

    data_manager = DataManager()

    artists = [seed_artist_info] + data_manager.fetch_similar_artists(seed_artist_info.id)
    top_tracks = data_manager.fetch_top_tracks(artists=artists)
    # convert_tracks_to_df(top_tracks, 'top_tracks.csv')  # write to df

    labels = cluster_tracks(top_tracks, min_cluster_size=4)

    # plot_data(top_tracks, labels)
    write_artist_clusters(top_tracks, labels)

    # get_artist_spreads_over_clusters(top_tracks, labels)


if __name__ == '__main__':
    main()
