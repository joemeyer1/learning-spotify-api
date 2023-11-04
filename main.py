#!usr/bin/env python3

from clustering_helpers import cluster_tracks, write_clusters_to_csv
from data_manager import DataManager
from music_info import ArtistInfo


def main(seed_artist_info: ArtistInfo = ArtistInfo(name='Led Zeppelin', id="36QJpDe2go2KgaRleHCDTp")):
    """Clusters top tracks for seed artist and similar artists, and writes clustering to csv."""

    data_manager = DataManager()

    artists = [seed_artist_info] + data_manager.fetch_similar_artists(seed_artist_info.id)
    top_tracks = data_manager.fetch_top_tracks(artists=artists)
    # convert_tracks_to_df(top_tracks, 'top_tracks.csv')  # write to df

    labels = cluster_tracks(top_tracks, min_cluster_size=4)

    # plot_data(top_tracks, labels)
    write_clusters_to_csv(top_tracks, labels)

    # get_artist_spreads_over_clusters(top_tracks, labels)


if __name__ == '__main__':
    main()
