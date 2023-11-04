# learning-spotify-api-private
 rough draft - learning spotify api assignment

1. You should get the top tracks for Led Zeppelin, and the audio features for all those top
tracks.

    - See `assignment_answers.get_top_led_zep_tracks_and_feats()`<br><br>
      
2. You should get the related artists for Led Zeppelin, each of those artists’ top tracks, and
the audio features for each of those tracks.

    - See `assignment_answers.get_similar_artist_tracks_and_feats()`<br><br>

3. You should pull the results together from all of the above into one dataframe.

    - see `assignment_answers.get_led_zep_and_co_tracks_and_feats()`<br><br>

4. Use one clustering technique to identify clusters of similar songs, 

    - See `assignment_answers.cluster_led_zep_and_co_songs()`<br>
    
    I used HDBSCAN, which builds  Minimum Spanning Tree (MST) then deletes weakest edges until clusters are separated.
    HDBSCAN can handle weirdly shaped clusters of different sizes, and requires minimal hyper-parameter tuning.
    In practice I find HDBSCAN often aligns better with my intuition compared with other clustering algorithms.<br><br>
    
    I first normalized data so that each column would have a mean of 0.
    I chose not to normalize standard deviation, because the clusterings yielded by normalizing it were disappointing.
    
    I set min_cluster_size to 4 because intuitively a set of 4 songs is about the minimum required to observe/assess meaningful patterns.
    Given min_cluster_size = 4, the precise number of clusters yielded by HDBSCAN may vary. A recent run yielded 24 clusters (after secondary step - see below).
    After initial HDBSCAN clustering, outliers are lumped together in their own cluster (labelled '-1'). I run one more HDBSCAN iteration to cluster these outliers.<br><br>
    
    The clustering yielded by this algorithm intuitively makes sense to me.
    
    We have "Stairway to Heaven" right next to "Riders on the Storm", "Light My Fire", and "LA Woman".
    These songs are all poetic/lyric-heavy, complex, atmospheric, with lyrics faintly gesturing towards divinity delivered in dark and menacing tones.
    They're songs you *could* dance to, but only if you really needed to compensate for your landlord not letting you smoke cigarettes in your apartment.
    
    Meanwhile, right beneath that cluster we have "Whole Lotta Love", "Smoke on the Water", and "Eminence Front", which are much bluesier, more distorted, and guitar/rhythm rather than lyric focused.
    
    Another cluster holds "Immigrant Song", "Crosstown Traffic", "Break On Through (To the Other Side)", and "Call On Me", which all involve high-pitched screaming.<br><br>
    
    Conversely, if we look at the column view (comparing tracks across clusters), we see "Good Times Bad Times" next to "Going to California", "Voodoo Child (Slight Return)", and "Sweet Home Alabama".
    These songs obviously do not go together. "Going to California" is soft and sad, "Sweet Home Alabama" is triumphant and taunting, "Voodoo Child" is an acid freakout.
    
    The column next to it isn't quite as bad, but still we have "Behind Blue Eyes" (self-pitying and deliberate, lyrics-focused), next to "Purple Haze" (egoless, wild, guitar-focused) and "Children of the Grave" (ominous, muddy, relatively minimalist).
    
    So not only does the clustering put similar tracks together, it also separates tracks that are different.<br><br>
    
    One weakness of this clustering approach is that as part of the MST pruning, outliers are stranded and ultimately grouped together.
    You can see them in the cluster labelled "-1".
    I partially dealt with this by running a second HDBSCAN iteration to re-cluster the outliers orphaned by the first run.
    After running a second iteration, there is still a (smaller) outlier cluster.
    This outlier cluster (still labelled "-1") could be merged into existing clusters (e.g. based on mean linkage).
    But it's not that bad. The songs in the outlier cluster, while perhaps not screaming out to be paired together, 
    still clash less than songs column-wise across clusters do. The outlier cluster holds songs like "Pinball Wizard", "Shine on You Crazy Diamond", and "Can't Stop Loving You", which have similar vocal (slightly strained, upper-mids registers) and chugging guitar styles.
    And it's not necessarily a bad thing to first identify the clusters that are 
    readily apparent, and then figure out what to do with the trickier leftover datapoints. That way you start with better clusters, 
    identify outliers (which is itself valuable), and have more flexibility and information to deal with them.<br><br>

5. Using this model, for each artist included, how many different clusters have at least one of their top tracks?

    - see `assignment_answers.get_num_clusters_per_artist()`<br>
    
    In this model, the number of clusters per artist is:
    
    artist_to_num_of_home_clusters = {<br>
            'Led Zeppelin': 9,<br>
            'Jimi Hendrix': 7,<br>
            'The Doors': 6,<br>
            'Deep Purple': 8,<br>
            'The Who': 9,<br>
            'Black Sabbath': 7,<br>
            'Cream': 6,<br>
            'Lynyrd Skynyrd': 9,<br>
            'Janis Joplin': 9,<br>
            'The Rolling Stones': 9,<br>
            'ZZ Top': 10,<br>
            'Pink Floyd': 9,<br>
            'Van Halen': 8,<br>
            'Jim Morrison': 7,<br>
            'Rush': 9,<br>
            'Ozzy Osbourne': 6,<br>
            'Robert Plant': 9,<br>
            'Steppenwolf': 9,<br>
            'Soundgarden': 8,<br>
            'The Animals': 7,<br>
            'Blue Öyster Cult': 9,<br>
    }
