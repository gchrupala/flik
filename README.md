# flik


## Datasets

### Cinedantan

Data: https://github.com/casbah-ma/cinedantan/blob/master/public/database/movies.json

Public domain movies, mainly from internetarchive.

- Number of movies: +2100
- Total duration: 2437 hours
- Total runtime downloaded on disk: 2158.67 hours.
- Size on disk: 705G

The film list is an older version of the upstream `movies.json`, saved as `data/in/cinedantan_movies.json` (2,138 entries).

Locations on `purpureus.uvt.nl`:

- Films: `/home/gchrupal/flik/data/out/cinedantan/`, one folder per Internet Archive identifier (2,136 folders). Downloaded with `download.py`.
- Transcripts: `/home/gshen/work_dir/flik/data/transcripts/` (1.4 GB, January 2026). WhisperX output for 2,057 videos from 1,969 films. Each video has a `.json` file (segments with timestamps), a `.srt` file and a `_language.txt` file. Detected language: English for 1,519 videos, Latin for 349 (probably misdetections on films without speech), other languages for the rest.
- Manifests and CLIP scores from the processing pipeline (`video-grounding` branch): `/home/gshen/work_dir/flik/data/`. For example, `batch_manifest.json` lists the 1,520 English videos and `expanded_manifest_segments.json` has 201,743 segments of 3–10 s.

### Large Scale Movie Description Challenge (LSMDC)

Data: https://sites.google.com/site/describingmovies/download?authuser=0

Seems like very short clips, no dialog.

### MSRVTT
Data: https://cove.thecvf.com/datasets/839

Web video clips.

Total duration: 41.2 hours

### Movienet

Data: https://movienet.github.io/#!

Only annotations available so far.


### Condensed movies

Data: https://github.com/m-bain/CondensedMovies/tree/master


