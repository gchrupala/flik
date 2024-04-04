# flik


## Datasets

### Cinedantan
Open domain movies: https://github.com/casbah-ma/cinedantan/tree/master
List: https://github.com/casbah-ma/cinedantan/blob/master/public/database/movies.json
Number of movies: +2100
Total duration: 2437 hours

### Large Scale Movie Description Challenge (LSMDC)
Data: https://sites.google.com/site/describingmovies/download?authuser=0

Seems like very short clips, no dialog.

### MSRVTT
Data: https://cove.thecvf.com/datasets/839
Web video clips
Total duration: 41.2 hours

### Movienet
Data: https://movienet.github.io/#!
Only annotations available so far.



Set up the environment on the compute cluster.

<pre>
ssh purpureus.uvt.nl
srun --nodes=1 --pty /bin/bash -l
conda activate home
cd flik
source bin/activate
</pre>

