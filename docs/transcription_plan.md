# Plan: improved transcription of the Cinedantan films

## Why

The current WhisperX transcripts (`purpureus:/home/gshen/work_dir/flik/data/transcripts/`, January 2026) were made without a fixed language. WhisperX then guesses the language from the first 30 seconds of each file, which in old films is often title music. A check of all 2,055 transcript files (November 2026) found:

- 349 files labelled Latin (`la`). 270 of them are ordinary English dialogue. The pipeline keeps only files labelled `en`, so about 165 hours of English speech are left out.
- Some English films were transcribed into another language, for example `UnderTexasSkies` and `Gold` into Bulgarian, `MurderatDawn` into Russian.
- English films labelled Telugu (`te`), such as `MoonoftheWolf` and `FrontierPonyExpress`, came out mostly as "." segments. Minutes of dialogue are missing.
- Silent films (265 files with almost no speech, 83% made before 1929) contain stock phrases Whisper produces on music: "Thank you.", "Абонирайте се!", "Teksting av Nicolai Winther".

Where the language was detected correctly, the English text looks accurate in spot checks.

## Procedure

### 1. Language from metadata

- Look up the original language of each film in Wikidata (property P364), using the IMDb ID from `data/in/cinedantan_movies.json`. In a sample of 200 films, 182 (91%) had an entry. Internet Archive metadata rarely has a language field.
- Mark likely silent films by year (before 1929) and genre.
- The original language is not always the language of the copy we have: some copies are English dubs, and silent films have only intertitles. The metadata gives the expected language; step 2 checks it.

### 2. Language from the audio

- Find stretches of speech with a voice activity detector (WhisperX includes one).
- Take about 10 windows of 30 seconds, spread over the film and chosen for having the most speech. Skip the first and last few minutes (titles, credits).
- Run Whisper language detection on each window and combine the probabilities. Keep the per-window results so that films with more than one language can be found.

### 3. Language decision

| Case | Decision |
|---|---|
| Audio confident and matches metadata | Use it |
| Audio confident, differs from metadata | Use the audio language; flag as a possible dub |
| Audio uncertain | Use the metadata language |
| Too little speech | Mark as silent; do not transcribe |

"Confident" is to be set after the pilot (step 7), for example a combined probability above 0.8 with most windows agreeing. All decisions go in one table: file, metadata language, audio language, probabilities, decision, reason.

### 4. Transcription

- Run WhisperX with `language` set to the decision from step 3.
- Load the alignment model for that language, or for now transcribe only English files.

### 5. Removing invented text

- Use stricter voice detection settings and the quality thresholds Whisper provides.
- After transcription, remove segments that are only punctuation, stock phrases ("Thank you.", "Subscribe", subtitle credits) or long runs of the same line.

### 6. Checks on the output

- Detect the language of each transcript from its text and compare it with the decision.
- Store per-file figures: share of filler, repetition, word alignment scores, amount of speech.
- Flag files that fail instead of dropping them silently.

### 7. Pilot

Run the procedure on:

- the 349 files labelled `la`;
- the files labelled `te` that are mostly blank;
- the English films transcribed into another language;
- 50 random English files, to check that nothing gets worse.

Compare with the current transcripts on language agreement, filler share and blank stretches. To measure word accuracy, transcribe about 30 one-minute clips by hand and compute the word error rate.

### 8. Full run

- Run steps 1–3 on all files (cheap).
- Re-transcribe the files whose decided language differs from the current label or that fail the checks; roughly 400–500 files based on the analysis above.
- Write to a new folder (for example `transcripts_v2/`) and leave the current transcripts unchanged.

## Open questions

- Keep non-English films for later use, or drop them?
- Treat English dubs of foreign films as English data?
- Re-transcribe all files for consistency, or only the affected ones?
