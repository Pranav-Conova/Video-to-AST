# 10 viral car edits from YouTube Shorts

editast's analysis of 10 popular car edits, collected on 2026-10-25. They were
chosen by running eight yt-dlp searches ("car edit shorts", "car velocity edit",
"jdm edit phonk shorts", ...) and keeping the most-viewed results between 6 and 60
seconds long. Raw footage, ads and CGI clips were left out.

Each video has an Edit AST as JSON (`*.edit.json`) and as EditScript (`*.edit`).
`batch_summary.json` holds the cross-video patterns. The videos and HTML reports
are not included, because they contain other creators' footage. To regenerate them:

```bash
editast batch https://www.youtube.com/shorts/kDdC6N4Q8O8 ... -o out/yt   # trees + patterns
editast analyze https://www.youtube.com/shorts/kDdC6N4Q8O8               # one video + HTML report
```

## The videos

| Video | Views | Length | BPM | Shots | Cuts/min | Drop at | Cuts/min before → after drop | Cuts on beat |
|---|---|---|---|---|---|---|---|---|
| [Supra x Candestine](https://www.youtube.com/shorts/NRBvrNAw2uQ) | 9.4M | 48.1 s | 84.6 ² | 28 | 34 | 19.7 s (41%) | 3 → 55 | 37% ² |
| [One Dance velocity edit](https://www.youtube.com/shorts/kDdC6N4Q8O8) | 9.3M | 32.1 s | 95.6 ² | 11 | 19 | 7.0 s (22%) | 26 → 17 | 10% ² |
| [BMW M4 G82](https://www.youtube.com/shorts/xyxBs_QDHV4) | 4.7M | 11.9 s | 115.2 | 12 | 55 | 4.3 s (36%) | 0 → 87 | 73% |
| [Supra, "It's my blood"](https://www.youtube.com/shorts/X0ZPhwJwtnw) | 3.1M | 16.6 s | 91.9 | 13 | 43 | 8.6 s (52%) | 7 → 82 | 33% ³ |
| [Ratatatata Supra edit](https://www.youtube.com/shorts/p-X42xUa2wQ) | 2.9M | 16.6 s | 132.8 | 50 ¹ | 177 ¹ | 7.6 s (46%) | 228 ¹ → 133 | 41% |
| [Cars edit (FLOP.EDITZZ)](https://www.youtube.com/shorts/CFG8DtIt7X0) | 2.4M | 41.8 s | 117.8 | 21 | 29 | 15.4 s (37%) | 19 → 34 | 5% ⁴ |
| [BMW M3 G80 CapCut](https://www.youtube.com/shorts/tUjIwYeVlQc) | 1.6M | 25.5 s | 127.2 ² | 16 | 35 | 12.8 s (50%) | 9 → 61 | 27% ² |
| [Bugatti x Montagem](https://www.youtube.com/shorts/XxFyqmOzRNQ) | 1.5M | 22.2 s | 181.9 | 19 | 49 | 13.6 s (61%) | 40 → 63 | 78% |
| [Lastfragment, phonk drift](https://www.youtube.com/shorts/0tQf0jCw9ZM) | 1.1M | 21.2 s | 122.0 | 36 | 99 | 8.1 s (38%) | 15 → 151 | 97% |
| [BMW M4 cinematic phonk](https://www.youtube.com/shorts/YFNSDdrElfc) | 1.1M | 22.6 s | 136.0 | 11 | 27 | 7.9 s (35%) | 8 → 37 | 90% |

View counts are from the day of collection. Footnotes mark numbers that frame-by-frame
checks showed to be wrong or unverified (see [Accuracy](#accuracy)).

## What these edits have in common

- **Build, then drop.** In 9 of 10 videos the music's first drop comes at 35–61% of
  the runtime, and the cutting speeds up there. Leaving out the over-counted video,
  the median goes from about 9 cuts/min before the drop to 61 after it. The One
  Dance velocity edit is the exception: a slow, cinematic cut throughout.
- **A long opening.** 6 of 10 videos hold their first shot for 4 s or more. The
  openings are a black title card with animated text, a channel logo, or one slow
  cinematic shot. The Supra x Candestine edit spells out "S U P R A" and a quote on
  black for 19 s before its first cut.
- **Cuts on the beat.** Where the beat grid is reliable, 73–97% of cuts land on a
  beat. In 7 of 10 videos the median shot is 0.5–1.0 s long, one to two beats
  at these tempos.
- **Constant camera motion.** Every video uses zooms (14 per video on average),
  pans (7) and rotations (4). 7 of 10 use speed ramps, and 7 of 10 use
  flash frames.
- **Few transition types.** Hard cuts dominate. Fades to black appear in 7 videos,
  and flash and whip transitions in 5 each.
- **Text.** Branding is common: watermarks ("SID.FX", "C.SHAN EDITZ", "SAGAR EDITZ"),
  animated subscribe buttons, and end cards ("THANKS FOR WATCHING / DON'T FORGET TO
  SUBSCRIBE"). Lyric captions are less common. 4 of 10 open with text in the first second.

## Accuracy

The suspicious results were checked frame by frame. The numbers above keep what
the analyzer reported; these are the corrections:

1. **Glitch transitions are over-cut.** In the Ratatatata Supra edit, static-noise
   glitch sequences and white flash frames between lyric cards are split into
   2–6-frame "shots". In the two stretches checked (0.45–1.0 s and 4.1–4.45 s),
   12 reported cuts stand for 2–4 real changes, so its shot count and opening pace
   are inflated. The fast montage after the drop is real,
   including genuine 2-frame flicker cuts.
2. **Off-beat lock.** In the FLOP.EDITZZ edit, the tempo matches the cut grid
   (118 BPM), but the beat tracker locked onto the off-beats. Most cuts fall at
   0.6–0.75 of a beat, so its 5% cut-on-beat is wrong.
3. **Unverified tempo.** For the Supra x Candestine, One Dance and BMW M3 G80
   edits, the detected tempo and the cut timing disagree. "One Dance" is 104 BPM in
   the original; 95.6 would fit a slowed version, but nobody has listened to check.
4. **Probable eighth-note cutting.** In the "It's my blood" Supra edit, 83% of the
   cuts fit a 185 BPM grid, twice the detected tempo, so it likely cuts on
   half-beats. Its 33% on-beat figure would then understate how tightly it is synced.
5. **Scene text read as an overlay.** In the BMW M3 G80 edit, the car's licence
   plate is reported as text overlays.
6. **Correct despite looking odd:** the 19 s single-shot opening of Supra x
   Candestine, and the "AAAA…"/"AEAEAE…" text in the "It's my blood" edit, which is
   a real typed-out lyric.

On real music, cuts land 10–50 ms before the detected beats. On synthetic click
tracks the detector is within 5 ms, so this offset comes from the footage or the
editors, not the detector, and it is inside the 70 ms on-beat tolerance.
