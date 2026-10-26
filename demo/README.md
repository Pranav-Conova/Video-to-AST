# README demo

`docs/demo.gif`, at the top of the main README, is rendered from this
[Remotion](https://www.remotion.dev/) project. The composition in `src/` plays 24 s
of the Dune trailer next to its compact EditScript and the real HTML report from
`editast analyze`, which it loads in an iframe. On every frame it moves a playhead
across the report's timeline, turns the timeline's pages, and outlines the shots,
transitions and effects under the playhead.

## Rebuild the GIF

Needs Node 20+ and the editast Python environment (see the main README).

```bash
cd demo
npm install
# The Dune Official Trailer, 1080p. If YouTube asks yt-dlp to sign in, the
# web_embedded client usually still works.
yt-dlp --js-runtimes node --extractor-args "youtube:player_client=web_embedded" \
  -f 137+140 -o dune_trailer.mp4 https://www.youtube.com/watch?v=n9xhJrPXop4
python prepare.py dune_trailer.mp4 --url https://www.youtube.com/watch?v=n9xhJrPXop4
npm run frames      # renders out/frames/*.png at 880x550
python make_gif.py  # writes ../docs/demo.gif
```

`prepare.py` analyses the video and puts it, its Edit AST, compact EditScript and
HTML report in `public/`. `npm run dev` opens Remotion Studio to preview the
composition.

The props in `src/Root.tsx` choose the files, the part of the video shown (`start`
and `end`, 72.7–96.9 s) and the last report lane shown (`lastLane`; lanes below it
are cropped so the text stays readable). To use another video, run `prepare.py` on
it and change the props. The layout assumes a landscape video.

## Licences

The Dune trailer is © Warner Bros. Pictures and Legendary. The GIF shows a short
excerpt of it to illustrate the analysis; it is not covered by this repository's
licence. Remotion is free for individuals and companies of up to 3 people; larger
companies need a [company license](https://www.remotion.dev/license).
