# BladeVision 3.0 — A2 poster

Deliverables (A2 portrait, 420 × 594 mm):

| file | use |
|---|---|
| `BladeVision_3.0_A2_200dpi.png` | print file (3307 × 4677 px, 200 dpi) |
| `BladeVision_3.0_A2_200dpi.jpg` | same, JPEG q94 for emailing / printing services |
| `BladeVision_3.0_A2_preview.png` | half-size preview for screens |
| `build_poster.py` | regenerates all of it; every number is traceable to the repo |

Imagery: `assets/blade_hero.png` (the rig — a generated render, whole blade and
stage, nothing cropped), `assets/real_mosaic.jpg` and `assets/real_sources.png`
(**real output** from one scan: the 5 680 x 12 697 px deliverable and the
per-pixel source map), and `assets/real_detail.png` (a 1:1 crop of that mosaic).
Only the hero is synthetic; the three scan images are the pipeline's own files.

Regenerate:

```bash
python build_poster.py --dpi 96  --out proof.png     # quick proof
python build_poster.py --dpi 200                      # print file
```

Layout: masthead with the IIT Mandi mark, the four headline numbers and a
whole-blade hero plate (fit, never cropped); a five-item spec bar (stage / Y axis
/ camera / lens / tests); eight numbered cards in three columns; then a bottom
band with the serpentine diagram (left) and two real full-blade images (right). Type is Archivo Black / Barlow / Barlow Condensed /
Space Mono (SIL OFL, in `assets/fonts`). Cyan = system, teal = process,
amber = anything measured on the rig.

Print notes: the A2 is designed with a 15 mm margin, so a 3–5 mm bleed trim is
safe. The background is a near-flat navy with a very light grain — if the
printer bands, ask for it to be printed at 200 dpi on matte stock rather than
downsampled.

The per-pixel source map is written by `global_stitch.compose()` as
`<prefix>_sources.png`: each canvas pixel is coloured by the frame the Potts-MRF
blend gave it (seed 7, so colours are stable between runs), red marks frames
placed by dead reckoning (`placed_by_prior`) and white marks label boundaries —
i.e. the seams. It is the diagnostic that answers "is that line a stitch error
or real surface?".

Optics are named on the poster as the rig has them: **Canon EOS 80D** body on a
**Questar QM-100** long-distance Maksutov-Cassegrain catadioptric microscope —
a 100 inch (2.54 m) standoff, which is why the `1.8 mm` field is quoted per
frame rather than as a magnification.

The institute mark is `assets/iit_mandi_logo.png` (white-background original,
trimmed, set on a white plate in the masthead). Replace it with a vector/CMYK
version for a commercial print run if the printer asks.

Numbers on the poster come from: `main_code/AUTOFOCUS.md` (DoF 0.09 mm),
`main_code/SWEEP_GEOMETRY.md` (344 px / 68 %, 426 px / 78 %, 1064 px/mm,
0.90 mm recommendation, 3 columns instead of 6), `main3.py` (800 / 1600
steps per mm, 1920×1080, boundary triggers), and the test suites
(151 checks). Imagery: `assets/blade_hero.png` (rig), `assets/surface_macro.png`
(1:1 surface tile) are generated renders, not photographs of the rig.
