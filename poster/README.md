# BladeVision 3.0 — A2 poster

Deliverables (A2 portrait, 420 × 594 mm):

| file | use |
|---|---|
| `BladeVision_3.0_A2_200dpi.png` | print file (3307 × 4677 px, 200 dpi) |
| `BladeVision_3.0_A2_200dpi.jpg` | same, JPEG q94 for emailing / printing services |
| `BladeVision_3.0_A2_preview.png` | half-size preview for screens |
| `build_poster.py` | regenerates all of it; every number is traceable to the repo |

Regenerate:

```bash
python build_poster.py --dpi 96  --out proof.png     # quick proof
python build_poster.py --dpi 200                      # print file
```

Layout: masthead with the IIT Mandi mark and the four headline numbers, a
five-item spec bar (stage / Y axis / camera / lens / tests), then nine numbered
cards — problem, pipeline, software map, data flow, measured geometry, result,
impact, engineering rules, and a full-width serpentine band — plus credits. Type is Archivo Black / Barlow / Barlow Condensed /
Space Mono (SIL OFL, in `assets/fonts`). Cyan = system, teal = process,
amber = anything measured on the rig.

Print notes: the A2 is designed with a 15 mm margin, so a 3–5 mm bleed trim is
safe. The background is a near-flat navy with a very light grain — if the
printer bands, ask for it to be printed at 200 dpi on matte stock rather than
downsampled.

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
