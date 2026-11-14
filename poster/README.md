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

Layout: 11 numbered cards in three columns (problem, pipeline, software map,
data flow, measured geometry, result, impact, engineering rules, what's next,
serpentine motion, the rig), a masthead with the four headline numbers, a KPI
strip, and a credits band. Type is Archivo Black / Barlow / Barlow Condensed /
Space Mono (SIL OFL, in `assets/fonts`). Cyan = system, teal = process,
amber = anything measured on the rig.

Print notes: the A2 is designed with a 15 mm margin, so a 3–5 mm bleed trim is
safe. The background is a near-flat navy with a very light grain — if the
printer bands, ask for it to be printed at 200 dpi on matte stock rather than
downsampled.

Numbers on the poster come from: `main_code/AUTOFOCUS.md` (DoF 0.09 mm),
`main_code/SWEEP_GEOMETRY.md` (344 px / 68 %, 426 px / 78 %, 1064 px/mm,
0.90 mm recommendation, 3 columns instead of 6), `main3.py` (800 / 1600
steps per mm, 1920×1080, boundary triggers), and the test suites
(151 checks). Imagery: `assets/blade_hero.png` (rig), `assets/surface_macro.png`
(1:1 surface tile) are generated renders, not photographs of the rig.
