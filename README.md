# Road defect area from a phone video

Turns the pixel area of each detected defect into square metres that do not change with distance, and checks the result against a tape.
Nothing here touches vision-stack, the database, S3 (other than downloading an input) or the VM.

## Quick start on a GPU machine

```bash
git clone https://github.com/himanshu-kr-jha/road-defect-area.git
cd road-defect-area
./setup.sh                                   # makes .venv; prints the GPU it found
MODEL=/path/to/model.pt ./run_clips.sh clips/krrda_2026-11-26.txt
```

The model weights are not in this repository.
Use the production defect model, `post_cons/3/model.pt`.
On the Photons VM, `run_clips.sh` finds it at `/home/ubuntu/vision-stack/infrastructure/triton/models/post_cons/3/model.pt` without `MODEL=`.
Reading `s3://` links needs the `aws` CLI with read access to the bucket.
ffmpeg is used from the system if present, otherwise from the `imageio-ffmpeg` package.

Each clip's results land in `out/<name>/`: open `report.html` and the `defects/` images.
For a new video, add a line `<name> <video> <gpx> <height_m>` to a file in `clips/` and run it the same way.
For Milestone B, fill `truth.csv` (below) and run `.venv/bin/python road_area.py evaluate --out out/<name> --truth truth.csv`.
On a laptop without a GPU, `setup.sh` installs CPU torch instead; `requirements.lock.txt` pins the CPU versions this was tested with.

## Record

1. Mount the phone in landscape on the bike, as for a normal survey, and do not touch it again until you finish.
2. Measure **H**: the centre of the camera lens straight down to the road, in metres, with the rider sitting on the bike.
   Measure twice; a 1 cm error is about 1%.
3. Use the main lens at 1x, and turn video stabilisation off if the phone allows it.
   Use the same app as usual, so that the GPX starts and stops with the video.
4. Ride straight and steadily at 10 to 15 km/h, so each defect is seen from far to near.
   Record 20 to 30 seconds if you can.
   10 seconds works, but the GPX scale is about 2% (up to 6.5%) less certain.
5. Make a second pass over the same defects.
   It is the test of whether the same pothole gets the same area in another video.

## Measure with the tape

Measure **at least 5 defects, better 8 to 10**, of different sizes and on both sides of the road.
Draw a chalk line around each edge first and measure the chalk, so that the video and the tape show the same edge.

- **length_m**: longest extent along the road.
- **width_m**: longest extent across the road.
- **area_m2**: optional real area. Measure the width every 10 cm along the length; area = sum of widths x 0.10.
- Include one plain chalk rectangle, 1.0 m x 1.0 m, as a defect named `REF`.
  Its size is exact, so it separates scale errors from the question of where a pothole's edge is.
- Photograph each one with the tape and its name (P1, P2, ...) next to it.

Copy `truth_template.csv` to `truth.csv` and fill it in, in metres.

## Run

```bash
cd road-defect-area
.venv/bin/python road_area.py run \
  --video VIDEO --gpx GPX --height 1.152 \
  --model /path/to/model.pt --out out/test1
```

`VIDEO` and `GPX` can be a file, an http(s) link, or `s3://...` with `--aws-profile`.
A 10-second clip takes about 70 to 80 seconds on this laptop's CPU; a GPU is used if there is one.

Then open `out/test1/label.html`, click around each measured defect in 3 to 5 frames between about 3 and 8 m away, name it as in `truth.csv`, and download `outlines.json`.
Outline the `REF` square too.
The model does not need to have found a defect for this.

```bash
.venv/bin/python road_area.py evaluate --out out/test1 --truth truth.csv --outlines ~/Downloads/outlines.json
```

This takes seconds, because the model is not run again.
Open `out/test1/report.html`.
Without drawing outlines, put the IDs from `annotated.mp4` into the `track_ids` column instead.

## Outputs

| File | What it holds |
| --- | --- |
| `report.html` | The answer: is the coefficient constant, Milestone A, Milestone B |
| `defects.csv` | One row per defect: old area, new area, length, width, confidence, image |
| `defects/` | One image per defect: the view used, its final new and old area, and the same defect from far to near with old and new area at each distance (and the tape area once `--truth` is given) |
| `coefficients.csv` | One row per defect per frame: the coefficient, on the go and with the whole clip |
| `per_frame.csv` | Horizon, roll, speed, and f*H in each analysed frame |
| `tape_comparison.csv` | Tape against production, model boxes and hand outlines |
| `annotated.mp4` | The video with `old x coefficient = new` on every box |
| `summary.json`, `run.json` | The same results for scripts |

## How the coefficient is found

```
coefficient = 145^2 x (f*H) x H / (y - v0)^3
area        = pixels / 145^2 x coefficient
```

| Part | Where it comes from | Changes |
| --- | --- | --- |
| H | Your tape | Fixed per mounting |
| f*H | GPX distance against how far the road moved in the video | Fixed per phone and mounting; estimated once per video |
| v0 | The horizon, from the road surface rebuilt out of tracked points | Every frame |
| y | The box | Every frame |

So it is not one constant.
It is worked out while the model runs, from past frames only (`coefficient_live`), and once more at the end with the whole clip (`coefficient`).
The formula is the simple form: the code uses the full tilted road plane over the whole box.

## Options worth knowing

- `--fh`: use f*H from an earlier report instead of this clip's GPX. Once a tape test has fixed it for a phone and mount, this is the better choice.
- `--trim-y`: take pixels off the top and bottom of each box. Pick it on one video and confirm it on another.
- `--imgsz`: model input size. Production uses 640.
- `--gpx-offset`: for a GPX that was not started together with the video.

## Tests

```bash
.venv/bin/python -m pytest -q
```

They render a road with known pothole sizes, a bouncing camera and a focal length the script is not told, and run the whole script on it.
Perfect boxes, perfect GPS and a flat road are assumed there, so they prove the maths and not the model.
