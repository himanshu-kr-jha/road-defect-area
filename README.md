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

## New area for a job already in post_construction_output

`rescore` keeps every row of a finished job and adds the corrected area next to the old one.
It never adds or removes a row.
It analyses the whole video once, working out the coefficient in every analysed frame as above.
It then finds each stored row in that run and gives it the final area of the same tracked defect.

1. Export the job's rows, and look up its video and GPX, in the Photons database (read only):

   ```sql
   \copy (select * from post_construction_output where job_id = 'B8R5K2M9X4P7N3Q' order by id) to 'B8R5K2M9X4P7N3Q.csv' csv header
   select input_video_s3_uri, gpx_s3_uri from video_processing_jobs where job_id = 'B8R5K2M9X4P7N3Q';
   ```

2. Run it with the same model the job used.
   The keys in the table have no bucket, so put `s3://photons-bucket-s3/` in front of them:

   ```bash
   .venv/bin/python road_area.py rescore --rows B8R5K2M9X4P7N3Q.csv --job-id B8R5K2M9X4P7N3Q \
     --video s3://photons-bucket-s3/cog-geo-snap-videos/... --gpx s3://photons-bucket-s3/cog-geo-snap-videos/....gpx \
     --height 1.152 --model model.pt --out out/B8R5K2M9X4P7N3Q --aws-profile pothole-ro
   ```

The answer is `out/<job>/post_construction_output_<job>_new_area.csv`.
It has every column of the input, unchanged, plus these:

| Column | Meaning |
| --- | --- |
| `defect_area_m2_new` | The corrected area. It equals the old area when the row could not be corrected (see `note`) |
| `area_changed` | `yes` when the row got a new area |
| `new_over_old`, `estimated_cost_inr_new` | The change. Cost follows the area only for classes whose stored cost is a fixed rate per m2 |
| `method` | `tracked defect` (median of its views at 3-8 m) or `one view` (see below) |
| `confidence`, `views_used`, `spread`, `distance_m`, `length_m`, `width_m` | As in `defects.csv`, for the tracked defect |
| `same_defect_as_id` | This row is the same tracked defect as an earlier row |
| `note` | Why the old area was kept, or what to check |
| `video_frame`, `track_id`, `box_*`, `defect_image` | Where the row was found in the video, and its sheet in `defects/` |

The worker's own per-video CSV works too (`object_id, frame, defect_name, defect_area_m2, ...`, no `frame_url`).
Its frame numbers count from the start of the video, so the script reads those frames from `--video` and needs no evidence images:

```bash
.venv/bin/python road_area.py rescore --rows <video>_defect.csv --video <video>.mp4 --model model.pt --out out/<name>
```

When `--out` already holds a `run` of that video (for example `out/krrda_22423` from `run_clips.sh`), the video is not analysed again.

How each row is found:

- Its evidence image (`frame_url`) is matched to a video frame by its look, because frame numbers in the file names do not count from the start of a merged video.
- The model is run on the evidence image to get the row's box back.
  Production keeps only the area, which is the box's pixels divided by 145², so the box whose pixels fit is the row's box.
- That box is matched to the box drawn by the run within two analysed frames, which gives the tracked defect.
  The model sometimes calls one spot by another class in the next frames (pothole, then patching).
  When no box of the row's class is there, a box of another class on the same spot is used, the confidence drops one step, and `note` says so.
- Production saves a defect the first time the model sees it, often a weak detection far away.
  `rescore` therefore starts tracks at confidence 0.20 (`run` uses 0.35); a track that matches no row is simply not used.
- A row whose box the run still did not track is measured once, on its own frame, with that frame's horizon (`method` = `one view`).
  This is done only up to 8 m away: farther, a few pixels of error grow with the cube of the distance.
  Its confidence is at most `medium`, and `low` when the box is cut by the frame edge.
- If the stored area is smaller than the box (the part off the road was dropped), the new area is reduced in the same proportion.

The old area is kept for sign boards and garbage piles, which stand above the road.
It is also kept when the evidence image is not in the video (for example, a trimmed part of a merged video), and when the defect was never seen close enough.

A merged video holds several recordings, and its GPX holds one track segment per recording.
The script finds where the video jumps from one recording to the next.
It then lines each recording up with its GPX segment by matching the camera's own speed to the GPS speed (`speed match` in the output, 1 is perfect).
Each recording keeps its own f*H.

`rescore` saves the analysed video in `--out`.
Running it again on the same `--out`, for example with a corrected CSV, takes minutes, not the whole video again.
Pass `--rerun` to analyse the video again.
Analysed frames are kept in `--out/frames` for the defect images: about 3 GB for a 10-minute video.
`run.json` records whether the model ran on `cuda` or `cpu`, and how long the analysis took.

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
