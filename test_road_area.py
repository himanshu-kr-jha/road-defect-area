"""Run: .venv/bin/python -m pytest -q

The end-to-end test renders a road video with exactly known defect sizes, a bouncing camera and a focal
length the script is not told, then checks the areas the script reports."""
import json
import math
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import pandas as pd
import pytest

import road_area as ra

F_TRUE, H_TRUE, SPEED, FPS, N, W, HH = 1500.0, 1.2, 4.0, 30.0, 300, 1920, 1080
DEFECTS = {"P1": (0.6, 14.0, 0.50, 0.40), "P2": (-0.9, 20.0, 0.90, 0.70), "P3": (0.2, 27.0, 0.35, 0.60), "P4": (-0.3, 34.0, 1.20, 0.80)}  # across, along, length, width


def pose(i):
    pitch = math.radians(6 + 1.2 * math.sin(i / 9) + .5 * math.sin(i / 3.1))
    roll = math.radians(1.0 + .7 * math.sin(i / 13))
    yaw = math.radians(.6 * math.sin(i / 25))
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    return Rz @ Rx @ Ry                                    # world (x right, y down, z forward) -> camera


def project(i, xz):
    """Road points (across, along) in metres -> pixels in frame i."""
    X = np.c_[xz[:, 0], np.full(len(xz), H_TRUE), xz[:, 1] - SPEED * i / FPS] @ pose(i).T
    return np.c_[X[:, 0] / X[:, 2] * F_TRUE + W / 2, X[:, 1] / X[:, 2] * F_TRUE + HH / 2], X[:, 2]


def outline(name):
    cx, cz, ln, wd = DEFECTS[name]
    a = np.linspace(0, 2 * math.pi, 40, endpoint=False)
    return np.c_[cx + wd / 2 * np.cos(a), cz + ln / 2 * np.sin(a)]


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("scene")
    rng = np.random.default_rng(0)
    ppm, xw, zw = 60, 16.0, 75.0                           # road texture, pixels per metre
    tex = cv2.GaussianBlur(rng.normal(120, 60, (int(zw * ppm), int(xw * ppm))).astype(np.float32), (0, 0), 1.2)
    tex = (tex + cv2.resize(rng.normal(0, 18, (300, 64)).astype(np.float32), (tex.shape[1], tex.shape[0]))).clip(0, 255).astype(np.uint8)
    for n in DEFECTS:
        cv2.fillPoly(tex, [np.round((outline(n) + [xw / 2, 0]) * ppm).astype(int)], 40)
    sky = cv2.GaussianBlur(rng.normal(150, 70, (3000, 5000)).astype(np.float32), (0, 0), 3).clip(0, 255).astype(np.uint8)
    out = cv2.VideoWriter(str(d / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, HH))
    K, Kt = np.array([[F_TRUE, 0, W / 2], [0, F_TRUE, HH / 2], [0, 0, 1]]), np.array([[F_TRUE, 0, 2500], [0, F_TRUE, 1500], [0, 0, 1]])
    for i in range(N):
        R = pose(i)
        z0 = SPEED * i / FPS
        T = np.array([[1 / ppm, 0, -xw / 2], [0, 0, H_TRUE], [0, 1 / ppm, -z0]])          # texture pixel -> world
        G = K @ R @ T
        far = cv2.warpPerspective(sky, K @ R @ np.linalg.inv(Kt), (W, HH))
        lo = int((z0 + 1.0) * ppm)                          # only what is in front of the camera
        sub = tex[lo:int((z0 + 45) * ppm)]
        road = cv2.warpPerspective(sub, G @ np.array([[1, 0, 0], [0, 1, lo], [0, 0, 1.0]]), (W, HH), flags=cv2.INTER_AREA)
        mask = cv2.warpPerspective(np.full_like(sub, 255), G @ np.array([[1, 0, 0], [0, 1, lo], [0, 0, 1.0]]), (W, HH))
        far[mask > 127] = road[mask > 127]
        out.write(cv2.cvtColor(far, cv2.COLOR_GRAY2BGR))
    out.release()
    t0 = datetime(2026, 11, 26, 6, 0, tzinfo=timezone.utc)
    pts = "".join(f'<trkpt lat="{13 + SPEED * s / 111320:.9f}" lon="77.5"><time>{(t0 + timedelta(seconds=s)).isoformat()}</time></trkpt>' for s in range(11))
    (d / "g.gpx").write_text(f"<gpx><trk><trkseg>{pts}</trkseg></trk></gpx>")
    pd.DataFrame([dict(name=n, length_m=v[2], width_m=v[3], area_m2=math.pi / 4 * v[2] * v[3], track_ids="") for n, v in DEFECTS.items()]).to_csv(d / "truth.csv", index=False)
    q = []
    for n in DEFECTS:
        for i in range(0, N, 3):
            p, z = project(i, outline(n))
            if 4 < z.mean() < 8 and (z > 0).all() and (p[:, 1] < HH - 5).all() and (abs(p[:, 0] - W / 2) < W / 2 - 5).all():
                q.append(dict(name=n, frame=i, points=p.tolist()))
    (d / "outlines.json").write_text(json.dumps(q))
    return d


class PerfectDetector:
    """Stands in for the model: the exact bounding box of every defect, in whichever frame it is shown."""
    device = "cpu"

    def __init__(self, *_):
        self.k = 0

    def __call__(self, frame, conf):
        i, out = self.k * 3, []
        self.k += 1
        for n in DEFECTS:
            p, z = project(i, outline(n))
            if z.min() > 1 and z.max() < 18 and p[:, 1].min() < HH:
                b = [*np.clip(p.min(0), 0, [W, HH]), *np.clip(p.max(0), 0, [W, HH])]
                if b[2] - b[0] > 6 and b[3] - b[1] > 4:
                    out.append(dict(label="pothole", conf=.9, box=[float(v) for v in b]))
        return out


@pytest.fixture(scope="module")
def result(scene):
    ra.Detector, keep = PerfectDetector, ra.Detector
    try:
        ra.main(["run", "--video", str(scene / "v.mp4"), "--gpx", str(scene / "g.gpx"), "--height", str(H_TRUE), "--model", "none.pt",
                 "--out", str(scene / "out"), "--truth", str(scene / "truth.csv"), "--outlines", str(scene / "outlines.json")])
    finally:
        ra.Detector = keep
    return scene / "out", json.loads((scene / "out" / "summary.json").read_text()), json.loads((scene / "out" / "run.json").read_text())


def test_scale_is_recovered_although_the_focal_length_is_not_given(result):
    assert result[2]["working_focal"] == 1400 and abs(result[2]["fh"] / (F_TRUE * H_TRUE) - 1) < .06


def test_horizon_follows_the_bouncing_camera(result):
    P = pd.read_csv(result[0] / "per_frame.csv")
    true = []
    for i in P.frame:
        R = pose(i)
        n = R @ np.array([0, 1, 0])
        true.append(HH / 2 - F_TRUE * n[2] / n[1])
    e = (P.horizon_row - true).dropna()
    assert abs(e.median()) < 12 and (e - e.median()).abs().median() < 8
    assert P.horizon_row.max() - P.horizon_row.min() > 40              # it really did move


def test_milestone_a(result):
    a = result[1]["milestone_a"]
    assert a["road_patches"]["area_new"]["spread"] < .06 < .3 < a["road_patches"]["area_old"]["spread"]
    assert a["detector_boxes"]["area_new"]["spread"] < .15 < a["detector_boxes"]["area_old"]["spread"]


def test_milestone_b_against_known_sizes(result):
    b = result[1]["milestone_b"]
    assert b["hand_outlines"]["n"] == 4 and b["hand_outlines"]["worst_error"] < .15
    assert b["model_boxes"]["n"] == 4 and b["model_boxes"]["worst_error"] < .20
    assert b["production_today"]["median_error"] > .3


def test_defects_are_matched_to_the_tape_through_their_outlines(result):
    T = pd.read_csv(result[0] / "tape_comparison.csv", dtype={"track_ids": str})
    assert T.track_ids.notna().all() and T.track_ids.nunique() == 4


def test_padding_of_a_perfect_box_is_zero(result):
    q = result[1]["box_padding_px"]
    assert max(abs(q[k]) for k in ("left", "top", "right", "bottom")) < 1.5


def test_on_the_go_agrees_with_the_whole_clip(result):
    C = pd.read_csv(result[0] / "coefficients.csv").dropna(subset=["coefficient", "coefficient_live"])
    assert len(C) > 40 and (C.coefficient_live / C.coefficient - 1).abs().median() < .10


def test_outputs_exist(result):
    for f in "report.html label.html annotated.mp4 coefficients.csv defects.csv per_frame.csv detections.csv tape_comparison.csv".split():
        assert (result[0] / f).stat().st_size > 100, f


def test_simple_formula_matches_the_code():
    """The coefficient written in the report is what the code computes for a level camera and a small box."""
    cam, sc, n = ra.Camera(1400.0, W, HH), ra.Scale(1.152, 1704.0), np.array([0, 1.0, 0])
    y, v0 = 900.0, HH / 2
    m = ra.measure_box((955, y - 4, 965, y + 4), n, cam, sc)
    assert m["area"] == pytest.approx(80 * 1704 * 1.152 / (y - v0) ** 3, rel=.01)
    assert m["distance"] == pytest.approx(1704 / (y - v0), rel=.01)
    assert m["width"] == pytest.approx(10 * 1.152 / (y - v0), rel=.01)


def test_near_horizon_is_refused():
    assert ra.measure_box((900, 545, 950, 590), np.array([0, 1.0, 0]), ra.Camera(1400.0, W, HH), ra.Scale(1.152, 1704.0)) is None


def test_truth_in_centimetres_is_refused(tmp_path):
    (tmp_path / "t.csv").write_text("name,length_m,width_m\nP1,60,45\n")
    with pytest.raises(SystemExit, match="centimetres"):
        ra.read_truth(tmp_path / "t.csv")


def test_template_is_valid():
    assert len(ra.read_truth("truth_template.csv")) == 2


def test_gpx_with_too_few_points_is_refused(tmp_path):
    (tmp_path / "g.gpx").write_text('<gpx><trkpt lat="1" lon="2"><time>2026-11-26T00:00:00Z</time></trkpt></gpx>')
    with pytest.raises(SystemExit, match="at least 4|At least 4"):
        ra.Gpx(tmp_path / "g.gpx", 10)


# ----------------------------------------------------------------------------- merged videos and existing rows (rescore)

def _gpx_text(segments, t0=datetime(2026, 11, 26, 6, 0, tzinfo=timezone.utc)):
    """segments: list of (start offset in s, speed function, seconds). Points once a second, heading north."""
    out = []
    for start, v, secs in segments:
        lat, pts = 13.0, []
        for s in range(secs + 1):
            pts.append(f'<trkpt lat="{lat:.9f}" lon="77.5"><time>{(t0 + timedelta(seconds=start + s)).isoformat()}</time></trkpt>')
            lat += v(s + .5) / 111320
        out.append("<trkseg>" + "".join(pts) + "</trkseg>")
    return "<gpx><trk>" + "".join(out) + "</trk></gpx>"


def test_merged_gpx_segments_are_laid_end_to_end(tmp_path):
    (tmp_path / "g.gpx").write_text(_gpx_text([(0, lambda s: 4.0, 10), (3600, lambda s: 8.0, 10)]))
    g = ra.Gpx(tmp_path / "g.gpx", 20.0)
    assert len(g.segments) == 2 and abs(g.gpx_seconds - 20) < 1e-6
    assert abs(g.speed(5) - 4) < .2 and abs(g.speed(15) - 8) < .2      # the hour between them is not video


def test_cut_finder_spots_the_jump_between_recordings_only():
    rng = np.random.default_rng(3)
    a, b = rng.integers(0, 255, (36, 64)), rng.integers(0, 255, (36, 64))
    f = ra.CutFinder(30.0)
    cuts = [i for i in range(200) if f.push(i, np.clip((a if i < 120 else b) + rng.normal(0, 4, a.shape) + i % 7, 0, 255).astype(np.uint8))]
    assert cuts == [120]


def test_each_recording_is_aligned_to_its_gpx_segment_by_speed(tmp_path):
    v1 = lambda s: 5 + 3 * math.sin(s / 7) + 1.5 * math.sin(s / 2.3)
    v2 = lambda s: 6 + 2.5 * math.sin(s / 5) + math.cos(s / 3.1)
    (tmp_path / "g.gpx").write_text(_gpx_text([(0, v1, 120), (4000, v2, 90)]))
    fps, gap, H, rng = 30.0, 9, 1.2, np.random.default_rng(0)
    g = ra.Gpx(tmp_path / "g.gpx", 150.0)
    # video: 80 s of segment 1 from its 17th second, then 70 s of segment 2 from its 4th second
    truth = [(0, 80, 0, 17.0), (80, 150, 1, 4.0)]
    recs = []
    for i in range(0, int(150 * fps), 3):
        t = (i - gap / 2) / fps
        lo, hi, seg, off = next(p for p in truth if p[0] <= t < p[1]) if t >= 0 else truth[0]
        v = np.interp(off + t - lo, g.segments[seg].t, g.segments[seg].v)
        recs.append(dict(frame=i, hrel=H / (v * gap / fps) * math.exp(rng.normal(0, .05))))
    got = ra.align_gpx(recs, g, [0, int(80 * fps), int(150 * fps)], fps, gap)
    for p, (lo, _, seg, off) in zip(got, truth):
        assert p["gpx_segment"] == seg and abs(p["gpx_offset_s"] - off) <= .5 and p["speed_match"] > .8


def test_an_evidence_image_finds_its_frame():
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.integers(0, 255, (1080, 1920)).astype(np.uint8), (0, 0), 9)
    frames = [np.roll(base, 6 * i, axis=1) for i in range(60)]
    thumbs = np.array([ra.thumb(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)) for f in frames])
    ev = cv2.imdecode(cv2.imencode(".jpg", cv2.cvtColor(frames[37], cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])[1], 1)
    k, e, e2 = ra.find_frame(ev, thumbs, 30.0)
    assert k == 37 and e < ra.FRAME_MATCH_MAX


def test_stored_area_picks_its_box_and_road_clipping_is_kept():
    det = lambda img, conf: [dict(label="pothole", conf=.8, box=[0, 0, 100, 100]), dict(label="pothole", conf=.8, box=[0, 0, 300, 200]),
                             dict(label="Ravelling", conf=.6, box=[0, 0, 50, 40])]
    rows = pd.DataFrame(dict(nlabel=["pothole", "pothole", "ravelling"],
                             defect_area_m2=[300 * 200 / 145 ** 2, .6 * 100 * 100 / 145 ** 2, 2000 / 145 ** 2]), index=[7, 8, 9])
    b = ra.recover_boxes(rows, None, det)
    assert b[7][0] == [0, 0, 300, 200] and b[7][1] == 1.0
    assert b[8][0] == [0, 0, 100, 100] and abs(b[8][1] - .6) < 1e-9
    assert b[9][0] == [0, 0, 50, 40]


class EvidenceDetector(PerfectDetector):
    """The perfect detector for a picture taken out of the video: it finds the frame by its look first."""
    thumbs = None

    def __call__(self, frame, conf):
        self.k = ra.find_frame(frame, self.thumbs, FPS)[0] // 3
        return super().__call__(frame, conf)


def test_rescore_gives_existing_rows_the_tracked_area(result, tmp_path):
    import shutil
    out = tmp_path / "run"
    shutil.copytree(result[0], out)
    vid, frames = cv2.VideoCapture(str(result[0].parent / "v.mp4")), []
    while True:
        ok, f = vid.read()
        if not ok:
            break
        frames.append(f)
    (tmp_path / "img").mkdir()
    rows = []
    for n, (cx, cz, ln, wd) in DEFECTS.items():
        i = next(i for i in range(0, N, 3) if 9 < cz - SPEED * i / FPS < 11)             # first stored when 10 m away, as production does
        p, _ = project(i, outline(n))
        px = np.ptp(p[:, 0]) * np.ptp(p[:, 1])
        cv2.imwrite(str(tmp_path / "img" / f"pothole_frame{i}.jpg"), frames[i])
        rows.append(dict(id=100 + len(rows), job_id="J1", label="Pothole", frame_url=f"defect_frame_images/pothole_frame{i}.jpg",
                         defect_area_m2=round(px / 145 ** 2, 4), estimated_cost_inr=round(px / 145 ** 2 * 1200, 2), true=ln * wd, name=n))
    first = rows[0]
    rows.append(dict(first, id=200, true=first["true"]))                                  # the same pothole stored twice
    rows.append(dict(first, id=201, label="Garbage Pile", estimated_cost_inr=50))          # not flat on the road
    cv2.imwrite(str(tmp_path / "img" / "noise.jpg"), np.random.default_rng(0).integers(0, 255, (HH, W, 3)).astype(np.uint8))
    rows.append(dict(first, id=202, frame_url="defect_frame_images/noise.jpg"))           # a picture from another video
    pd.DataFrame(rows).drop(columns=["true", "name"]).to_csv(tmp_path / "rows.csv", index=False)

    EvidenceDetector.thumbs = np.load(out / "thumbs.npy")
    ra.Detector, keep = EvidenceDetector, ra.Detector
    try:
        path = ra.main(["rescore", "--rows", str(tmp_path / "rows.csv"), "--out", str(out), "--images", str(tmp_path / "img"), "--model", __file__])
    finally:
        ra.Detector = keep
    R = pd.read_csv(path, dtype={"same_defect_as_id": str})
    src = pd.read_csv(tmp_path / "rows.csv")
    assert list(R.id) == list(src.id) and set(src.columns) <= set(R.columns)                # no row added, dropped or reordered
    assert (R.defect_area_m2 == src.defect_area_m2).all()                                   # the old value stays as it was
    pot = R[R.id < 200].assign(true=[r["true"] for r in rows[:4]])
    assert (pot.area_changed == "yes").all() and pot.track_id.nunique() == 4
    assert ((pot.defect_area_m2_new / pot.true - 1).abs() < .2).all(), pot[["defect_area_m2", "defect_area_m2_new", "true"]]
    assert ((pot.defect_area_m2 / pot.true - 1).abs() > .3).any()                           # the old area was off
    assert (pot.estimated_cost_inr_new / pot.defect_area_m2_new / 1200 - 1).abs().max() < .02     # cost follows the area at the stored rate
    dup, garbage, other = (R[R.id == k].iloc[0] for k in (200, 201, 202))
    assert dup.same_defect_as_id == "100" and dup.track_id == pot.track_id.iloc[0]
    assert garbage.area_changed == "no" and garbage.estimated_cost_inr_new == 50 and "above the road" in garbage.note
    assert other.area_changed == "no" and "not in this video" in other.note
    assert (out / R.defect_image.dropna().iloc[0]).is_file()


def test_rescore_reads_the_workers_own_csv_by_frame_number(result, tmp_path):
    """The worker's per-video CSV has object_id, frame and defect_name, and no evidence image."""
    import shutil
    out = tmp_path / "run"
    shutil.copytree(result[0], out)
    rows = []
    for n, (cx, cz, ln, wd) in DEFECTS.items():
        i = next(i for i in range(1, N, 3) if 9 < cz - SPEED * i / FPS < 11)              # a frame the run did not analyse
        p, _ = project(i, outline(n))
        rows.append(dict(object_id=len(rows) + 1, frame=i, timestamp="00:01", defect_name="Pothole",
                         defect_area_m2=round(np.ptp(p[:, 0]) * np.ptp(p[:, 1]) / 145 ** 2, 4), true=ln * wd))
    pd.DataFrame(rows).drop(columns="true").to_csv(tmp_path / "video_defect.csv", index=False)
    EvidenceDetector.thumbs = np.load(out / "thumbs.npy")
    ra.Detector, keep = EvidenceDetector, ra.Detector
    try:
        path = ra.main(["rescore", "--rows", str(tmp_path / "video_defect.csv"), "--out", str(out), "--video", str(result[0].parent / "v.mp4"), "--model", __file__])
    finally:
        ra.Detector = keep
    R = pd.read_csv(path)
    assert path.name == "video_defect_new_area.csv" and list(R.columns[:6]) == ["object_id", "frame", "timestamp", "defect_name", "defect_area_m2", "defect_area_m2_new"]
    assert (R.area_changed == "yes").all() and R.track_id.nunique() == 4
    assert ((R.defect_area_m2_new / [r["true"] for r in rows] - 1).abs() < .2).all()
