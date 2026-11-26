#!/usr/bin/env python3
"""Distance-corrected road defect area from a phone video and its GPX.

    road_area.py run      --video V --gpx G --height 1.152 --model M --out OUT [--truth truth.csv]
    road_area.py evaluate --out OUT --truth truth.csv [--outlines outlines.json]

`run` reads the video once. In that single pass it detects the defects and works out, in each
analysed frame, the coefficient that turns a box's pixel area into square metres. `evaluate`
re-scores a finished run against tape measurements without running the model again.

Geometry, in one place:
  * v0 is the horizon row. A road point at row y is f*H / (y - v0) metres ahead.
  * Across the road:  metres per pixel = H / (y - v0)
  * Along the road:   metres per pixel = f*H / (y - v0)^2
  * Area:             m2 per pixel     = f*H * H / (y - v0)^3
v0 changes in every frame (the phone tilts), so the coefficient does too. f*H and H are fixed for
one mounting of one phone. The code uses the full road plane, so phone roll is handled as well.
"""
import argparse
import base64
import html
import io
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.optimize import linear_sum_assignment

WORK_W = 1920              # every pixel number in the outputs is at this frame width
PROD_PPM = 145.0           # production today: area = pixels / 145^2
MILESTONE_A_MAX = 0.15     # agreed: spread of one defect's area while approaching it
MILESTONE_B_MAX = 0.25     # agreed: error against the tape
A_RANGE = (4.0, 12.0)      # metres; the distances Milestone A is judged over
FINAL_RANGE = (3.0, 8.0)   # metres; the views that make up a defect's stored area
MIN_VB = 60                # px below the horizon; closer to it the geometry is unusable
UPRIGHT = {"sign board", "garbage pile"}      # not flat on the road


# ----------------------------------------------------------------------------- geometry

@dataclass
class Camera:
    """Assumed pinhole camera. `f` is a working focal length; `Scale` corrects for its error."""
    f: float
    w: int
    h: int

    @property
    def K(self):
        return np.array([[self.f, 0, self.w / 2], [0, self.f, self.h / 2], [0, 0, 1.0]])


@dataclass
class Scale:
    """Metres per 'camera height' across the road (H) and along it (fH / working focal)."""
    height: float
    fh: float

    def along(self, cam):
        return self.fh / cam.f


def horizon(n, cam):
    """Horizon row at the image centre column, and phone roll in degrees."""
    return cam.h / 2 - cam.f * n[2] / n[1], math.degrees(math.atan2(n[0], n[1]))


def to_road(pts, n, cam):
    """Image points -> (across, along) on the road, in units of the camera height."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    ray = np.c_[(pts[:, 0] - cam.w / 2) / cam.f, (pts[:, 1] - cam.h / 2) / cam.f, np.ones(len(pts))]
    den = ray @ n
    X = ray / np.where(den > 1e-4, den, np.nan)[:, None]
    e1 = np.array([1.0, 0, 0]) - n[0] * n
    e1 /= np.linalg.norm(e1)
    return np.c_[X @ e1, X @ np.cross(e1, n)]


def polygon_area(g):
    return 0.5 * abs(np.dot(g[:, 0], np.roll(g[:, 1], 1)) - np.dot(g[:, 1], np.roll(g[:, 0], 1)))


def measure(poly, n, cam, scale):
    """Size of an image polygon on the road, or None when it is too near the horizon."""
    poly = np.asarray(poly, float).reshape(-1, 2)
    if n is None or poly[:, 1].min() - horizon(n, cam)[0] < MIN_VB:
        return None
    g = to_road(poly, n, cam)
    if not np.isfinite(g).all():
        return None
    g = g * [scale.height, scale.along(cam)]
    return dict(area=polygon_area(g), length=float(np.ptp(g[:, 1])), distance=float(g[:, 1].mean()),
                width=float(np.ptp(g[:, 0])))


def box_poly(b):
    return np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], float)


def measure_box(b, n, cam, scale):
    """As `measure`, for an upright box. Width is read at the box's middle row: an upright box is
    wider on the road at its far edge, so its full extent would overstate the width."""
    m = measure(box_poly(b), n, cam, scale)
    if m is None:
        return None
    mid = (b[1] + b[3]) / 2
    g = to_road([[b[0], mid], [b[2], mid], [(b[0] + b[2]) / 2, b[1]], [(b[0] + b[2]) / 2, b[3]],
                 [(b[0] + b[2]) / 2, mid]], n, cam)
    along = scale.along(cam)
    m.update(width=float(abs(g[1, 0] - g[0, 0]) * scale.height), length=float(abs(g[3, 1] - g[2, 1]) * along),
             distance=float(g[4, 1] * along))
    return m


# ----------------------------------------------------------------------------- GPX

class Gpx:
    """Speed against video time."""

    def __init__(self, path, duration, offset=None):
        pts, text = [], Path(path).read_text()
        for m in re.finditer(r"<trkpt\b([^>]*)>(.*?)</trkpt>", text, re.S):
            lat, lon = (re.search(rf'\b{k}\s*=\s*["\']([-\d.]+)', m[1]) for k in ("lat", "lon"))
            t = re.search(r"<time>\s*(.*?)\s*</time>", m[2], re.S)
            if lat and lon and t:
                pts.append((datetime.fromisoformat(t[1].replace("Z", "+00:00")).timestamp(), float(lat[1]), float(lon[1])))
        pts = sorted({p[0]: p for p in pts}.values())
        if len(pts) < 4:
            raise SystemExit(f"GPX {path}: only {len(pts)} timed points. At least 4 are needed.")
        P = np.array(pts)
        self.n, self.duration = len(P), duration
        T = P[:, 0] - P[0, 0]
        self.gpx_seconds = float(T[-1])
        x = (P[:, 2] - P[0, 2]) * 111320 * math.cos(math.radians(P[0, 1]))
        y = (P[:, 1] - P[0, 1]) * 111320
        self.metres = float(np.hypot(np.diff(x), np.diff(y)).sum())
        v = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(T), 0.2)
        self.v = gaussian_filter1d(median_filter(v, min(5, len(v) | 1), mode="nearest"), 1.0, mode="nearest")
        self.t = (T[1:] + T[:-1]) / 2
        # The app starts and stops both together, so by default the GPX is stretched over the video.
        self.map = (lambda s: s * T[-1] / duration) if offset is None else (lambda s: s + offset)
        self.start_lat_lon = (T, P[:, 1], P[:, 2])

    def speed(self, sec):
        return float(np.interp(self.map(sec), self.t, self.v))

    def position(self, sec):
        T, la, lo = self.start_lat_lon
        s = self.map(sec)
        return float(np.interp(s, T, la)), float(np.interp(s, T, lo))


# ----------------------------------------------------------------------------- camera motion

def road_plane(R, t, a1, a2, cam, rng):
    """Road normal and camera height (in units of the distance moved) from tracked road points."""
    if (-R.T @ t)[2] < 0.6:                      # not moving forwards: depth would be unreliable
        return None
    road = (a2[:, 1] > cam.h * .55) & (abs(a2[:, 0] - cam.w / 2) < cam.w * .38) & (np.linalg.norm(a1 - a2, axis=1) > 3)
    if road.sum() < 40:
        return None
    Ki, one = np.linalg.inv(cam.K), np.ones(road.sum())
    X = cv2.triangulatePoints(np.eye(3, 4), np.c_[R, t], (Ki @ np.c_[a1[road], one].T)[:2], (Ki @ np.c_[a2[road], one].T)[:2])
    X = (X[:3] / X[3]).T @ R.T + t
    X = X[(X[:, 2] > 0) & (X[:, 1] > 0) & (X[:, 2] < 60)]
    if len(X) < 30:
        return None
    best = (0, None, 0)
    for _ in range(150):
        p = X[rng.choice(len(X), 3, False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        L = np.linalg.norm(n)
        if L < 1e-9:
            continue
        n = n / L * (np.sign(n[1]) or 1)
        if n[1] < .85:
            continue
        h = n @ p[0]
        c = (abs(X @ n - h) < .04 * h).sum()     # relative: depth noise grows with distance
        if c > best[0]:
            best = (c, n, h)
    if best[1] is None or best[0] < 25:
        return None
    Z = X[abs(X @ best[1] - best[2]) < .04 * best[2]]
    w = 1 / np.linalg.norm(Z, axis=1) ** 2
    mu = (Z * w[:, None]).sum(0) / w.sum()
    n = np.linalg.svd((Z - mu) * np.sqrt(w)[:, None])[2][2]
    n *= np.sign(n[1])
    return n, float(n @ mu), len(Z)


class Motion:
    """Tracks points through the video and yields the road plane every `step` frames."""

    def __init__(self, cam, step, gap):
        self.cam, self.step, self.gap = cam, step, gap
        self.clahe = cv2.createCLAHE(3, (8, 8))
        self.live, self.prev, self.rng = [], None, np.random.default_rng(0)

    def _features(self, g):
        h, w = g.shape
        out = []
        for y0, y1, x0, x1, num, d in ((0, .45, 0, 1, 350, 14), (.45, .6, .1, .9, 350, 7), (.6, .95, .05, .95, 600, 9)):
            m = np.zeros_like(g)
            m[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = 255
            p = cv2.goodFeaturesToTrack(self.clahe.apply(g), num, .004, d, mask=m)
            if p is not None:
                out.append(p)
        return np.vstack(out).astype(np.float32) if out else np.empty((0, 1, 2), np.float32)

    def _pose(self, L):
        a1, a2 = L["s"][:, 0] * 2, L["c"][:, 0] * 2
        if len(a1) < 60 or np.median(np.linalg.norm(a2 - a1, axis=1)) < 1.0:
            return None
        E, m = cv2.findEssentialMat(a1, a2, self.cam.K, cv2.USAC_MAGSAC, .999, 1.0)
        if E is None or E.shape != (3, 3):
            return None
        _, R, t, mp = cv2.recoverPose(E, a1, a2, self.cam.K, mask=m.copy())
        k = mp.ravel() > 0
        return R, t.ravel(), a1[k], a2[k]

    def push(self, i, frame):
        """Feed every frame. Returns a record on analysed frames, otherwise None."""
        g = cv2.cvtColor(cv2.resize(frame, (self.cam.w // 2, self.cam.h // 2), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        g = self.clahe.apply(g)
        for L in self.live if self.prev is not None else []:
            if len(L["c"]) == 0:
                continue
            kw = dict(winSize=(17, 17), maxLevel=3)
            q, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, L["c"], None, **kw)
            b, _, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, q, None, **kw)
            k = (st.ravel() == 1) & (np.linalg.norm(L["c"] - b, axis=2).ravel() < .5)
            L["c"], L["s"] = q[k], L["s"][k]
        self.prev = g
        if i % self.step:
            return None
        rec = dict(frame=i, n=None, hrel=np.nan, R=np.eye(3), homography=None, inliers=0)
        near = [L for L in self.live if L["i"] == i - self.step]
        done = [L for L in self.live if L["i"] == i - self.gap]
        if near:
            z = self._pose(near[0])
            if z:
                rec["R"] = z[0]
            s, c = near[0]["s"][:, 0] * 2, near[0]["c"][:, 0] * 2
            k = s[:, 1] > self.cam.h * .5
            if k.sum() >= 12:
                rec["homography"] = cv2.findHomography(s[k], c[k], cv2.RANSAC, 3.0)[0]
        if done:
            z = self._pose(done[0])
            if z:
                p = road_plane(*z, self.cam, self.rng)
                if p:
                    rec.update(n=p[0], hrel=p[1], inliers=p[2])
        self.live = [L for L in self.live if L["i"] > i - self.gap]
        s = self._features(g)
        self.live.append(dict(i=i, s=s, c=s.copy()))
        return rec


def smooth_normal(recs, k, win, causal):
    """Average the road normals near record k in one common orientation, then rotate back."""
    C, Q = np.eye(3), {}
    lo = max(0, k - win)
    hi = k + 1 if causal else min(len(recs), k + win + 1)
    for j in range(lo, hi):
        if j > lo:
            C = C @ recs[j]["R"].T
        Q[j] = C
    s = [j for j in range(lo, hi) if recs[j]["n"] is not None]
    if len(s) < 4:
        return None
    N = np.array([Q[j] @ recs[j]["n"] for j in s])
    w = np.array([recs[j]["inliers"] for j in s]) * np.exp(-.5 * ((np.array(s) - k) / (win / 2)) ** 2)
    m = np.median(N, 0)
    for _ in range(10):
        m /= np.linalg.norm(m)
        ww = w / np.maximum(np.linalg.norm(N - m, axis=1), .01)
        m = (N * ww[:, None]).sum(0) / ww.sum()
    return Q[k].T @ (m / np.linalg.norm(m))


def fill_normals(recs, win):
    """Final (both directions) normals. A short gap, such as the first frames, borrows a wider window."""
    out = []
    for k in range(len(recs)):
        n = None
        for m in (1, 2, 4):
            n = smooth_normal(recs, k, win * m, causal=False)
            if n is not None:
                break
        out.append(n)
    return out


def estimate_fh(recs, gpx, cam, fps, gap):
    """f*H per record: relative camera height x GPX metres moved, times the working focal length."""
    out = []
    for r in recs:
        v = gpx.speed((r["frame"] - gap / 2) / fps)
        out.append(cam.f * r["hrel"] * v * gap / fps if r["n"] is not None and v > 1.0 else np.nan)
    return np.array(out)


# ----------------------------------------------------------------------------- road patches (model-free test)

class PatchTracker:
    """Follows texture points painted by nothing but the road itself: a Milestone A test with no model."""

    def __init__(self, cam, step, fps):
        self.cam, self.step, self.every, self.life = cam, step, max(step, int(round(fps / step)) * step), int(3 * fps)
        self.clahe, self.prev, self.live, self.done = cv2.createCLAHE(3, (8, 8)), None, [], []

    def push(self, i, frame):
        g = self.clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        h, w = g.shape
        for L in self.live:
            kw = dict(winSize=(25, 25), maxLevel=4)
            q, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, L["c"], None, **kw)
            b, _, _ = cv2.calcOpticalFlowPyrLK(g, self.prev, q, None, **kw)
            a = L["a"]
            a &= (st.ravel() == 1) & (np.linalg.norm(L["c"] - b, axis=2).ravel() < .8) & (q[:, 0, 1] < h - 20) & (q[:, 0, 0] > 10) & (q[:, 0, 0] < w - 10)
            if a.sum() >= 8:
                ii = np.where(a)[0]
                ms = cv2.findHomography(L["p0"][ii], q[ii, 0], cv2.RANSAC, 4.0)[1]
                if ms is not None:
                    a[ii[ms.ravel() == 0]] = False
            L["c"] = q
            if i % self.step == 0:
                L["t"][i] = np.where(a[:, None], q[:, 0], np.nan)
        out = [L for L in self.live if L["a"].sum() < 8 or i - L["s"] >= self.life]
        self.done += out
        self.live = [L for L in self.live if not any(L is o for o in out)]
        if i % self.every == 0:
            m = np.zeros_like(g)
            m[int(.40 * h):int(.70 * h), int(.25 * w):int(.75 * w)] = 255
            p = cv2.goodFeaturesToTrack(g, 500, .01, 8, mask=m)
            if p is not None:
                self.live.append(dict(s=i, p0=p[:, 0].copy(), c=p, a=np.ones(len(p), bool), t={i: p[:, 0].copy()}))
        self.prev = g

    def rows(self, normals, cam, scale):
        rng, rows, pid = np.random.default_rng(1), [], 0
        for L in self.done + self.live:
            n0, fs = normals.get(L["s"]), sorted(L["t"])
            if n0 is None or len(fs) < 8:
                continue
            G = to_road(L["p0"], n0, cam) * [scale.height, scale.along(cam)]
            ok = np.where(np.isfinite(L["t"][fs[6]][:, 0]) & (G[:, 1] > 5) & (G[:, 1] < 15) & (abs(G[:, 0]) < 4))[0]
            k = 0
            for _ in range(300 if len(ok) >= 6 else 0):
                tri = rng.choice(ok, 3, False)
                e = np.linalg.norm(G[tri] - np.roll(G[tri], 1, 0), axis=1)
                A0 = polygon_area(G[tri])
                if not .1 < A0 < 3 or 2 * A0 / e.max() < .35:
                    continue
                pid, k = pid + 1, k + 1
                for f in fs:
                    pts = L["t"][f][tri]
                    if np.isnan(pts).any():
                        break
                    m = measure(pts, normals.get(f), cam, scale)
                    if m:
                        rows.append(dict(track_id=pid, frame=f, distance_m=m["distance"], pixels=polygon_area(pts),
                                         area_old=polygon_area(pts) / PROD_PPM ** 2, area_new=m["area"]))
                if k >= 15:
                    break
        return pd.DataFrame(rows, columns="track_id frame distance_m pixels area_old area_new".split())


# ----------------------------------------------------------------------------- detection and tracking

class Detector:
    def __init__(self, path, device, imgsz):
        from ultralytics import YOLO
        import torch
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.m, self.imgsz = YOLO(str(path)), imgsz

    def __call__(self, frame, conf):
        r = self.m(frame, verbose=False, conf=conf, device=self.device, imgsz=self.imgsz)[0].boxes
        return [dict(label=self.m.names[int(c)], conf=float(p), box=[float(v) for v in b])
                for b, p, c in zip(r.xyxy.cpu().numpy(), r.conf.cpu().numpy(), r.cls.cpu().numpy())]


def iou(a, b):
    w, h = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


class Tracker:
    """Keeps one ID per defect: last boxes are moved with the road's own motion, then matched."""

    def __init__(self, start_conf, patience):
        self.tracks, self.next, self.start, self.patience = {}, 1, start_conf, patience

    def step(self, dets, H):
        for t in self.tracks.values():
            if H is not None:
                q = cv2.perspectiveTransform(box_poly(t["box"])[None], H)[0]
                b = [q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()]
                if np.isfinite(b).all() and (b[2] - b[0]) < 4 * (t["box"][2] - t["box"][0]) + 50:
                    t["box"] = [float(v) for v in b]
            t["age"] += 1
        ids = list(self.tracks)
        for d in dets:
            d["track"] = None
        if ids and dets:
            C = np.array([[iou(self.tracks[i]["box"], d["box"]) if self.tracks[i]["label"] == d["label"] else 0 for d in dets] for i in ids])
            for a, b in zip(*linear_sum_assignment(-C)):
                if C[a, b] > .25:
                    dets[b]["track"] = ids[a]
                    self.tracks[ids[a]].update(box=dets[b]["box"], age=0)
        for d in dets:
            if d["track"] is None and d["conf"] >= self.start:
                d["track"] = self.next
                self.tracks[self.next] = dict(box=d["box"], label=d["label"], age=0)
                self.next += 1
        self.tracks = {k: t for k, t in self.tracks.items() if t["age"] <= self.patience}
        return [d for d in dets if d["track"] is not None]


# ----------------------------------------------------------------------------- inputs

def fetch(src, folder, name, profile):
    """Local path, http(s) URL or s3:// URI -> local path."""
    if re.match(r"https?://", src):
        dst = folder / name
        print("downloading", src.split("?")[0])
        with urllib.request.urlopen(src) as r, open(dst, "wb") as f:
            shutil.copyfileobj(r, f)
        return dst
    if src.startswith("s3://"):
        dst = folder / name
        subprocess.run(["aws", "s3", "cp", src, str(dst), "--only-show-errors"] + (["--profile", profile] if profile else []), check=True)
        return dst
    p = Path(src).expanduser()
    if not p.is_file():
        raise SystemExit(f"not found: {p}")
    return p


class Video:
    """Sequential reader. Never seeks: phone video has a variable frame rate and seeking by number lands elsewhere."""

    def __init__(self, path):
        self.c = cv2.VideoCapture(str(path))
        if not self.c.isOpened():
            raise SystemExit(f"cannot open video: {path}")
        self.c.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        ok, f = self.c.read()
        if not ok:
            raise SystemExit(f"no frames in video: {path}")
        self.fps = float(self.c.get(cv2.CAP_PROP_FPS)) or 30.0
        if not 5 < self.fps < 250:
            self.fps = 30.0
        self.src = (f.shape[1], f.shape[0])
        self.w, self.h = WORK_W, int(round(f.shape[0] * WORK_W / f.shape[1] / 2)) * 2
        self.first, self.count = f, int(self.c.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        f = self.first
        while f is not None:
            yield f if f.shape[1] == self.w and f.shape[0] == self.h else cv2.resize(f, (self.w, self.h), interpolation=cv2.INTER_AREA)
            ok, f = self.c.read()
            f = f if ok else None


# ----------------------------------------------------------------------------- run

def run(a):
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        video = fetch(a.video, Path(tmp), "video.mp4", a.aws_profile)
        gpxfile = fetch(a.gpx, Path(tmp), "track.gpx", a.aws_profile)
        shutil.copy(gpxfile, out / "track.gpx")
        _run(a, out, video, out / "track.gpx")
    evaluate_dir(out, a.truth, a.outlines, (a.trim_x, a.trim_y))


def _run(a, out, video, gpxfile):
    vid = Video(video)
    cam = Camera(a.focal or 1400.0, vid.w, vid.h)
    gpx = Gpx(gpxfile, vid.count / vid.fps, a.gpx_offset)
    step = max(1, int(round(vid.fps / 10)))                       # about 10 analysed frames a second
    v = float(np.median([gpx.speed(s) for s in np.arange(0, vid.count / vid.fps, .5)]))
    gap = int(np.clip(round(1.5 / max(v, .5) * vid.fps / step), 2, 10)) * step   # about 1.5 m of travel
    win = max(6, int(round(1.2 * vid.fps / step)))
    print(f"video {vid.src[0]}x{vid.src[1]} {vid.fps:.2f} fps {vid.count} frames | gpx {gpx.n} points, {gpx.gpx_seconds:.1f} s, "
          f"{gpx.metres:.0f} m, {v * 3.6:.1f} km/h | analysing every {step} frames, baseline {gap} frames")
    warnings = []
    if vid.h > vid.w:
        warnings.append("Video is portrait. The method expects landscape, as the app records it.")
    if abs(gpx.gpx_seconds - vid.count / vid.fps) > 2 and a.gpx_offset is None:
        warnings.append(f"GPX lasts {gpx.gpx_seconds:.1f} s but the video {vid.count / vid.fps:.1f} s. They were stretched to match; pass --gpx-offset if they did not start together.")
    if v < 1.5:
        warnings.append("Very slow: GPS speed is unreliable below about 1.5 m/s.")
    if vid.count / vid.fps < 20:
        warnings.append("Short clip: on past recordings a 10 s clip put about 2% (up to 6.5%) error on the GPX scale.")

    det, motion = Detector(a.model, a.device, a.imgsz), Motion(cam, step, gap)
    patches, tracker = PatchTracker(cam, step, vid.fps), Tracker(a.conf, int(round(vid.fps / step)))
    recs, boxes, live, fh_seen, n_frames = [], [], [], [], 0
    for i, frame in enumerate(vid):
        n_frames += 1
        patches.push(i, frame)
        r = motion.push(i, frame)
        if r is None:
            continue
        recs.append(r)
        k = len(recs) - 1
        # ---- the coefficient "on the go": only this frame and earlier ones are used
        n_live = smooth_normal(recs, k, win, causal=True)
        x = estimate_fh([r], gpx, cam, vid.fps, gap)[0]
        if np.isfinite(x):
            fh_seen.append(x)
        fh_live = a.fh or (float(np.median(fh_seen)) if len(fh_seen) >= 5 else np.nan)
        ds = tracker.step(det(frame, a.conf_low), r["homography"])
        for d in ds:
            m = measure_box(d["box"], n_live, cam, Scale(a.height, fh_live)) if n_live is not None and np.isfinite(fh_live) else None
            boxes.append(dict(frame=i, track_id=d["track"], label=d["label"], conf=round(d["conf"], 4),
                              **dict(zip(("x1", "y1", "x2", "y2"), np.round(d["box"], 2))), live_area=m["area"] if m else np.nan))
        live.append(dict(frame=i, live_horizon=horizon(n_live, cam)[0] if n_live is not None else np.nan, live_fh=fh_live))
        cv2.imwrite(str(out / "frames" / f"{i:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if k % 25 == 0:
            print(f"  frame {i}/{vid.count}  defects tracked {tracker.next - 1}", flush=True)
    if n_frames != vid.count:
        warnings.append(f"Video header says {vid.count} frames; {n_frames} were readable.")

    # ---- second look, with the whole clip available
    fh_all = estimate_fh(recs, gpx, cam, vid.fps, gap)
    good = fh_all[np.isfinite(fh_all)]
    if len(good) < 5 and not a.fh:
        raise SystemExit("Could not recover the camera motion in enough frames (blurred, static or turning video?). Pass --fh to continue.")
    fh = a.fh or (a.focal * a.height if a.focal else float(np.median(good)))
    halves = [float(np.median(x)) for x in np.array_split(good, 2)] if len(good) >= 10 else []
    if halves and abs(halves[0] - halves[1]) / fh > .10:
        warnings.append(f"f*H differs between the first and second half of the clip ({halves[0]:.0f} vs {halves[1]:.0f}): scale is less certain.")
    normals = fill_normals(recs, win)
    N = [n for n in normals if n is not None]
    n_fixed = np.median(N, 0) / np.linalg.norm(np.median(N, 0))
    meta = dict(video=str(a.video).split("?")[0], gpx=str(a.gpx).split("?")[0], fps=vid.fps, frames=n_frames, width=vid.w, height=vid.h,
                source_size=vid.src, step=step, gap=gap, working_focal=cam.f, camera_height_m=a.height, fh=fh,
                fh_source="--fh" if a.fh else "--focal x height" if a.focal else "GPX", fh_gpx=float(np.median(good)) if len(good) else None,
                fh_halves=halves, implied_focal=fh / a.height, gpx_points=gpx.n, gpx_seconds=gpx.gpx_seconds, gpx_metres=gpx.metres,
                speed_kmh=v * 3.6, warnings=warnings, n_fixed=n_fixed.tolist(), planes_found=int(len(N)), planes_total=len(recs),
                model=str(a.model), conf=a.conf, imgsz=a.imgsz)
    P = pd.DataFrame([dict(frame=r["frame"], time_s=r["frame"] / vid.fps, nx=n[0] if n is not None else np.nan, ny=n[1] if n is not None else np.nan,
                           nz=n[2] if n is not None else np.nan, horizon_row=horizon(n, cam)[0] if n is not None else np.nan,
                           roll_deg=horizon(n, cam)[1] if n is not None else np.nan, speed_kmh=gpx.speed(r["frame"] / vid.fps) * 3.6,
                           fh_this_frame=x, road_points=r["inliers"]) for r, n, x in zip(recs, normals, fh_all)]).merge(pd.DataFrame(live), on="frame")
    P.to_csv(out / "per_frame.csv", index=False)
    pd.DataFrame(boxes, columns="frame track_id label conf x1 y1 x2 y2 live_area".split()).to_csv(out / "detections.csv", index=False)
    patches.rows({r["frame"]: n for r, n in zip(recs, normals)}, cam, Scale(a.height, fh)).to_csv(out / "patches.csv", index=False)
    (out / "run.json").write_text(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------- scoring

def load(out):
    meta = json.loads((out / "run.json").read_text())
    cam = Camera(meta["working_focal"], meta["width"], meta["height"])
    P = pd.read_csv(out / "per_frame.csv")
    normals = {int(r.frame): np.array([r.nx, r.ny, r.nz]) for r in P.itertuples() if np.isfinite(r.ny)}
    return meta, cam, Scale(meta["camera_height_m"], meta["fh"]), P, normals


def coefficient_rows(out, trim=(0.0, 0.0)):
    """One row per defect per analysed frame, with the area under each way of choosing the coefficient.
    `trim` takes that many pixels off each side / off the top and bottom, for models that draw loose boxes."""
    meta, cam, scale, P, normals = load(out)
    d = pd.read_csv(out / "detections.csv")
    nf, rows = np.array(meta["n_fixed"]), []
    for r in d.itertuples():
        n, b0 = normals.get(r.frame), (r.x1, r.y1, r.x2, r.y2)
        tx, ty = min(trim[0], .4 * (r.x2 - r.x1)), min(trim[1], .4 * (r.y2 - r.y1))
        b = (r.x1 + tx, r.y1 + ty, r.x2 - tx, r.y2 - ty)
        px = (r.x2 - r.x1) * (r.y2 - r.y1)
        m = measure_box(b, n, cam, scale) if n is not None else None
        m0 = measure_box(b0, n, cam, scale) if n is not None else None
        f = measure_box(b, nf, cam, scale)
        area = m["area"] if m else np.nan
        live_area = r.live_area * (m["area"] / m0["area"] if m and m0 else 1.0)
        rows.append(dict(track_id=r.track_id, label=r.label, frame=r.frame, time_s=r.frame / meta["fps"], conf=r.conf, x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2,
                         distance_m=m["distance"] if m else np.nan, box_row=(r.y1 + r.y2) / 2, horizon_row=horizon(n, cam)[0] if n is not None else np.nan,
                         pixels=px, area_old=px / PROD_PPM ** 2, coefficient=area / (px / PROD_PPM ** 2), area_new=area,
                         width_m=m["width"] if m else np.nan, length_m=m["length"] if m else np.nan,
                         coefficient_live=live_area / (px / PROD_PPM ** 2), area_live=live_area, area_fixed_horizon=f["area"] if f else np.nan,
                         cut_by_edge=bool(r.x1 < 8 or r.x2 > cam.w - 8 or r.y2 > cam.h - 8), upright=r.label.lower() in UPRIGHT))
    C = pd.DataFrame(rows)
    if len(C):
        C = C.sort_values(["track_id", "frame"])
    return C


def spread_table(C, cols, zr=A_RANGE, min_views=6):
    """Per-track spread (standard deviation / mean) and biggest / smallest, for tracks seen from clearly different distances."""
    if not len(C):
        return None, C
    u = C[C.distance_m.between(*zr)].dropna(subset=list(cols))
    u = u.groupby("track_id").filter(lambda x: len(x) >= min_views and x.distance_m.max() / x.distance_m.min() > 1.35)
    if not len(u):
        return None, u
    g = u.groupby("track_id")
    T = pd.DataFrame({f"{c}:spread": g[c].std() / g[c].mean() for c in cols} | {f"{c}:ratio": g[c].max() / g[c].min() for c in cols})
    return T, u


def final_areas(C):
    """Stored value per defect: the median of its views between 3 and 8 m."""
    rows = []
    for t, x in C.groupby("track_id") if len(C) else []:
        ok = x[~x.cut_by_edge & x.area_new.notna()]
        v = ok[ok.distance_m.between(*FINAL_RANGE)]
        first = x.iloc[0]
        o = dict(track_id=t, label=first.label, first_frame=int(first.frame), first_time_s=round(first.time_s, 2), views=len(x), views_used=len(v),
                 area_old=first.area_old, area_new=np.nan, width_m=np.nan, length_m=np.nan, distance_m=np.nan, spread=np.nan, confidence="none", note="")
        if first.upright:
            o["note"] = "stands above the road: flat-road geometry does not apply"
        elif len(v):
            o.update(area_new=v.area_new.median(), width_m=v.width_m.median(), length_m=v.length_m.median(), distance_m=v.distance_m.median(),
                     spread=1.4826 * (v.area_new - v.area_new.median()).abs().median() / v.area_new.median() if len(v) > 1 else np.nan)
            o["confidence"] = "high" if len(v) >= 4 and o["spread"] < .25 else "medium" if len(v) >= 2 and o["spread"] < .4 else "low"
        elif len(ok):
            n = ok.loc[ok.distance_m.idxmin()]
            o.update(area_new=n.area_new, width_m=n.width_m, length_m=n.length_m, distance_m=n.distance_m, confidence="low", note="no view between 3 and 8 m")
        else:
            o["note"] = "no usable view (cut by the frame edge or too far)"
        rows.append(o)
    F = pd.DataFrame(rows)
    for a, b in same_place(C):
        k = F.track_id == b
        F.loc[k, "note"] = (F.loc[k, "note"] + f"; same place as #{a}").str.lstrip("; ")
    return F


def same_place(C):
    """Pairs of tracks that the model drew over the same spot, usually under two class names."""
    if not len(C):
        return []
    frames = {t: {r.frame: (r.x1, r.y1, r.x2, r.y2) for r in x.itertuples()} for t, x in C.groupby("track_id")}
    out, ids = [], sorted(frames)
    for j, b in enumerate(ids):
        for a in ids[:j]:
            common = frames[a].keys() & frames[b].keys()
            hit = sum(iou(frames[a][f], frames[b][f]) > .6 for f in common)
            if hit >= 2 and hit >= .5 * min(len(frames[a]), len(frames[b])):
                out.append((a, b))
                break
    return out


def outline_tracks(q_list, C):
    """Tracks whose box covers a hand outline in the same frame."""
    ids = set()
    for q in q_list if len(C) else []:
        p = np.array(q["points"], float)
        bb = (*p.min(0), *p.max(0))
        x = C[C.frame == int(q["frame"])]
        ids |= {int(r.track_id) for r in x.itertuples() if iou(bb, (r.x1, r.y1, r.x2, r.y2)) > .3}
    return sorted(ids)


def box_padding(outlines, C):
    """How many pixels the model's box sticks out past the hand outline on each side (median)."""
    rows = []
    for q in outlines if len(C) else []:
        p = np.array(q["points"], float)
        bb = (*p.min(0), *p.max(0))
        x = [(iou(bb, (r.x1, r.y1, r.x2, r.y2)), r) for r in C[C.frame == int(q["frame"])].itertuples()]
        if x and max(x, key=lambda z: z[0])[0] > .3:
            r = max(x, key=lambda z: z[0])[1]
            rows.append((bb[0] - r.x1, bb[1] - r.y1, r.x2 - bb[2], r.y2 - bb[3]))
    return dict(zip(("left", "top", "right", "bottom"), (np.median(rows, 0).round(1) + 0.0).tolist())) | dict(n=len(rows)) if rows else None


def read_truth(path):
    t = pd.read_csv(path, comment="#", dtype={"track_ids": str, "name": str})
    t.columns = [c.strip() for c in t.columns]
    need = {"name", "length_m", "width_m"}
    if not need <= set(t.columns):
        raise SystemExit(f"{path}: needs columns {sorted(need)} (optional: area_m2, track_ids)")
    for c in ("area_m2", "track_ids"):
        if c not in t:
            t[c] = np.nan
    if t.name.duplicated().any():
        raise SystemExit(f"{path}: duplicate names: {t.name[t.name.duplicated()].tolist()}")
    for c in ("length_m", "width_m", "area_m2"):
        t[c] = pd.to_numeric(t[c], errors="raise")
    bad = t[(t.length_m > 10) | (t.width_m > 10) | (t.length_m <= 0) | (t.width_m <= 0)]
    if len(bad):
        raise SystemExit(f"{path}: these look wrong. Use metres, not centimetres: {bad.name.tolist()}")
    return t


def truth_rows(truth, C, outlines, normals, cam, scale):
    rows = []
    for t in truth.itertuples():
        box_true = t.length_m * t.width_m
        o = dict(name=t.name, track_ids="", tape_length=t.length_m, tape_width=t.width_m, tape_box=box_true, tape_area=t.area_m2 if pd.notna(t.area_m2) else np.nan)
        mine = [q for q in outlines if q["name"] == t.name]
        ids = [int(float(x)) for x in re.split(r"[;,\s]+", t.track_ids.strip()) if x] if isinstance(t.track_ids, str) else outline_tracks(mine, C)
        if ids and len(C):
            unknown = [i for i in ids if i not in set(C.track_id)]
            if unknown:
                raise SystemExit(f"truth row {t.name}: no such track_id {unknown}")
            x = C[C.track_id.isin(ids) & ~C.cut_by_edge & C.distance_m.between(*FINAL_RANGE) & C.area_new.notna()]
            o["track_ids"] = " ".join(map(str, ids))
            if len(x):
                o.update(model_area=x.area_new.median(), model_length=x.length_m.median(), model_width=x.width_m.median(), model_views=len(x),
                         old_area=C[C.track_id.isin(ids)].sort_values("frame").area_old.iloc[0])
        ms = [m for q in mine if (m := measure(q["points"], normals.get(int(q["frame"])), cam, scale))]
        if ms:
            o.update(hand_area=np.median([m["area"] for m in ms]), hand_length=np.median([m["length"] for m in ms]),
                     hand_width=np.median([m["width"] for m in ms]), hand_views=len(ms))
        rows.append(o)
    T = pd.DataFrame(rows)
    for c in "model_area model_length model_width model_views old_area hand_area hand_length hand_width hand_views".split():
        if c not in T:
            T[c] = np.nan
    return T


def leave_one_out(est, true):
    """Calibrate one multiplier on all the other defects, test on the one left out."""
    k = np.log(np.asarray(true) / np.asarray(est))
    if len(k) < 3:
        return None
    err = [abs(est[i] * math.exp(np.median(np.delete(k, i))) / true[i] - 1) for i in range(len(k))]
    return float(math.exp(np.median(k))), np.array(err)


def evaluate_dir(out, truth_path=None, outlines_path=None, trim=(0.0, 0.0)):
    out = Path(out).expanduser()
    meta, cam, scale, P, normals = load(out)
    C = coefficient_rows(out, trim)
    F = final_areas(C)
    C.to_csv(out / "coefficients.csv", index=False)
    F.to_csv(out / "defects.csv", index=False)
    S = dict(trim_px=list(trim), milestone_a={}, milestone_b=None)
    patches = pd.read_csv(out / "patches.csv")
    flat = C[~C.upright & ~C.cut_by_edge] if len(C) else C
    for key, data, cols in (("road_patches", patches, ("area_old", "area_new")),
                            ("detector_boxes", flat, ("area_old", "area_fixed_horizon", "area_live", "area_new"))):
        T, _ = spread_table(data, cols)
        if T is not None:
            S["milestone_a"][key] = dict(tracks=len(T)) | {c: dict(spread=float(T[f"{c}:spread"].median()), ratio=float(T[f"{c}:ratio"].median())) for c in cols}
    outlines = json.loads(Path(outlines_path).read_text()) if outlines_path else []
    T = None
    if truth_path:
        truth = read_truth(truth_path)
        missing = sorted({q["name"] for q in outlines} - set(truth.name))
        if missing:
            raise SystemExit(f"outlines name {missing}, which are not in {truth_path}")
        T = truth_rows(truth, C, outlines, normals, cam, scale)
        T.to_csv(out / "tape_comparison.csv", index=False)
        S["milestone_b"] = milestone_b(T)
        S["box_padding_px"] = box_padding(outlines, coefficient_rows(out))
    S["trim_sweep"] = []
    for ty in (0, 3, 6, 9, 12):
        X = coefficient_rows(out, (0, ty))
        A, _ = spread_table(X[~X.upright & ~X.cut_by_edge] if len(X) else X, ("area_new",))
        b = milestone_b(truth_rows(truth, X, outlines, normals, cam, scale)).get("model_boxes") if T is not None else None
        S["trim_sweep"].append(dict(trim_y=ty, a_spread=float(A["area_new:spread"].median()) if A is not None else None, b_error=b["median_error"] if b else None, b_ratio=b["median_ratio"] if b else None))
    if len(C):
        annotate(out, C, F, P, meta, cam)
        imgs = defect_images(out, C, F, T)
        F["image"] = F.track_id.map(imgs).fillna("")
        F.to_csv(out / "defects.csv", index=False)
        if T is not None:
            T["images"] = T.track_ids.map(lambda s: " ".join(imgs[int(i)] for i in str(s).split() if int(i) in imgs))
            T.to_csv(out / "tape_comparison.csv", index=False)
    label_page(out, C, meta)
    write_report(out, meta, C, F, P, patches, S, T)
    (out / "summary.json").write_text(json.dumps(S, indent=1))
    print_summary(out, meta, S, C, P)
    return S


def milestone_b(T):
    B = {}
    for kind, est in (("production_today", "old_area"), ("model_boxes", "model_area"), ("hand_outlines", "hand_area")):
        # A drawn outline is the hole itself, so it is judged on the true area when the tape gives one;
        # boxes are always judged on length x width, which is what a box covers.
        target = T.tape_area.where(T.tape_area.notna(), T.tape_box) if kind == "hand_outlines" else T.tape_box
        k = T[est].notna() & target.notna()
        if kind == "hand_outlines" and T.tape_area.isna().all():
            est_v = (T.hand_length * T.hand_width)[k].values      # like for like: bounding length x width
        else:
            est_v = T[est][k].values
        if not k.any():
            continue
        tr = target[k].values
        e = abs(est_v / tr - 1)
        B[kind] = dict(n=int(k.sum()), median_error=float(np.median(e)), worst_error=float(e.max()), within_target=int((e <= MILESTONE_B_MAX).sum()),
                       median_ratio=float(np.median(est_v / tr)))
        loo = leave_one_out(est_v, tr)
        if loo:
            B[kind]["calibrated"] = dict(multiplier=loo[0], median_error=float(np.median(loo[1])), worst_error=float(loo[1].max()), within_target=int((loo[1] <= MILESTONE_B_MAX).sum()))
    # What the tape says the two constants should be. Width fixes H, length fixes f*H.
    for kind, L, W in (("hand_outlines", "hand_length", "hand_width"), ("model_boxes", "model_length", "model_width")):
        k = T[L].notna()
        if k.sum():
            kw, kl = float(np.median(T.tape_width[k] / T[W][k])), float(np.median(T.tape_length[k] / T[L][k]))
            B["constants_from_tape"] = dict(source=kind, n=int(k.sum()), width_factor=kw, length_factor=kl)
            break
    return B


# ----------------------------------------------------------------------------- outputs

def ffmpeg_exe():
    """The system ffmpeg, or the copy that comes with the imageio-ffmpeg package when the machine has none."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("ffmpeg not found: install it, or pip install imageio-ffmpeg")


def annotate(out, C, F, P, meta, cam):
    by, hz = {k: g for k, g in C.groupby("frame")}, dict(zip(P.frame, P.horizon_row))
    fin = F.set_index("track_id").area_new.to_dict()
    frames = sorted((out / "frames").glob("*.jpg"))
    p = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{cam.w}x{cam.h}", "-r", f"{meta['fps'] / meta['step']:.4f}",
                          "-i", "-", "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(out / "annotated.mp4")], stdin=subprocess.PIPE)
    for f in frames:
        i, im = int(f.stem), cv2.imread(str(f))
        if np.isfinite(hz.get(i, np.nan)):
            y = int(hz[i])
            cv2.line(im, (0, y), (cam.w, y), (255, 200, 0), 1)
            cv2.putText(im, f"horizon v0={y}", (cam.w - 260, y - 8), 0, .7, (255, 200, 0), 2)
        for r in by[i].itertuples() if i in by else []:
            cv2.rectangle(im, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), (0, 220, 255), 2)
            lines = [f"#{r.track_id} {r.label}" + (f" {r.distance_m:.1f} m" if np.isfinite(r.distance_m) else "")]
            if np.isfinite(r.area_new) and not r.upright:
                lines += [f"old {r.area_old:.2f} x {r.coefficient:.2f} = {r.area_new:.2f} m2", f"final {fin.get(r.track_id, np.nan):.2f} m2"]
            y, x = int(r.y1) - 8 - 22 * (len(lines) - 1), max(4, min(int(r.x1), cam.w - 380))
            for k, s in enumerate(lines):
                yy = max(20, y) + 22 * k
                cv2.putText(im, s, (x, yy), 0, .62, (0, 0, 0), 4)
                cv2.putText(im, s, (x, yy), 0, .62, (255, 255, 255), 2)
        p.stdin.write(im.tobytes())
    p.stdin.close()
    if p.wait():
        raise SystemExit("ffmpeg failed while writing annotated.mp4")


def _text(im, s, xy, scale=.7, color=(255, 255, 255)):
    cv2.putText(im, s, xy, 0, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(im, s, xy, 0, scale, color, 2, cv2.LINE_AA)


def defect_images(out, C, F, T=None):
    """defects/<id>_<class>.jpg: one sheet per defect. On top, the view nearest its stored distance, with the box and
    the final area. Below, the same defect at up to five distances: the old area grows as the camera closes in, the new
    one should not (Milestone A). With tape measurements the sheet also shows the tape area (Milestone B)."""
    d = out / "defects"
    d.mkdir(exist_ok=True)
    for f in d.glob("*.jpg"):
        f.unlink()
    tape = {}
    if T is not None:
        for r in T.itertuples():
            for i in str(r.track_ids).split():
                tape[int(i)] = (r.name, r.tape_length, r.tape_width, r.tape_area)
    im = lambda fr: cv2.imread(str(out / "frames" / f"{int(fr):06d}.jpg"))
    names = {}
    for r in F.itertuples():
        x = C[(C.track_id == r.track_id) & ~C.cut_by_edge & C.area_new.notna()].sort_values("distance_m")
        if not len(x) or not np.isfinite(r.area_new):
            continue
        best = x.loc[(x.distance_m - r.distance_m).abs().idxmin()]
        top = im(best.frame)
        if top is None:
            continue
        cv2.rectangle(top, (int(best.x1), int(best.y1)), (int(best.x2), int(best.y2)), (0, 220, 255), 3)
        top = cv2.resize(top, (1280, int(round(top.shape[0] * 1280 / top.shape[1]))), interpolation=cv2.INTER_AREA)
        lines = [f"#{r.track_id} {r.label}   seen at {r.first_time_s:.1f} s   confidence {r.confidence}",
                 f"NEW area {r.area_new:.2f} m2  ({r.length_m:.2f} m along x {r.width_m:.2f} m across), median of {r.views_used} views at {FINAL_RANGE[0]:g}-{FINAL_RANGE[1]:g} m",
                 f"OLD area (production: pixels / 145 / 145, first box) {r.area_old:.2f} m2"]
        if r.track_id in tape:
            n, L, W, A = tape[r.track_id]
            ref = A if np.isfinite(A) else L * W
            lines.append(f"TAPE '{n}': {L:.2f} x {W:.2f} = {L * W:.2f} m2" + (f", area {A:.2f} m2" if np.isfinite(A) else "") + f"   new / tape = {r.area_new / ref:.2f}")
        for k, s in enumerate(lines):
            _text(top, s, (14, 34 + 32 * k), .74, (120, 255, 255) if s.startswith(("NEW", "TAPE")) else (255, 255, 255))
        # the approach: up to five views spread over the distances this defect was seen at, farthest first
        near = x[x.distance_m <= 15]
        pick = near.iloc[np.unique(np.linspace(0, len(near) - 1, min(5, len(near))).round().astype(int))[::-1]] if len(near) else near
        tiles = []
        for q in pick.itertuples():
            fr = im(q.frame)
            if fr is None:
                continue
            m = max(q.x2 - q.x1, q.y2 - q.y1) * .6 + 40
            x1, y1 = int(max(0, q.x1 - m)), int(max(0, q.y1 - m))
            x2, y2 = int(min(fr.shape[1], q.x2 + m)), int(min(fr.shape[0], q.y2 + m))
            cv2.rectangle(fr, (int(q.x1), int(q.y1)), (int(q.x2), int(q.y2)), (0, 220, 255), 2)
            t = fr[y1:y2, x1:x2]
            s = min(200 / t.shape[0], 250 / t.shape[1])
            t = cv2.resize(t, (max(1, int(t.shape[1] * s)), max(1, int(t.shape[0] * s))), interpolation=cv2.INTER_AREA)
            t = cv2.copyMakeBorder(t, 0, 200 - t.shape[0], 0, 250 - t.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
            t = cv2.copyMakeBorder(t, 0, 84, 0, 6, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            for k, s in enumerate((f"{q.distance_m:.1f} m away, t={q.time_s:.1f} s", f"old {q.area_old:.2f} m2", f"new {q.area_new:.2f} m2")):
                cv2.putText(t, s, (6, 226 + 24 * k), 0, .58, (120, 255, 255) if k == 2 else (235, 235, 235), 1 if k == 0 else 2, cv2.LINE_AA)
            tiles.append(t)
        if tiles:
            row = np.hstack(tiles)
            row = cv2.copyMakeBorder(row, 6, 6, 4, max(0, 1276 - row.shape[1]), cv2.BORDER_CONSTANT, value=(30, 30, 30))[:, :1280]
            top = np.vstack([top, row])
        name = f"{int(r.track_id):03d}_{re.sub(r'[^a-z0-9]+', '_', str(r.label).lower()).strip('_')}.jpg"
        cv2.imwrite(str(d / name), top, [cv2.IMWRITE_JPEG_QUALITY, 88])
        names[r.track_id] = f"defects/{name}"
    return names


def label_page(out, C, meta):
    """Self-contained page for drawing the true outline of each measured defect."""
    files = [f.name for f in sorted((out / "frames").glob("*.jpg"))]
    boxes = {}
    for r in C.itertuples() if len(C) else []:
        boxes.setdefault(int(r.frame), []).append([int(r.track_id), r.label, r.x1, r.y1, r.x2, r.y2])
    (out / "label.html").write_text(LABEL_HTML.replace("__FILES__", json.dumps(files)).replace("__BOXES__", json.dumps(boxes))
                                    .replace("__W__", str(meta["width"])).replace("__H__", str(meta["height"])))


LABEL_HTML = """<!doctype html><meta charset=utf-8><title>Outline defects</title><style>
body{font:14px system-ui;margin:0;background:#16181c;color:#eee}header{padding:8px 12px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
button,input{font:inherit;padding:4px 10px}canvas{display:block;width:100%;cursor:crosshair}#list{padding:8px 12px}</style>
<header><button id=prev>&larr;</button><input id=s type=range min=0 style="flex:1;min-width:200px"><button id=next>&rarr;</button><span id=info></span>
<label>name <input id=name size=8 placeholder="P1"></label><button id=save>Save outline (Enter)</button><button id=undo>Undo point (Z)</button>
<label><input type=checkbox id=bx checked> model boxes</label><button id=dl>Download outlines.json</button></header>
<canvas id=c width=__W__ height=__H__></canvas><div id=list></div><script>
const files=__FILES__,boxes=__BOXES__,$=i=>document.getElementById(i),c=$('c'),g=c.getContext('2d');let k=0,pts=[],saved=[],im=new Image();
$('s').max=files.length-1;const fr=()=>parseInt(files[k]);
function draw(){g.drawImage(im,0,0,c.width,c.height);g.lineWidth=2;g.font='18px sans-serif';
if($('bx').checked)for(const b of boxes[fr()]||[]){g.strokeStyle='#ffd400';g.strokeRect(b[2],b[3],b[4]-b[2],b[5]-b[3]);g.fillStyle='#ffd400';g.fillText('#'+b[0]+' '+b[1],b[2],b[3]-4)}
for(const q of saved.filter(q=>q.frame==fr()))poly(q.points,'#35d07f',q.name,true);poly(pts,'#ff4d6d','',false);
$('info').textContent='frame '+fr()+' ('+(k+1)+'/'+files.length+')';
$('list').replaceChildren(...saved.map((q,i)=>{const b=document.createElement('button');b.textContent=q.name+' @'+q.frame+' (remove)';b.onclick=()=>{saved.splice(i,1);draw()};return b}))}
function poly(p,col,n,close){if(!p.length)return;g.strokeStyle=col;g.fillStyle=col;g.beginPath();p.forEach((q,i)=>i?g.lineTo(...q):g.moveTo(...q));if(close)g.closePath();g.stroke();
for(const q of p)g.fillRect(q[0]-2,q[1]-2,4,4);if(n)g.fillText(n,p[0][0],p[0][1]-6)}
function go(i){k=Math.max(0,Math.min(files.length-1,i));$('s').value=k;pts=[];im=new Image();im.onload=draw;im.src='frames/'+files[k]}
c.onclick=e=>{const r=c.getBoundingClientRect();pts.push([Math.round((e.clientX-r.left)*c.width/r.width*10)/10,Math.round((e.clientY-r.top)*c.height/r.height*10)/10]);draw()};
function save(){const n=$('name').value.trim();if(!n){$('name').focus();return}if(pts.length<3)return;saved.push({name:n,frame:fr(),points:pts});pts=[];draw()}
$('save').onclick=save;$('undo').onclick=()=>{pts.pop();draw()};$('bx').onchange=draw;$('prev').onclick=()=>go(k-1);$('next').onclick=()=>go(k+1);$('s').oninput=e=>go(+e.target.value);
$('dl').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(saved)],{type:'application/json'}));a.download='outlines.json';a.click()};
onkeydown=e=>{if(e.key=='Enter')save();if(e.target.tagName=='INPUT'&&e.target.type!='range')return;if(e.key=='ArrowLeft')go(k-1);if(e.key=='ArrowRight')go(k+1);if(e.key=='z'){pts.pop();draw()}};go(0)</script>"""


def coefficient_fit(C):
    """How well three ideas explain the per-frame coefficient. Returned as the typical error of each."""
    u = C.dropna(subset=["coefficient"])
    u = u[(u.coefficient > 0) & ~u.upright & ~u.cut_by_edge] if len(u) else u
    if len(u) < 10:
        return None
    L = np.log(u.coefficient)
    const = float(np.median(abs(np.exp(L - L.median()) - 1)))
    fixed = float(np.median(abs(u.area_fixed_horizon / u.area_new - 1)))
    live = u.dropna(subset=["area_live"])
    return dict(n=len(u), low=float(u.coefficient.quantile(.05)), high=float(u.coefficient.quantile(.95)), one_constant=const, fixed_horizon=fixed,
                live=float(np.median(abs(live.area_live / live.area_new - 1))) if len(live) else np.nan)


def print_summary(out, meta, S, C, P):
    print("\n=== constants of this recording")
    print(f"f*H = {meta['fh']:.0f} px*m ({meta['fh_source']}), H = {meta['camera_height_m']} m, implied focal = {meta['implied_focal']:.0f} px, "
          f"horizon row {P.horizon_row.min():.0f}..{P.horizon_row.max():.0f} (changes per frame)")
    for w in meta["warnings"]:
        print("WARNING:", w)
    q = coefficient_fit(C) if len(C) else None
    if q:
        print(f"coefficient: 90% of views between {q['low']:.2f} and {q['high']:.2f}. Error if it were one constant: {q['one_constant']:.0%}; "
              f"a fixed formula per row: {q['fixed_horizon']:.0%}; on the go vs whole clip: {q['live']:.0%}")
    print("=== Milestone A (spread of one defect's area while approaching, target under 15%)")
    for k, v in S["milestone_a"].items():
        print(f"{k} ({v['tracks']}): " + ", ".join(f"{c} {x['spread']:.1%} ({x['ratio']:.1f}x)" for c, x in v.items() if c != "tracks"))
    if not S["milestone_a"]:
        print("no defect was seen from clearly different distances")
    if S["milestone_b"] is not None:
        print("=== Milestone B (error against the tape, target 25%)")
        for k, v in S["milestone_b"].items():
            print(k, json.dumps(v))
    print("outputs in", out)


def _png(fig):
    import matplotlib.pyplot as plt
    b = io.BytesIO()
    fig.savefig(b, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}">'


def write_report(out, meta, C, F, P, patches, S, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E, body = html.escape, []
    add = body.append
    pct = lambda x: "-" if x is None or not np.isfinite(x) else f"{x:.0%}"
    num = lambda x, d=2: "-" if x is None or pd.isna(x) else f"{x:.{d}f}"

    def table(head, rows):
        add("<div class=s><table><tr>" + "".join(f"<th>{E(h)}</th>" for h in head) + "</tr>" +
            "".join("<tr>" + "".join(f"<td>{E(str(c))}</td>" for c in r) + "</tr>" for r in rows) + "</table></div>")

    verdict = lambda ok: "MET" if ok else "NOT MET"
    add("<h1>Defect area report</h1>")
    add(f"<p>{E(Path(meta['video']).name)} - {meta['frames'] / meta['fps']:.1f} s, {meta['speed_kmh']:.0f} km/h, {len(F)} tracked detections</p>")
    for w in meta["warnings"]:
        add(f"<p class=w>{E(w)}</p>")
    if any(S["trim_px"]):
        add(f"<p class=w>Boxes were trimmed by {S['trim_px'][0]:g} px left and right, {S['trim_px'][1]:g} px top and bottom.</p>")

    add("<h2>1. Is the coefficient a constant?</h2>")
    q = coefficient_fit(C) if len(C) else None
    add("<p><b>No single number works, not even within one video.</b> Two numbers are fixed per recording (f&middot;H and H). One number, the horizon row, is worked out "
        "again in every analysed frame. Together they give each box its own coefficient:</p><pre>coefficient = 145&sup2; &times; (f&middot;H) &times; H / (y - v0)&sup3;      area = pixels / 145&sup2; &times; coefficient</pre>"
        "<p>That is the simple form. The script uses the full tilted road plane and the whole box, so its numbers differ a little from this formula.</p>")
    table(["Ingredient", "This recording", "Changes?"],
          [["H, camera height", f"{meta['camera_height_m']:.3f} m", "Fixed while the phone stays mounted"],
           ["f*H, scale along the road", f"{meta['fh']:.0f} px*m (from {meta['fh_source']})", "Fixed per phone, lens and mounting"],
           ["f, implied focal length", f"{meta['implied_focal']:.0f} px at 1920 wide", "Fixed per phone and lens"],
           ["v0, horizon row", f"{P.horizon_row.min():.0f} to {P.horizon_row.max():.0f}", "Every frame, as the phone tilts"],
           ["y, the box's row", "-", "Every frame, as the defect gets closer"]])
    if q:
        table(["If the coefficient were...", "Typical error against the per-frame coefficient"],
              [["one constant for the whole video", pct(q["one_constant"])], ["one fixed formula of the row (horizon never moves)", pct(q["fixed_horizon"])],
               ["calculated on the go (past frames only)", pct(q["live"])], ["calculated per frame with the whole clip", "reference"]])
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    ax[0].plot(P.time_s, P.horizon_row, label="whole clip")
    ax[0].plot(P.time_s, P.live_horizon, lw=1, label="on the go")
    ax[0].set(xlabel="time (s)", ylabel="horizon row (px)", title="Horizon v0 in each frame")
    ax[0].legend()
    if len(C):
        u = C[~C.upright & C.coefficient.notna() & C.distance_m.between(2, 15)]
        ax[1].scatter(u.distance_m, u.coefficient, s=8, alpha=.5)
    ax[1].set(xlabel="distance (m)", ylabel="coefficient", title="Coefficient against distance", yscale="log")
    add(_png(fig))

    add("<h2>2. Milestone A: the same area from any distance</h2>")
    add("<p>Spread is the standard deviation divided by the mean for one defect over its approach, 4 to 12 m. Target: under 15%. Only defects seen at least 6 times, "
        "with the farthest view at least 1.35 times the nearest, are counted.</p>")
    names = dict(area_old="production today (pixels / 145²)", area_fixed_horizon="fixed formula per row", area_live="on the go", area_new="per frame, whole clip")
    rows = []
    for key, title in (("road_patches", "Road patches (no model)"), ("detector_boxes", "Model boxes")):
        v = S["milestone_a"].get(key)
        if not v:
            rows.append([title, "-", "too few usable tracks", "-", "-", "-"])
            continue
        for c, x in v.items():
            if c != "tracks":
                rows.append([title, v["tracks"], names[c], f"{x['spread']:.1%}", f"{x['ratio']:.2f}x", verdict(x["spread"] < MILESTONE_A_MAX)])
    table(["Test", "Defects", "Method", "Spread", "Biggest / smallest", "Target"], rows)
    add("<p>Road patches test the geometry alone. Model boxes also include how steadily the model draws its boxes. If patches pass and boxes do not, the boxes are the problem.</p>")
    Tt, u = spread_table(C[~C.upright & ~C.cut_by_edge] if len(C) else C, ("area_old", "area_new"))
    if Tt is not None:
        ids = u.groupby("track_id").size().nlargest(6).index
        fig, ax = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
        for a_, t in zip(ax.flat, ids):
            x = u[u.track_id == t]
            a_.plot(x.distance_m, x.area_old, "o-", ms=3, label="today")
            a_.plot(x.distance_m, x.area_new, "o-", ms=3, label="corrected")
            a_.invert_xaxis()
            a_.set(title=f"#{t} {x.label.iloc[0]}", ylim=(0, None))
        for a_ in list(ax.flat)[len(ids):]:
            a_.axis("off")
        ax.flat[0].legend()
        fig.supxlabel("distance (m), approaching to the right")
        fig.supylabel("area (m²)")
        fig.tight_layout()
        add(_png(fig))

    add("<h3>Are the boxes too loose?</h3><p>The same test after taking pixels off the top and bottom of every box. A falling spread means the boxes are padded. "
        "A trim chosen on this clip must be confirmed on another one before it is used.</p>")
    table(["Trim, px top and bottom", "Milestone A spread", "Milestone B median error", "Estimate / tape"],
          [[s["trim_y"], pct(s["a_spread"]), pct(s["b_error"]), num(s["b_ratio"])] for s in S["trim_sweep"]])
    if S.get("box_padding_px"):
        q = S["box_padding_px"]
        add(f"<p>Measured against {q['n']} hand outlines, the model's box sticks out by {q['left']} px left, {q['top']} px top, {q['right']} px right, {q['bottom']} px bottom.</p>")

    add("<h2>3. Milestone B: true m², against the tape</h2>")
    B = S["milestone_b"]
    if not B:
        add("<p>No usable tape measurements yet. Fill truth.csv, then run <code>evaluate</code>.</p>")
    else:
        rows = []
        for key, title in (("production_today", "Production today (first-seen box)"), ("model_boxes", "Model boxes, corrected"), ("hand_outlines", "Hand outlines (geometry only)")):
            if key in B:
                v = B[key]
                rows.append([title, v["n"], pct(v["median_error"]), pct(v["worst_error"]), f"{v['within_target']}/{v['n']}", f"{v['median_ratio']:.2f}", verdict(v["median_error"] <= MILESTONE_B_MAX)])
                if "calibrated" in v:
                    c = v["calibrated"]
                    rows.append([f"  ...after one tape multiplier ({c['multiplier']:.2f})", v["n"], pct(c["median_error"]), pct(c["worst_error"]), f"{c['within_target']}/{v['n']}", "-", verdict(c["median_error"] <= MILESTONE_B_MAX)])
        table(["Method", "Defects", "Median error", "Worst", "Within 25%", "Estimate / tape", "Target"], rows)
        add("<p>The multiplier rows are honest tests: each defect is predicted using a multiplier learned from the other defects only. "
            "Boxes are compared with tape length &times; width. Hand outlines are compared with the tape area when it was given.</p>")
        k = B.get("constants_from_tape")
        if k:
            H, fh = meta["camera_height_m"] * k["width_factor"], meta["fh"] * k["length_factor"]
            k.update(camera_height_m=H, fh=fh, focal_px=fh / H)
            add(f"<p><b>What the tape says the constants are</b> (from {k['source'].replace('_', ' ')}, {k['n']} defects): H = {k['camera_height_m']:.3f} m, f&middot;H = {k['fh']:.0f}, "
                f"f = {k['focal_px']:.0f} px. Used here: H = {meta['camera_height_m']:.3f}, f&middot;H = {meta['fh']:.0f}. Reuse on other videos from this phone and mount with <code>--height</code> and <code>--fh</code>.</p>")
        table(["Defect", "IDs", "Tape L x W", "Tape area", "Production", "Model (L x W)", "Model area", "Views", "Outline area"],
              [[r.name, r.track_ids, f"{r.tape_length:.2f} x {r.tape_width:.2f} = {r.tape_box:.2f}", num(r.tape_area), num(r.old_area), f"{num(r.model_length)} x {num(r.model_width)}", num(r.model_area), num(r.model_views, 0), num(r.hand_area)]
               for r in T.rename(columns={"name": "nm"}).rename(columns={"nm": "name"}).itertuples()])

    add("<h2>4. Every detection</h2>")
    table(["ID", "Class", "Seen at (s)", "Old m²", "New m²", "L x W m", "Distance m", "Views used", "Confidence", "Note"],
          [[r.track_id, r.label, r.first_time_s, num(r.area_old), num(r.area_new), f"{num(r.length_m)} x {num(r.width_m)}", num(r.distance_m, 1), r.views_used, r.confidence, r.note] for r in F.itertuples()])
    pics = [r for r in F.itertuples() if getattr(r, "image", "")]
    if pics:
        add("<h2>5. Defect images</h2><p>One sheet per defect, also saved in the <code>defects/</code> folder. The top shows the view nearest the stored distance with the final area. "
            "The strip below shows the same defect from far (left) to near (right): the old area grows as the camera gets closer, the new one should stay about the same.</p>")
        add("<div class=g>" + "".join(f"<a href='{E(r.image)}'><img loading=lazy src='{E(r.image)}' alt='defect {r.track_id}'></a>" for r in pics) + "</div>")
    (out / "report.html").write_text("<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width'><title>Defect area report</title><style>"
        "body{font:15px/1.5 system-ui;max-width:1100px;margin:24px auto;padding:0 16px;color:#1b1e23;background:#fff}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #d5d9e0;padding:4px 8px;text-align:left;white-space:nowrap}"
        "th{background:#f1f3f6}.s{overflow-x:auto;margin:12px 0}.w{background:#fff4d6;padding:8px 12px;border-radius:6px}img{max-width:100%}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(min(100%,520px),1fr));gap:12px}.g img{border:1px solid #d5d9e0;border-radius:4px}pre{background:#f1f3f6;padding:10px;overflow-x:auto}</style>" + "\n".join(body))


# ----------------------------------------------------------------------------- command line

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="analyse a video")
    r.add_argument("--video", required=True, help="path, http(s) URL or s3:// URI")
    r.add_argument("--gpx", required=True, help="path, http(s) URL or s3:// URI")
    r.add_argument("--height", type=float, required=True, help="lens height above the road in metres, measured with the rider on")
    r.add_argument("--model", required=True, type=lambda s: Path(s).expanduser(), help="YOLO .pt defect model")
    r.add_argument("--out", required=True)
    r.add_argument("--truth", help="tape measurements CSV")
    r.add_argument("--outlines", help="outlines.json saved from label.html")
    r.add_argument("--fh", type=float, help="use this f*H instead of the GPX one (from an earlier report)")
    r.add_argument("--focal", type=float, help="known focal length in px at 1920 wide; replaces the GPX scale")
    r.add_argument("--gpx-offset", type=float, help="GPX seconds at video start; stops the stretch-to-fit")
    r.add_argument("--conf", type=float, default=.35, help="confidence to start a track (production uses 0.35)")
    r.add_argument("--conf-low", type=float, default=.15, help="confidence to continue a track")
    r.add_argument("--imgsz", type=int, default=640, help="model input size (production uses 640)")
    r.add_argument("--device", default="auto", help="auto, cpu or cuda")
    r.add_argument("--aws-profile")
    e = sub.add_parser("evaluate", help="re-score a finished run against the tape")
    e.add_argument("--out", required=True)
    e.add_argument("--truth")
    e.add_argument("--outlines")
    for q in (r, e):
        q.add_argument("--trim-x", type=float, default=0.0, help="px taken off the left and right of every box")
        q.add_argument("--trim-y", type=float, default=0.0, help="px taken off the top and bottom of every box")
    a = p.parse_args(argv)
    if a.cmd == "run":
        if not .3 <= a.height <= 4:
            p.error("--height is in metres, for example 1.152")
        run(a)
    else:
        evaluate_dir(a.out, a.truth, a.outlines, (a.trim_x, a.trim_y))


if __name__ == "__main__":
    sys.exit(main())
