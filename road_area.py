#!/usr/bin/env python3
"""Distance-corrected road defect area from a phone video and its GPX.

    road_area.py run      --video V --gpx G --height 1.152 --model M --out OUT [--truth truth.csv]
    road_area.py evaluate --out OUT --truth truth.csv [--outlines outlines.json]
    road_area.py rescore  --rows post_construction_output.csv --video V --gpx G --height 1.152 --model M --out OUT

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
import tempfile
import time
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

class GpxSegment:
    """One <trkseg>: speed and position against seconds since its first point."""

    def __init__(self, P):
        self.T = P[:, 0] - P[0, 0]
        self.lat, self.lon, self.seconds = P[:, 1], P[:, 2], float(self.T[-1])
        x = (P[:, 2] - P[0, 2]) * 111320 * math.cos(math.radians(P[0, 1]))
        y = (P[:, 1] - P[0, 1]) * 111320
        self.metres = float(np.hypot(np.diff(x), np.diff(y)).sum())
        v = np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(self.T), 0.2)
        self.v = gaussian_filter1d(median_filter(v, min(5, len(v) | 1), mode="nearest"), 1.0, mode="nearest")
        self.t = (self.T[1:] + self.T[:-1]) / 2


class Gpx:
    """Speed against video time. A merged video comes with one GPX segment per recording, so video time is mapped
    piece by piece: piece = (first video second, GPX segment, GPX seconds at that video second)."""

    def __init__(self, path, duration, offset=None):
        text = Path(path).read_text()
        self.segments = []
        for seg in re.findall(r"<trkseg\b.*?</trkseg>", text, re.S) or [text]:
            pts = []
            for m in re.finditer(r"<trkpt\b([^>]*)>(.*?)</trkpt>", seg, re.S):
                lat, lon = (re.search(rf'\b{k}\s*=\s*["\']([-\d.]+)', m[1]) for k in ("lat", "lon"))
                t = re.search(r"<time>\s*(.*?)\s*</time>", m[2], re.S)
                if lat and lon and t:
                    pts.append((datetime.fromisoformat(t[1].replace("Z", "+00:00")).timestamp(), float(lat[1]), float(lon[1])))
            pts = sorted({p[0]: p for p in pts}.values())
            if len(pts) >= 4:
                self.segments.append(GpxSegment(np.array(pts)))
        self.n = sum(len(s.T) for s in self.segments)
        if not self.segments:
            raise SystemExit(f"GPX {path}: fewer than 4 timed points in every track segment. At least 4 are needed.")
        self.duration = duration
        self.gpx_seconds = float(sum(s.seconds for s in self.segments))
        self.metres = float(sum(s.metres for s in self.segments))
        if offset is not None:
            self.pieces = [(0.0, 0, float(offset), 1.0)]
        else:
            # The app starts and stops both together, so by default the GPX is stretched over the video. Segments are
            # laid end to end without the time between them, which is what merging the videos does.
            k, at, self.pieces = self.gpx_seconds / duration, 0.0, []
            for i, s in enumerate(self.segments):
                self.pieces.append((at / k, i, 0.0, k))
                at += s.seconds

    def _piece_at(self, sec):
        return max((p for p in self.pieces if p[0] <= sec + 1e-9), key=lambda p: p[0], default=self.pieces[0])

    def _where(self, sec):
        p = self._piece_at(sec)
        return self.segments[p[1]], p[2] + (sec - p[0]) * p[3]

    def speed(self, sec):
        s, t = self._where(sec)
        return float(np.interp(t, s.t, s.v))

    def position(self, sec):
        s, t = self._where(sec)
        return float(np.interp(t, s.T, s.lat)), float(np.interp(t, s.T, s.lon))


def _video_speed(recs, k, fps, gap):
    """Times and log speed of the camera, known only up to one factor: the distance moved over the baseline is the
    camera height divided by `hrel`, and the height does not change."""
    fr = np.array([r["frame"] for r in recs])[k]
    h = np.array([r["hrel"] for r in recs])[k]
    ok = np.isfinite(h) & (h > 0)
    if ok.sum() < 9:
        return np.array([]), np.array([])
    return (fr[ok] - gap / 2) / fps, median_filter(-np.log(h[ok]), 9, mode="nearest")


def _corr(t, L, seg, start_s, offset):
    tt = offset + (t - start_s)
    v = np.interp(tt, seg.t, seg.v)
    ok = (tt >= 0) & (tt <= seg.seconds) & (v > 1.0)
    if ok.sum() < max(30, .6 * len(t)):
        return -1.0, int(ok.sum())
    return float(np.corrcoef(L[ok], np.log(v[ok]))[0, 1]), int(ok.sum())


def speed_match(recs, k, gpx, fps, gap):
    """How well the camera's own speed follows the GPX speed as currently mapped (correlation, 1 is perfect)."""
    t, L = _video_speed(recs, k, fps, gap)
    v = np.array([gpx.speed(x) for x in t])
    ok = v > 1.0
    return round(float(np.corrcoef(L[ok], np.log(v[ok]))[0, 1]), 3) if ok.sum() >= 30 else None


def align_gpx(recs, gpx, bounds, fps, gap):
    """For each recording in the video, the GPX segment and offset whose speed rises and falls with the camera's."""
    fr = np.array([r["frame"] for r in recs])
    out = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        t, L = _video_speed(recs, (fr >= lo + gap) & (fr < hi), fps, gap)
        start, dur, best = lo / fps, (hi - lo) / fps, (-1.0, 0, 0.0)
        for si, s in enumerate(gpx.segments):
            for o in np.arange(-.5 * dur, s.seconds - .5 * dur + .25, .25):
                c, _ = _corr(t, L, s, start, o) if len(t) else (-1.0, 0)
                if c > best[0]:
                    best = (c, si, float(o))
        out.append(dict(start_s=start, gpx_segment=best[1], gpx_offset_s=best[2], speed_match=round(best[0], 3)))
        print(f"  video {start:.0f}-{start + dur:.0f} s follows GPX segment {best[1] + 1} from {best[2]:.1f} s (speed match {best[0]:.2f})", flush=True)
    return out


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


THUMB = (64, 36)


def thumb(frame):
    """Tiny grey copy of a frame: enough to find a frame again from its picture."""
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), THUMB, interpolation=cv2.INTER_AREA)


class CutFinder:
    """Spots the jump where one recording ends and the next begins in a merged video."""

    def __init__(self, fps):
        self.prev, self.d, self.last, self.min_gap = None, [], -10 ** 9, int(5 * fps)

    def push(self, i, th):
        th, cut = th.astype(np.int16), False
        if self.prev is not None:
            x = float(np.abs(th - self.prev).mean())
            base = float(np.median(self.d[-30:])) if len(self.d) >= 10 else np.inf
            if x > 25 and x > 6 * max(base, 1.0) and i - self.last > self.min_gap:
                cut, self.last = True, i
            self.d.append(x)
        self.prev = th
        return cut


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

    t_start = time.time()
    det, motion = Detector(a.model, a.device, a.imgsz), Motion(cam, step, gap)
    print(f"model {Path(a.model).name} on {det.device}", flush=True)
    use_patches = getattr(a, "patches", True)
    patches, tracker = PatchTracker(cam, step, vid.fps), Tracker(a.conf, int(round(vid.fps / step)))
    recs, boxes, live, fh_seen, thumbs, cuts, n_frames = [], [], [], [], [], [], 0
    cut_finder, piece_k = CutFinder(vid.fps), 0
    for i, frame in enumerate(vid):
        n_frames += 1
        thumbs.append(thumb(frame))
        if cut_finder.push(i, thumbs[-1]):
            # A merged video jumps to the next recording here: nothing tracked before the jump carries over.
            cuts.append(i)
            motion, tracker.tracks, patches.live, piece_k = Motion(cam, step, gap), {}, [], len(recs)
            print(f"  frame {i}: the video jumps to another recording", flush=True)
        if use_patches:
            patches.push(i, frame)
        r = motion.push(i, frame)
        if r is None:
            continue
        recs.append(r)
        k = len(recs) - 1
        # ---- the coefficient "on the go": only this frame and earlier ones are used
        n_live = smooth_normal(recs[piece_k:], k - piece_k, win, causal=True)
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
    np.save(out / "thumbs.npy", np.array(thumbs))

    # ---- second look, with the whole clip available
    bounds = [0] + cuts + [n_frames]
    if len(gpx.segments) > 1 or cuts:
        found = align_gpx(recs, gpx, bounds, vid.fps, gap)
        if len(cuts) + 1 != len(gpx.segments):
            warnings.append(f"The GPX has {len(gpx.segments)} track segments but the video {len(cuts) + 1} recordings: each recording was matched to the GPX by its speed.")
        for p in found:
            if p["speed_match"] < .3:
                warnings.append(f"Video from {p['start_s']:.0f} s: its speed does not follow any GPX segment (match {p['speed_match']:.2f}); the scale there is less certain.")
        old = [gpx._piece_at(p["start_s"]) for p in found]
        gpx.pieces = [(p["start_s"], p["gpx_segment"], p["gpx_offset_s"], 1.0) if p["speed_match"] >= .3 else
                      (p["start_s"], q[1], q[2] + (p["start_s"] - q[0]) * q[3], q[3]) for p, q in zip(found, old)]
    fh_all = estimate_fh(recs, gpx, cam, vid.fps, gap)
    good = fh_all[np.isfinite(fh_all)]
    if len(good) < 5 and not a.fh:
        raise SystemExit("Could not recover the camera motion in enough frames (blurred, static or turning video?). Pass --fh to continue.")
    fh = a.fh or (a.focal * a.height if a.focal else float(np.median(good)))
    halves = [float(np.median(x)) for x in np.array_split(good, 2)] if len(good) >= 10 else []
    rec_frames = np.array([r["frame"] for r in recs])
    pieces, normals = [], []
    for j, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        k = (rec_frames >= lo) & (rec_frames < hi)
        normals += fill_normals([r for r, kk in zip(recs, k) if kk], win)
        x = fh_all[k & np.isfinite(fh_all)]
        g = gpx._piece_at(lo / vid.fps)
        pieces.append(dict(start_frame=int(lo), end_frame=int(hi), gpx_segment=g[1], gpx_offset_s=round(g[2], 2), gpx_rate=g[3],
                           speed_match=speed_match(recs, k, gpx, vid.fps, gap),
                           fh=a.fh or (a.focal * a.height if a.focal else float(np.median(x)) if len(x) >= 20 else fh), fh_frames=int(len(x))))
    if len(pieces) == 1 and halves and abs(halves[0] - halves[1]) / fh > .10:
        warnings.append(f"f*H differs between the first and second half of the clip ({halves[0]:.0f} vs {halves[1]:.0f}): scale is less certain.")
    if len(pieces) > 1:
        f = [p["fh"] for p in pieces]
        if max(f) / min(f) > 1.10:
            warnings.append("f*H differs between the recordings in this video (" + ", ".join(f"{x:.0f}" for x in f) + "): each recording uses its own.")
    N = [n for n in normals if n is not None]
    n_fixed = np.median(N, 0) / np.linalg.norm(np.median(N, 0))
    meta = dict(video=str(a.video).split("?")[0], gpx=str(a.gpx).split("?")[0], fps=vid.fps, frames=n_frames, width=vid.w, height=vid.h,
                source_size=vid.src, step=step, gap=gap, working_focal=cam.f, camera_height_m=a.height, fh=fh,
                fh_source="--fh" if a.fh else "--focal x height" if a.focal else "GPX", fh_gpx=float(np.median(good)) if len(good) else None,
                fh_halves=halves, implied_focal=fh / a.height, gpx_points=gpx.n, gpx_seconds=gpx.gpx_seconds, gpx_metres=gpx.metres,
                gpx_segments=len(gpx.segments), pieces=pieces,
                speed_kmh=v * 3.6, warnings=warnings, n_fixed=n_fixed.tolist(), planes_found=int(len(N)), planes_total=len(recs),
                model=str(a.model), conf=a.conf, imgsz=a.imgsz, device=det.device, seconds=round(time.time() - t_start))
    P = pd.DataFrame([dict(frame=r["frame"], time_s=r["frame"] / vid.fps, nx=n[0] if n is not None else np.nan, ny=n[1] if n is not None else np.nan,
                           nz=n[2] if n is not None else np.nan, horizon_row=horizon(n, cam)[0] if n is not None else np.nan,
                           roll_deg=horizon(n, cam)[1] if n is not None else np.nan, speed_kmh=gpx.speed(r["frame"] / vid.fps) * 3.6,
                           fh_this_frame=x, road_points=r["inliers"]) for r, n, x in zip(recs, normals, fh_all)]).merge(pd.DataFrame(live), on="frame")
    P.to_csv(out / "per_frame.csv", index=False)
    pd.DataFrame(boxes, columns="frame track_id label conf x1 y1 x2 y2 live_area".split()).to_csv(out / "detections.csv", index=False)
    if use_patches:
        patches.rows({r["frame"]: n for r, n in zip(recs, normals)}, cam, Scale(a.height, fh)).to_csv(out / "patches.csv", index=False)
    else:
        pd.DataFrame(columns="track_id frame distance_m pixels area_old area_new".split()).to_csv(out / "patches.csv", index=False)
    (out / "run.json").write_text(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------- scoring

def load(out):
    meta = json.loads((out / "run.json").read_text())
    cam = Camera(meta["working_focal"], meta["width"], meta["height"])
    P = pd.read_csv(out / "per_frame.csv")
    normals = {int(r.frame): np.array([r.nx, r.ny, r.nz]) for r in P.itertuples() if np.isfinite(r.ny)}
    return meta, cam, Scales(meta), P, normals


class Scales:
    """The scale of each frame. A merged video holds several recordings, and each keeps its own f*H."""

    def __init__(self, meta):
        P = meta.get("pieces") or [dict(start_frame=0, fh=meta["fh"])]
        self.starts, self.s = [p["start_frame"] for p in P], [Scale(meta["camera_height_m"], p["fh"]) for p in P]

    def at(self, frame):
        return self.s[max(0, int(np.searchsorted(self.starts, frame, side="right")) - 1)]


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
        sc = scale.at(r.frame)
        m = measure_box(b, n, cam, sc) if n is not None else None
        m0 = measure_box(b0, n, cam, sc) if n is not None else None
        f = measure_box(b, nf, cam, sc)
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
        ms = [m for q in mine if (m := measure(q["points"], normals.get(int(q["frame"])), cam, scale.at(int(q["frame"]))))]
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


# ----------------------------------------------------------------------------- new area for existing detections

LABEL_ALIASES = {"longitudial crack": "longitudinal crack"}
FRAME_MATCH_MAX = 4.0      # mean grey difference of the thumbnails; JPEG noise alone gives about 2
NEED_COLUMNS = ("id", "label", "defect_area_m2")
COLUMN_ALIASES = {"id": "object_id", "label": "defect_name"}     # the worker's own per-video CSV


def norm_label(s):
    s = re.sub(r"[_\s]+", " ", str(s).strip().lower())
    return LABEL_ALIASES.get(s, s)


def _local_name(url):
    return re.sub(r"[^\w.-]+", "_", url.split("?")[0].strip("/"))


def fetch_images(urls, folder, base, profile):
    """Evidence images by their frame_url, into `folder`. `base` is a local folder, or the S3 or http(s) location the
    frame_url is relative to. Images already there are not fetched again."""
    folder.mkdir(parents=True, exist_ok=True)
    got, s3 = {}, {}
    for u in sorted(set(urls)):
        dst = folder / _local_name(u)
        if dst.is_file() and dst.stat().st_size:
            got[u] = dst
        elif base and not re.match(r"(s3|https?)://", base):
            for p in (Path(base).expanduser() / u, Path(base).expanduser() / u.rsplit("/", 1)[-1]):
                if p.is_file():
                    shutil.copy(p, dst)
                    got[u] = dst
                    break
        elif re.match(r"https?://", u) or (base or "").startswith("http"):
            try:
                with urllib.request.urlopen(u if re.match(r"https?://", u) else base.rstrip("/") + "/" + u.lstrip("/")) as r, open(dst, "wb") as f:
                    shutil.copyfileobj(r, f)
                got[u] = dst
            except OSError as e:
                print("  could not fetch", u, e)
        else:
            key = u if u.startswith("s3://") else base.rstrip("/") + "/" + u.lstrip("/")
            s3.setdefault(key.rsplit("/", 1)[0], []).append((key.rsplit("/", 1)[1], u, dst))
    for d, files in s3.items():          # one call per folder: hundreds of single copies would take minutes
        print(f"  fetching {len(files)} images from {d}/", flush=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            inc = [x for name, _, _ in files for x in ("--include", name)]
            subprocess.run(["aws", "s3", "cp", d + "/", tmp, "--recursive", "--only-show-errors", "--exclude", "*", *inc]
                           + (["--profile", profile] if profile else []), check=True)
            for name, u, dst in files:
                if (Path(tmp) / name).is_file():
                    shutil.move(Path(tmp) / name, dst)
                    got[u] = dst
    return got


def find_frame(img, thumbs, fps):
    """The video frame an evidence image was taken from: (frame, error, error of the best frame more than a second away)."""
    z = thumb(img).reshape(-1).astype(np.int16)
    e = np.concatenate([np.abs(thumbs[i:i + 20000].reshape(-1, z.size).astype(np.int16) - z).mean(1) for i in range(0, len(thumbs), 20000)])
    k = int(e.argmin())
    w = max(1, int(fps))
    rest = np.r_[e[:max(0, k - w)], e[k + w + 1:]]
    return k, float(e[k]), float(rest.min()) if len(rest) else np.inf


def recover_boxes(rows, img, det):
    """Production keeps no box, only area = pixels / 145^2. Running the same model on the same picture gives the
    boxes back; within one picture and class, each row takes the box whose pixel count fits its area."""
    found = det(img, .05)
    out = {}
    for lab, g in rows.groupby("nlabel"):
        q = [x for x in found if norm_label(x["label"]) == lab]
        if not q:
            continue
        px = g.defect_area_m2.values * PROD_PPM ** 2
        bp = np.array([(x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]) for x in q])
        # A stored area can be smaller than its box (the part off the road is dropped), never larger.
        cost = np.array([[abs(math.log(max(p, 1) / b)) + (0 if x["conf"] >= .3 else .5) + (1 if p > b * 1.1 else 0) for b, x in zip(bp, q)] for p in px])
        for i, j in zip(*linear_sum_assignment(cost)):
            frac = px[i] / bp[j]
            if .2 < frac < 1.1:
                out[g.index[i]] = (q[j]["box"], min(frac, 1.0), q[j]["conf"])
    return out


def video_frames(want, folder, video, profile):
    """The given frames of the video as images in `folder` (at the working width), reading the video once."""
    folder.mkdir(parents=True, exist_ok=True)
    path = lambda k: folder / f"frame_{k:06d}.jpg"
    todo = {k for k in want if not path(k).is_file()}
    if todo:
        print(f"  reading {len(todo)} frames from the video", flush=True)
        with tempfile.TemporaryDirectory() as tmp:
            for i, f in enumerate(Video(fetch(video, Path(tmp), "video.mp4", profile))):
                if i in todo:
                    cv2.imwrite(str(path(i)), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    todo.discard(i)
                    if not todo:
                        break
    return {k: path(k) for k in want if path(k).is_file()}


def link_track(C, frame, lab, box, step):
    """The track in this run that drew the same box within a couple of analysed frames. The model sometimes calls one
    spot by another class in the next frames (pothole, then patching), so a box of another class on the same spot
    (overlap at least half) is taken when no box of the row's class is there."""
    near = C[(C.frame - frame).abs() <= 2 * step]
    for same, floor in ((True, .3), (False, .5)):
        best = (floor, None)
        for r in near[(near.nlabel == lab) == same].itertuples():
            s = iou(box, (r.x1, r.y1, r.x2, r.y2)) - .02 * abs(r.frame - frame) / step
            if s > best[0]:
                best = (s, r)
        if best[1] is not None:
            return best[1]
    return None


def rescore(a):
    """New area for the defects already in a post_construction_output CSV. The whole video is analysed once for the
    per-frame coefficient; each stored row is then found in it and given the area of its tracked defect. No row is
    added or removed."""
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(a.rows, dtype=str, keep_default_na=False)
    if a.job_id:
        rows = rows[rows.get("job_id", pd.Series("", index=rows.index)) == a.job_id]
    d = rows.copy()
    for c, alias in COLUMN_ALIASES.items():
        if c not in d and alias in d:
            d[c] = d[alias]
    # Where a row was taken: its evidence image (frame_url), or else its frame number in this very video.
    by_frame = ("frame_url" not in d or not d.frame_url.str.strip().any()) and "frame" in d
    has_box = all(c in d for c in ("x", "y", "width", "height"))
    missing = [c for c in NEED_COLUMNS if c not in d] + ([] if by_frame or "frame_url" in d else ["frame_url or frame"])
    if missing or not len(rows):
        raise SystemExit(f"{a.rows}: " + (f"needs columns {missing}" if missing else "no rows" + (f" for job {a.job_id}" if a.job_id else "")))
    if not (out / "run.json").is_file() or a.rerun:
        if not (a.video and a.gpx and a.height and a.model):
            raise SystemExit("first run on this --out: --video, --gpx, --height and --model are needed")
        if not .3 <= a.height <= 4:
            raise SystemExit("--height is in metres, for example 1.152")
        (out / "frames").mkdir(exist_ok=True)
        a.patches = False
        with tempfile.TemporaryDirectory() as tmp:
            video = fetch(a.video, Path(tmp), "video.mp4", a.aws_profile)
            shutil.copy(fetch(a.gpx, Path(tmp), "track.gpx", a.aws_profile), out / "track.gpx")
            _run(a, out, video, out / "track.gpx")
    else:
        print(f"reusing the analysed video in {out} (pass --rerun to analyse it again)")
    meta, cam, _, _, _ = load(out)
    C = coefficient_rows(out)
    F = final_areas(C)
    C.to_csv(out / "coefficients.csv", index=False)
    F.to_csv(out / "defects.csv", index=False)
    C["nlabel"] = C.label.map(norm_label)
    step = meta["step"]

    d["defect_area_m2"] = pd.to_numeric(d.defect_area_m2, errors="coerce")
    d["nlabel"] = d.label.map(norm_label)
    if by_frame:
        d["frame"] = pd.to_numeric(d.frame, errors="coerce")
        key = "frame"
        print(f"{len(d)} rows, located by their frame number in the video", flush=True)
        imgs = {} if has_box else video_frames(sorted({int(f) for f in d.frame.dropna()}), out / "evidence", a.video or meta["video"], a.aws_profile)
    else:
        key = "frame_url"
        print(f"{len(d)} rows; fetching their evidence images", flush=True)
        imgs = fetch_images(d.frame_url, out / "evidence", a.images, a.aws_profile)
        thumbs = np.load(out / "thumbs.npy")
    model = a.model or Path(meta["model"])
    if not has_box and not model.is_file():
        raise SystemExit(f"model not found: {model}. Pass --model with the model this video was analysed with.")
    det = None if has_box else Detector(model, a.device, meta.get("imgsz", 640))
    R = pd.DataFrame(index=d.index, columns="video_frame frame_match track_id frac x1 y1 x2 y2 how note".split(), dtype=object)
    R["note"] = ""
    for src, g in d.groupby(key):
        img = cv2.imread(str(imgs[src])) if src in imgs else None
        if by_frame:
            k, sx = int(src), meta["width"] / (img.shape[1] if img is not None else meta["source_size"][0])
            R.loc[g.index, "video_frame"] = k
        else:
            if img is None:
                R.loc[g.index, "note"] = "evidence image not found"
                continue
            k, e, e2 = find_frame(img, thumbs, meta["fps"])
            sx = meta["width"] / img.shape[1]
            R.loc[g.index, "video_frame"], R.loc[g.index, "frame_match"] = k, round(e, 2)
            if e > FRAME_MATCH_MAX or e2 < 1.5 * e:
                R.loc[g.index, "note"] = "evidence image is not in this video" if e > FRAME_MATCH_MAX else "evidence image matches several moments of the video"
                continue
        flat = g[~g.nlabel.isin(UPRIGHT) & g.defect_area_m2.gt(0)]
        if has_box:      # the worker's CSV keeps the box: nothing to recover
            boxes = {}
            for i, r in flat.iterrows():
                x, y, w, h = (float(r[c]) for c in ("x", "y", "width", "height"))
                if w > 0 and h > 0:
                    boxes[i] = ([x, y, x + w, y + h], min(1.0, r.defect_area_m2 * PROD_PPM ** 2 / (w * h)), np.nan)
        elif img is None:
            R.loc[g.index, "note"] = "frame not found in the video"
            continue
        else:
            boxes = recover_boxes(flat, img, det) if len(flat) else {}
        for i in g.index:
            if d.nlabel[i] in UPRIGHT:
                R.loc[i, "note"] = "stands above the road: flat-road geometry does not apply"
                continue
            if i in boxes:
                b, frac, _ = boxes[i]
                b = [v * sx for v in b]
                t, how = link_track(C, k, d.nlabel[i], b, step), "same box"
            else:
                # No box of this class fits: fall back to the run's own box of the same class and size nearby.
                px = d.defect_area_m2[i] * PROD_PPM ** 2 * sx * sx
                x = C[(C.nlabel == d.nlabel[i]) & ((C.frame - k).abs() <= step)]
                x = x.assign(err=np.log(px / x.pixels).abs()) if len(x) and px > 0 else x.iloc[:0]
                t = x.loc[x.err.idxmin()] if len(x) and x.err.min() < math.log(1.3) else None
                b, frac, how = None, (min(1.0, px / t.pixels) if t is not None else np.nan), "same class and size"
            if t is None:
                R.loc[i, "note"] = "the model did not find this box again in the video" if i in boxes else "no box of this class and size in the evidence image"
                continue
            R.loc[i, ["track_id", "frac", "how"]] = [int(t.track_id), frac, how]
            R.loc[i, ["x1", "y1", "x2", "y2"]] = [round(float(v), 1) for v in (b or (t.x1, t.y1, t.x2, t.y2))]

    Fi = F.set_index("track_id")
    new, conf, notes = [], [], []
    for i in d.index:
        t, n = R.track_id[i], R.note[i]
        f = Fi.loc[t] if pd.notna(t) and t in Fi.index else None
        if f is None or not np.isfinite(f.area_new):
            new.append(np.nan)
            conf.append("")
            notes.append(n or (f.note if f is not None else ""))
            continue
        new.append(f.area_new * R.frac[i])
        exact = R.how[i] == "same box" and norm_label(f.label) == d.nlabel[i]
        conf.append(f.confidence if exact else {"high": "medium"}.get(f.confidence, "low"))
        notes.append("; ".join(x for x in (f.note, "box partly off the road: scaled by the stored pixels" if R.frac[i] < .9 else "",
                                           f"the model called this spot {f.label} in the frames around it" if norm_label(f.label) != d.nlabel[i] else "",
                                           "matched by class and size only" if R.how[i] != "same box" else "") if x))
    d["new"], d["conf"], d["note"] = new, conf, notes
    # Two rows on one tracked defect are the same defect seen twice; the later ones say so.
    first, d["same_as"] = {}, ""
    for i in d.sort_values("id", key=lambda s: pd.to_numeric(s, errors="coerce")).index:
        t = R.track_id[i]
        if pd.notna(t):
            if t in first:
                d.loc[i, "same_as"] = d.id[first[t]]
            else:
                first[t] = i
    for p, q in same_place(C):           # one spot drawn under two class names
        if p in first and q in first:
            j = first[q]
            d.loc[j, "note"] = "; ".join(x for x in (d.note[j], f"same place as row {d.id[first[p]]}") if x)

    # Cost: only where this file's own cost is a fixed rate per m2 for the class, so it scales with the area.
    cost = pd.to_numeric(d["estimated_cost_inr"], errors="coerce") if "estimated_cost_inr" in d else pd.Series(np.nan, index=d.index)
    rate = cost / d.defect_area_m2
    linear = rate.groupby(d.nlabel).transform(lambda r: bool(r.notna().all() and r.min() > 0 and r.max() / r.min() < 1.01)).astype(bool)
    used = sorted({int(t) for t in R.track_id.dropna()})
    sheets = defect_images(out, C, F[F.track_id.isin(used)]) if used else {}

    res = rows.copy()
    at = list(res.columns).index("defect_area_m2") + 1
    res.insert(at, "defect_area_m2_new", np.where(d.new.notna(), d.new.round(4), d.defect_area_m2))
    res.insert(at + 1, "area_changed", np.where(d.new.notna(), "yes", "no"))
    ratio = d.new / d.defect_area_m2
    num = lambda s, k: pd.to_numeric(R.track_id.map(Fi[s]), errors="coerce").round(k)
    extra = dict(new_over_old=ratio.round(3), estimated_cost_inr_new=np.where(linear & d.new.notna(), (d.new * rate).round(2), cost),
                 confidence=d.conf, length_m=num("length_m", 2), width_m=num("width_m", 2), distance_m=num("distance_m", 2),
                 views_used=R.track_id.map(Fi.views_used), spread=num("spread", 3), same_defect_as_id=d.same_as, note=d.note,
                 video_frame=R.video_frame, frame_match=R.frame_match, track_id=R.track_id, box_x1=R.x1, box_y1=R.y1, box_x2=R.x2, box_y2=R.y2,
                 defect_image=R.track_id.map(lambda t: sheets.get(t, "") if pd.notna(t) else ""))
    for k, v in extra.items():
        res[k] = v
    big = d.new.notna() & ((ratio > 8) | (ratio < .1))
    res.loc[big, "confidence"] = "low"
    res.loc[big, "note"] = (res.note[big] + "; ").str.lstrip("; ") + "large change, check by eye"
    if a.job_id or ("job_id" in rows and rows.job_id.nunique() == 1):
        path = out / f"post_construction_output_{a.job_id or rows.job_id.iloc[0]}_new_area.csv"
    else:
        path = out / f"{Path(a.rows).stem}_new_area.csv"
    res.to_csv(path, index=False)

    ok = d.new.notna()
    S = dict(rows=len(d), updated=int(ok.sum()), confidence=res.confidence[ok].value_counts().to_dict(),
             kept_old=res.note[~ok].replace("", "no usable view").value_counts().to_dict(), same_defect_rows=int((d.same_as != "").sum()),
             by_class={k: dict(rows=len(g), updated=int(g.new.notna().sum()), old_m2=round(float(g.defect_area_m2.sum()), 2),
                               new_m2=round(float(np.where(g.new.notna(), g.new, g.defect_area_m2).sum()), 2),
                               median_new_over_old=round(float((g.new / g.defect_area_m2).median()), 3) if g.new.notna().any() else None)
                       for k, g in d.groupby("label")},
             pieces=meta.get("pieces"), warnings=meta["warnings"], csv=path.name)
    (out / "rescore.json").write_text(json.dumps(S, indent=1, default=str))
    print(f"\n=== {S['updated']} of {S['rows']} rows got a new area; confidence {S['confidence']}")
    for k, v in S["kept_old"].items():
        print(f"kept the old area ({v} rows): {k}")
    print(pd.DataFrame(S["by_class"]).T.to_string())
    for p in meta.get("pieces") or []:
        print(f"video frames {p['start_frame']}-{p['end_frame']}: GPX segment {p['gpx_segment'] + 1} from {p['gpx_offset_s']} s, "
              f"speed match {p['speed_match']}, f*H {p['fh']:.0f}")
    for w in meta["warnings"]:
        print("WARNING:", w)
    print("wrote", path)
    return path


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
    s = sub.add_parser("rescore", help="new area for the rows of an existing post_construction_output CSV")
    s.add_argument("--rows", required=True, help="post_construction_output CSV (needs id, label, frame_url, defect_area_m2)")
    s.add_argument("--job-id", help="only the rows of this job_id")
    s.add_argument("--video", help="the video the job processed: path, http(s) URL or s3:// URI")
    s.add_argument("--gpx", help="its GPX: path, http(s) URL or s3:// URI")
    s.add_argument("--height", type=float, help="lens height above the road in metres")
    s.add_argument("--model", type=lambda x: Path(x).expanduser(), help="the YOLO .pt model the job used")
    s.add_argument("--out", required=True, help="working folder; a second rescore on it reuses the analysed video")
    s.add_argument("--images", default="s3://photons-bucket-s3", help="where frame_url points: S3 or http(s) base, or a local folder (default %(default)s)")
    s.add_argument("--rerun", action="store_true", help="analyse the video again even if --out has it")
    s.add_argument("--fh", type=float, help="use this f*H instead of the GPX one")
    s.add_argument("--focal", type=float, help="known focal length in px at 1920 wide; replaces the GPX scale")
    s.add_argument("--gpx-offset", type=float, help="GPX seconds at video start (one recording only)")
    s.add_argument("--conf", type=float, default=.35, help="confidence to start a track")
    s.add_argument("--conf-low", type=float, default=.15, help="confidence to continue a track")
    s.add_argument("--imgsz", type=int, default=640, help="model input size (production uses 640)")
    s.add_argument("--device", default="auto", help="auto, cpu or cuda")
    s.add_argument("--aws-profile")
    a = p.parse_args(argv)
    if a.cmd == "run":
        if not .3 <= a.height <= 4:
            p.error("--height is in metres, for example 1.152")
        run(a)
    elif a.cmd == "rescore":
        return rescore(a)
    else:
        evaluate_dir(a.out, a.truth, a.outlines, (a.trim_x, a.trim_y))


if __name__ == "__main__":
    main()
