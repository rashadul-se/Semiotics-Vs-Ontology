"""
discourse_app.py
================
Streamlit app for RST Discourse Analysis with:
  - Redis caching   (graceful fallback to in-memory if Redis unavailable)
  - SQLite storage  (persistent job history + results)
  - PDF / TXT file upload
  - Salience Tree + Ontology Table figures
  - CSV + PNG download buttons

Usage:
    pip install streamlit redis pypdf pandas numpy matplotlib
    streamlit run discourse_app.py
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import pickle
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════
# 0.  CONSTANTS & THEME
# ═══════════════════════════════════════════════════════════════════════

DB_PATH    = "discourse_cache.db"
CACHE_TTL  = 3600          # Redis TTL in seconds
MAX_CHARS  = 200_000       # hard limit on input text

BG      = "#0F1117"
PANEL   = "#1A1D27"
FG      = "#E8E8E8"
SUB     = "#AAAAAA"
GRID    = "#2A2D3A"

RELATION_COLOR: dict[str, str] = {
    "Statement":    "#4C72B0",
    "Contrast":     "#DD8452",
    "Cause":        "#55A868",
    "Elaboration":  "#C44E52",
    "Cause-Effect": "#8172B3",
    "Joint":        "#937860",
    "Sequence":     "#DA8BC3",
    "Summary":      "#8C8C8C",
    "Concession":   "#CCB974",
    "Condition":    "#64B5CD",
    "Restatement":  "#E377C2",
}

_STOPWORDS = set("""a an the and or but if in on at to of for with by from as is was are were
be been being have has had do does did will would could should may might
shall can this that these those it its i we you he she they them their
our your his her my all also just not no nor so yet both either neither
each every any more most other such same than then there here when where
which who whom what how much many very too quite rather""".split())

_DISCOURSE_MARKERS: dict[str, tuple[str, str]] = {
    "however":        ("Contrast",      "satellite"),
    "but":            ("Contrast",      "satellite"),
    "although":       ("Concession",    "satellite"),
    "though":         ("Concession",    "satellite"),
    "despite":        ("Concession",    "satellite"),
    "nevertheless":   ("Contrast",      "satellite"),
    "nonetheless":    ("Contrast",      "satellite"),
    "yet":            ("Contrast",      "satellite"),
    "therefore":      ("Cause-Effect",  "nucleus"),
    "thus":           ("Cause-Effect",  "nucleus"),
    "hence":          ("Cause-Effect",  "nucleus"),
    "consequently":   ("Cause-Effect",  "nucleus"),
    "as a result":    ("Cause-Effect",  "nucleus"),
    "because":        ("Cause",         "satellite"),
    "since":          ("Cause",         "satellite"),
    "for example":    ("Elaboration",   "satellite"),
    "for instance":   ("Elaboration",   "satellite"),
    "such as":        ("Elaboration",   "satellite"),
    "specifically":   ("Elaboration",   "satellite"),
    "in addition":    ("Joint",         "nucleus"),
    "furthermore":    ("Joint",         "nucleus"),
    "moreover":       ("Joint",         "nucleus"),
    "first":          ("Sequence",      "nucleus"),
    "second":         ("Sequence",      "nucleus"),
    "finally":        ("Sequence",      "nucleus"),
    "in conclusion":  ("Summary",       "nucleus"),
    "in summary":     ("Summary",       "nucleus"),
    "to summarize":   ("Summary",       "nucleus"),
    "that is":        ("Restatement",   "satellite"),
    "in other words": ("Restatement",   "satellite"),
    "if":             ("Condition",     "satellite"),
    "unless":         ("Condition",     "satellite"),
}


# ═══════════════════════════════════════════════════════════════════════
# 1.  CACHE LAYER  (Redis with dict fallback)
# ═══════════════════════════════════════════════════════════════════════

class CacheManager:
    """
    Thin cache abstraction.
    Tries Redis first; falls back to an in-process dict so the app
    works without a Redis server installed.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = CACHE_TTL):
        self._ttl    = ttl
        self._client = None
        self._local: dict[str, Any] = {}
        try:
            import redis  # type: ignore
            r = redis.Redis(host=host, port=port, socket_connect_timeout=1)
            r.ping()
            self._client = r
        except Exception:
            pass  # Redis unavailable → use _local dict

    @property
    def backend(self) -> str:
        return "Redis" if self._client else "In-Memory"

    # ── public interface ─────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        if self._client:
            raw = self._client.get(key)
            return pickle.loads(raw) if raw else None
        return self._local.get(key)

    def set(self, key: str, value: Any) -> None:
        if self._client:
            self._client.setex(key, self._ttl, pickle.dumps(value))
        else:
            self._local[key] = value

    def delete(self, key: str) -> None:
        if self._client:
            self._client.delete(key)
        else:
            self._local.pop(key, None)

    def flush(self) -> None:
        if self._client:
            self._client.flushdb()
        else:
            self._local.clear()

    @staticmethod
    def make_key(text: str) -> str:
        return "discourse:" + hashlib.sha256(text.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════════════
# 2.  DATABASE LAYER  (SQLite)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class JobRecord:
    job_id:      str
    filename:    str
    char_count:  int
    sent_count:  int
    created_at:  float
    status:      str          # pending | done | error
    error_msg:   str = ""


class DatabaseManager:
    """All SQLite I/O for job history and result persistence."""

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        self._bootstrap()

    # ── internal ────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _bootstrap(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id      TEXT PRIMARY KEY,
                    filename    TEXT NOT NULL,
                    char_count  INTEGER,
                    sent_count  INTEGER,
                    created_at  REAL NOT NULL,
                    status      TEXT NOT NULL DEFAULT 'pending',
                    error_msg   TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS spans (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id      TEXT NOT NULL REFERENCES jobs(job_id),
                    sent_index  INTEGER,
                    relation    TEXT,
                    role        TEXT,
                    salience    REAL,
                    preview     TEXT,
                    full_text   TEXT
                );

                CREATE TABLE IF NOT EXISTS tfidf_terms (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id      TEXT NOT NULL REFERENCES jobs(job_id),
                    term        TEXT,
                    score       REAL,
                    rank        INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_spans_job  ON spans(job_id);
                CREATE INDEX IF NOT EXISTS idx_tfidf_job  ON tfidf_terms(job_id);
            """)

    # ── public interface ─────────────────────────────────────────────

    def upsert_job(self, record: JobRecord) -> None:
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO jobs (job_id, filename, char_count, sent_count,
                                  created_at, status, error_msg)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    status=excluded.status,
                    error_msg=excluded.error_msg,
                    sent_count=excluded.sent_count
            """, (record.job_id, record.filename, record.char_count,
                  record.sent_count, record.created_at,
                  record.status, record.error_msg))

    def save_spans(self, job_id: str, df: pd.DataFrame) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM spans WHERE job_id=?", (job_id,))
            rows = [
                (job_id, int(r["index"]), r["relation"], r["role"],
                 float(r["salience"]), r["preview"], r["full"])
                for _, r in df.iterrows()
            ]
            conn.executemany(
                "INSERT INTO spans (job_id,sent_index,relation,role,"
                "salience,preview,full_text) VALUES (?,?,?,?,?,?,?)", rows)

    def save_tfidf(self, job_id: str, tfidf: dict[str, float],
                   top_terms: list[str]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM tfidf_terms WHERE job_id=?", (job_id,))
            rows = [(job_id, term, tfidf[term], i + 1)
                    for i, term in enumerate(top_terms)]
            conn.executemany(
                "INSERT INTO tfidf_terms (job_id,term,score,rank) VALUES (?,?,?,?)",
                rows)

    def load_spans(self, job_id: str) -> Optional[pd.DataFrame]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT sent_index,relation,role,salience,preview,full_text "
                "FROM spans WHERE job_id=? ORDER BY sent_index", (job_id,)
            ).fetchall()
        if not rows:
            return None
        df = pd.DataFrame([dict(r) for r in rows])
        df.rename(columns={"sent_index": "index", "full_text": "full"}, inplace=True)
        return df

    def load_tfidf(self, job_id: str) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT term, score, rank FROM tfidf_terms WHERE job_id=? ORDER BY rank",
                (job_id,)
            ).fetchall()
        if not rows:
            return {}
        return {r["term"]: r["score"] for r in rows}

    def list_jobs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_job(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM spans       WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM tfidf_terms WHERE job_id=?", (job_id,))
            conn.execute("DELETE FROM jobs        WHERE job_id=?", (job_id,))


# ═══════════════════════════════════════════════════════════════════════
# 3.  TEXT PROCESSOR
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class PrepResult:
    sentences:   list[str]
    tfidf:       dict[str, float]
    token_count: int
    vocab_size:  int
    top_terms:   list[str]


class TextProcessor:
    """Stateless NLP pre-processing pipeline."""

    _SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"])')

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        return [s.strip() for s in TextProcessor._SENT_RE.split(text) if s.strip()]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    @classmethod
    def preprocess(cls, text: str) -> PrepResult:
        sentences  = cls.split_sentences(text)
        all_tokens = cls.tokenize(text)
        filtered   = [t for t in all_tokens if t not in _STOPWORDS]
        tf         = Counter(filtered)
        total      = sum(tf.values()) or 1
        tf_norm    = {w: c / total for w, c in tf.most_common(200)}
        N          = len(sentences) or 1
        sent_toks  = [set(cls.tokenize(s)) - _STOPWORDS for s in sentences]
        df_cnt     = Counter(w for st in sent_toks for w in st)
        idf        = {w: math.log(N / (df_cnt[w] + 1)) + 1 for w in tf_norm}
        tfidf      = {w: tf_norm[w] * idf.get(w, 1) for w in tf_norm}
        top_terms  = sorted(tfidf, key=tfidf.get, reverse=True)[:30]  # type: ignore[arg-type]
        return PrepResult(
            sentences   = sentences,
            tfidf       = tfidf,
            token_count = len(all_tokens),
            vocab_size  = len(set(filtered)),
            top_terms   = top_terms,
        )

    @staticmethod
    def extract_pdf(file_bytes: bytes) -> str:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(io.BytesIO(file_bytes))
        pages  = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)

    @staticmethod
    def read_upload(uploaded_file) -> str:
        name = uploaded_file.name.lower()
        raw  = uploaded_file.read()
        if name.endswith(".pdf"):
            return TextProcessor.extract_pdf(raw)
        return raw.decode("utf-8", errors="replace")


# ═══════════════════════════════════════════════════════════════════════
# 4.  DISCOURSE ANALYSER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AnalysisResult:
    df:   pd.DataFrame
    prep: PrepResult


class DiscourseAnalyzer:
    """Builds the RST discourse dataframe from preprocessed text."""

    @staticmethod
    def _label_sentence(sent: str) -> tuple[str, str]:
        lo = sent.lower()
        for marker, (relation, role) in _DISCOURSE_MARKERS.items():
            if re.search(r'\b' + re.escape(marker) + r'\b', lo):
                return relation, role
        return "Statement", "nucleus"

    @classmethod
    def analyze(cls, text: str) -> AnalysisResult:
        prep = TextProcessor.preprocess(text)
        rows = []
        for i, sent in enumerate(prep.sentences):
            relation, role = cls._label_sentence(sent)
            ws  = [w for w in TextProcessor.tokenize(sent) if w not in _STOPWORDS]
            sal = float(np.mean([prep.tfidf.get(w, 0) for w in ws])) if ws else 0.0
            rows.append({
                "index":    i + 1,
                "relation": relation,
                "role":     role,
                "salience": round(sal, 5),
                "preview":  (sent[:65] + "…") if len(sent) > 65 else sent,
                "full":     sent,
            })
        return AnalysisResult(df=pd.DataFrame(rows), prep=prep)


# ═══════════════════════════════════════════════════════════════════════
# 5.  VISUALISER
# ═══════════════════════════════════════════════════════════════════════

class Visualizer:
    """Produces both publication-quality matplotlib figures."""

    # ── shared helpers ───────────────────────────────────────────────

    @staticmethod
    def _alpha(sal: float, min_sal: float, rng: float) -> float:
        return 0.45 + 0.50 * (sal - min_sal) / rng

    @staticmethod
    def _snorm(sal: float, min_sal: float, rng: float) -> float:
        return (sal - min_sal) / rng

    # ── Figure 1 : Salience Tree ─────────────────────────────────────

    # Layout constants
    N_PER_ROW   = 12    # max nucleus nodes per wrapped row
    ROOT_W      = 0.13
    ROOT_H      = 0.052
    NODE_W      = 0.068  # narrow — text is rotated 90°
    NODE_H      = 0.155  # tall
    SAT_W       = 0.115
    SAT_H       = 0.095
    ROW_GAP     = 0.07   # vertical gap between nucleus rows
    SAT_GAP     = 0.14   # gap between last nucleus row and satellite row

    @classmethod
    def salience_tree(cls, df: pd.DataFrame) -> plt.Figure:
        n_nuc    = int((df["role"] == "nucleus").sum())
        n_rows   = max(1, math.ceil(n_nuc / cls.N_PER_ROW))
        # Adaptive figure height: base + extra per extra nucleus row
        fig_h    = 10.5 + max(0, n_rows - 1) * 3.5
        fig, ax  = plt.subplots(figsize=(20, fig_h))
        fig.patch.set_facecolor(BG)
        cls._draw_tree(df, ax)
        fig.suptitle("RST Discourse Salience Tree",
                     fontsize=15, fontweight="bold", color=FG, y=0.995)
        return fig

    @classmethod
    def _draw_tree(cls, df: pd.DataFrame, ax: plt.Axes) -> None:
        # ── coordinate system ────────────────────────────────────────
        nuclei     = df[df["role"] == "nucleus"].reset_index(drop=True)
        satellites = df[df["role"] == "satellite"].reset_index(drop=True)
        n_nuc      = len(nuclei)
        n_rows     = max(1, math.ceil(n_nuc / cls.N_PER_ROW))

        # Y positions (top-down: root → nuc rows → sat row)
        ROOT_Y   = 0.955
        ROOT_X   = 0.50
        # First nucleus row starts just below root
        NUC_Y0   = ROOT_Y - cls.ROOT_H / 2 - 0.06 - cls.NODE_H / 2
        SAT_Y    = NUC_Y0 - (n_rows - 1) * (cls.NODE_H + cls.ROW_GAP) \
                   - cls.NODE_H / 2 - cls.SAT_GAP - cls.SAT_H / 2

        # Rescale axes to fit everything
        total_h  = ROOT_Y + 0.05 - (SAT_Y - cls.SAT_H / 2 - 0.06)
        ax.set_xlim(0, 1)
        ax.set_ylim(SAT_Y - cls.SAT_H / 2 - 0.07, ROOT_Y + 0.05)
        ax.axis("off")
        ax.set_facecolor(BG)

        min_sal = df["salience"].min()
        max_sal = df["salience"].max()
        rng     = max_sal - min_sal + 1e-9

        # ── helpers ──────────────────────────────────────────────────

        def draw_root(cx, cy):
            ax.add_patch(FancyBboxPatch(
                (cx - cls.ROOT_W / 2, cy - cls.ROOT_H / 2),
                cls.ROOT_W, cls.ROOT_H,
                boxstyle="round,pad=0.012", linewidth=1.8,
                edgecolor="#AAAAAA", facecolor=(*to_rgb("#AAAAAA"), 0.18), zorder=3))
            ax.text(cx, cy, "DISCOURSE ROOT", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=FG, zorder=4)

        def draw_nucleus(cx, cy, idx, relation, salience):
            col = RELATION_COLOR.get(relation, "#888888")
            a   = cls._alpha(salience, min_sal, rng)
            ax.add_patch(FancyBboxPatch(
                (cx - cls.NODE_W / 2, cy - cls.NODE_H / 2),
                cls.NODE_W, cls.NODE_H,
                boxstyle="round,pad=0.010", linewidth=1.4,
                edgecolor=col, facecolor=(*to_rgb(col), a), zorder=3))
            # Salience bar at bottom of box (horizontal fill)
            bw = cls.NODE_W * 0.82 * cls._snorm(salience, min_sal, rng)
            ax.add_patch(FancyBboxPatch(
                (cx - bw / 2, cy - cls.NODE_H / 2 + 0.005), bw, 0.009,
                boxstyle="round,pad=0.001", facecolor=col, alpha=0.95, zorder=4))
            # Rotated text — index on top half, relation name on bottom half
            ax.text(cx, cy + cls.NODE_H * 0.18, f"[{idx}]",
                    ha="center", va="center", rotation=90,
                    fontsize=7.8, fontweight="bold", color=FG, zorder=4)
            ax.text(cx, cy - cls.NODE_H * 0.18, relation,
                    ha="center", va="center", rotation=90,
                    fontsize=6.5, color="#DDDDDD", zorder=4, style="italic")
            return col

        def draw_satellite(cx, cy, idx, relation, salience, preview):
            col = RELATION_COLOR.get(relation, "#888888")
            a   = cls._alpha(salience, min_sal, rng) * 0.85
            ax.add_patch(FancyBboxPatch(
                (cx - cls.SAT_W / 2, cy - cls.SAT_H / 2),
                cls.SAT_W, cls.SAT_H,
                boxstyle="round,pad=0.010", linewidth=1.4,
                edgecolor=col, facecolor=(*to_rgb(col), a), zorder=3))
            bw = cls.SAT_W * 0.82 * cls._snorm(salience, min_sal, rng)
            ax.add_patch(FancyBboxPatch(
                (cx - bw / 2, cy - cls.SAT_H / 2 + 0.005), bw, 0.008,
                boxstyle="round,pad=0.001", facecolor=col, alpha=0.92, zorder=4))
            ax.text(cx, cy + cls.SAT_H * 0.15,
                    f"[{idx}] {relation}", ha="center", va="center",
                    fontsize=7.2, fontweight="bold", color=FG, zorder=4)
            ax.text(cx, cy - cls.SAT_H * 0.20,
                    preview[:40] + ("…" if len(preview) > 40 else ""),
                    ha="center", va="center",
                    fontsize=5.5, color="#CCCCCC", zorder=4, style="italic")
            return col

        def arrow(x1, y1, x2, y2, col, sal, dash=False, thin=False):
            lw  = (0.7 if thin else 1.0) + 2.0 * cls._snorm(sal, min_sal, rng)
            sty = "dashed" if dash else "solid"
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(
                            arrowstyle="-|>", color=col, lw=lw,
                            linestyle=sty, mutation_scale=9,
                            connectionstyle="arc3,rad=0.0"), zorder=2)

        def row_label(y, lbl):
            ax.text(0.003, y, lbl, va="center", ha="left",
                    fontsize=7, color="#555566", fontweight="bold")

        # ── draw root ────────────────────────────────────────────────
        draw_root(ROOT_X, ROOT_Y)

        # ── draw nucleus rows ─────────────────────────────────────────
        nuc_pos: dict[int, tuple[float, float]] = {}

        for row_i in range(n_rows):
            start     = row_i * cls.N_PER_ROW
            batch     = nuclei.iloc[start: start + cls.N_PER_ROW]
            n_in_row  = len(batch)
            cy        = NUC_Y0 - row_i * (cls.NODE_H + cls.ROW_GAP)
            xs        = np.linspace(0.04, 0.96, n_in_row) if n_in_row > 1 else [ROOT_X]

            # Draw a subtle bus connector line for this row
            if n_in_row > 1:
                ax.plot([xs[0], xs[-1]], [cy + cls.NODE_H / 2 + 0.008] * 2,
                        color="#333344", lw=0.8, zorder=1)

            if row_i == 0:
                row_label(cy, "NUCLEUS")
            else:
                row_label(cy, f"NUC · row {row_i + 1}")

            for j, (_, row) in enumerate(batch.iterrows()):
                cx  = xs[j]
                col = draw_nucleus(cx, cy, row["index"], row["relation"], row["salience"])
                nuc_pos[row["index"]] = (cx, cy)

                # Arrow: root → first row nodes directly; later rows via bus
                if row_i == 0:
                    arrow(ROOT_X, ROOT_Y - cls.ROOT_H / 2,
                          cx, cy + cls.NODE_H / 2,
                          col, row["salience"], thin=(n_nuc > 8))
                else:
                    # Connect from bottom of previous row's nearest node
                    prev_row_start = (row_i - 1) * cls.N_PER_ROW
                    prev_batch     = nuclei.iloc[prev_row_start:
                                                  prev_row_start + cls.N_PER_ROW]
                    nearest_prev   = min(prev_batch.iterrows(),
                                        key=lambda x: abs(x[1]["index"] - row["index"]))
                    px, py_prev = nuc_pos.get(nearest_prev[1]["index"],
                                              (ROOT_X, NUC_Y0))
                    arrow(px, py_prev - cls.NODE_H / 2,
                          cx, cy  + cls.NODE_H / 2,
                          col, row["salience"], thin=True)

        # ── draw satellite row ────────────────────────────────────────
        nuc_idx = list(nuc_pos.keys())
        groups: dict[int, list] = defaultdict(list)
        for _, row in satellites.iterrows():
            parent = min(nuc_idx, key=lambda n: abs(n - row["index"]))
            groups[parent].append(row)

        if groups:
            row_label(SAT_Y, "SATELLITE")

        for parent, sat_rows in groups.items():
            px, py = nuc_pos[parent]
            n_s    = len(sat_rows)
            offs   = np.linspace(-0.10 * (n_s - 1) / 2,
                                  0.10 * (n_s - 1) / 2, n_s)
            for off, row in zip(offs, sat_rows):
                sx  = np.clip(px + off, 0.06, 0.94)
                col = draw_satellite(sx, SAT_Y, row["index"], row["relation"],
                                     row["salience"], row["preview"])
                arrow(px, py - cls.NODE_H / 2,
                      sx, SAT_Y + cls.SAT_H / 2,
                      col, row["salience"], dash=True)

        # ── gradient colour bar ───────────────────────────────────────
        bar_y   = SAT_Y - cls.SAT_H / 2 - 0.045
        bar_lbl = SAT_Y - cls.SAT_H / 2 - 0.020
        for xi in range(100):
            ax.barh(bar_y, 0.0026, left=0.69 + xi * 0.0026, height=0.018,
                    color=plt.cm.plasma(xi / 99), alpha=0.9)
        mn, mx = df["salience"].min(), df["salience"].max()
        ax.text(0.69, bar_lbl, f"Salience  {mn:.4f}", fontsize=6.5, color=SUB, ha="left")
        ax.text(0.95, bar_lbl, f"{mx:.4f}",           fontsize=6.5, color=SUB, ha="right")

        # ── legend ────────────────────────────────────────────────────
        patches = [mpatches.Patch(facecolor=RELATION_COLOR[r], label=r, alpha=0.85)
                   for r in df["relation"].unique() if r in RELATION_COLOR]
        ax.legend(handles=patches,
                  loc="lower left",
                  fontsize=7, framealpha=0.15, labelcolor="white",
                  facecolor="#1E1E2E", edgecolor="#444",
                  title="Rhetorical Relation", title_fontsize=7.5,
                  ncol=2,
                  bbox_to_anchor=(0.0, bar_y - 0.02),
                  bbox_transform=ax.transData)

        ax.set_title(
            "RST Discourse Salience Tree\n"
            "Solid = nucleus  ·  Dashed = satellite  ·  Bar width = salience  "
            "·  Text rotated 90°",
            fontsize=11, fontweight="bold", color=FG, pad=10)

    # ── Figure 2 : Ontology Tables ───────────────────────────────────

    @classmethod
    def ontology_tables(cls, df: pd.DataFrame, prep: PrepResult) -> plt.Figure:
        fig = plt.figure(figsize=(21, 15))
        fig.patch.set_facecolor(BG)
        fig.suptitle("Discourse Ontology Tables",
                     fontsize=15, fontweight="bold", color=FG, y=0.99)
        gs = GridSpec(1, 3, figure=fig,
                      left=0.02, right=0.98,
                      top=0.93,  bottom=0.04,
                      wspace=0.06)
        cls._draw_ontology(df, prep, fig, gs)
        return fig

    @classmethod
    def _table_style(cls, ax: plt.Axes, df_t: pd.DataFrame,
                     col_widths: list[float], row_colors: list[str],
                     header_color: str = "#252836") -> None:
        ax.axis("off")
        n_rows = len(df_t)
        col_x  = np.cumsum([0] + col_widths[:-1])
        total_w = sum(col_widths)
        row_h   = 1.0 / (n_rows + 1)

        def cell(x, y, w, h, text, bg, fg=FG, bold=False, align="left", fs=7.5):
            ax.add_patch(plt.Rectangle((x, y), w, h,
                                       facecolor=bg, edgecolor=GRID,
                                       linewidth=0.6,
                                       transform=ax.transAxes, clip_on=False))
            tx = x + (w / 2 if align == "center" else 0.012)
            ax.text(tx, y + h / 2, str(text),
                    ha=align, va="center", fontsize=fs,
                    color=fg, fontweight="bold" if bold else "normal",
                    transform=ax.transAxes, clip_on=False)

        for j, (col, cw, cx) in enumerate(zip(df_t.columns, col_widths, col_x)):
            cell(cx / total_w, 1 - row_h, cw / total_w, row_h,
                 col, header_color, fg="#FFD700", bold=True, align="center")

        for i, (_, row) in enumerate(df_t.iterrows()):
            y  = 1 - row_h * (i + 2)
            bg = row_colors[i % len(row_colors)]
            for j, (val, cw, cx) in enumerate(zip(row.values, col_widths, col_x)):
                align = "center" if j > 0 else "left"
                cell(cx / total_w, y, cw / total_w, row_h, val, bg, align=align)

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    @classmethod
    def _draw_ontology(cls, df: pd.DataFrame, prep: PrepResult,
                       fig: plt.Figure, gs: GridSpec) -> None:
        ROW_A = ["#1C1F2E", "#20253A"]
        ROW_B = ["#1C2A1C", "#20351C"]
        ROW_C = ["#1C1F2E", "#1E2130"]

        # Panel A — Term Ontology
        ax_a = fig.add_subplot(gs[0, 0])
        ax_a.set_facecolor(PANEL)
        ax_a.set_title("A · Term Ontology  (TF-IDF)", color=FG,
                       fontsize=9.5, fontweight="bold", pad=6)
        top20   = prep.top_terms[:20]
        max_t   = max((prep.tfidf[w] for w in top20), default=1)
        term_df = pd.DataFrame({
            "Term":      top20,
            "TF-IDF":    [f"{prep.tfidf[w]:.5f}" for w in top20],
            "Rel score": [f"{prep.tfidf[w] / max_t * 100:.1f}%" for w in top20],
            "Rank":      [str(i + 1) for i in range(len(top20))],
        })
        cls._table_style(ax_a, term_df, col_widths=[5, 3, 3, 2], row_colors=ROW_A)

        # Panel B — Relation Ontology
        ax_b = fig.add_subplot(gs[0, 1])
        ax_b.set_facecolor(PANEL)
        ax_b.set_title("B · Relation Ontology", color=FG,
                       fontsize=9.5, fontweight="bold", pad=6)
        rel_stats = (
            df.groupby(["relation", "role"])
              .agg(Count=("salience", "count"), Avg_sal=("salience", "mean"))
              .reset_index()
              .sort_values("Count", ascending=False)
        )
        rel_stats["Avg_sal"] = rel_stats["Avg_sal"].map("{:.5f}".format)
        rel_df = rel_stats.rename(columns={
            "relation": "Relation", "role": "Role",
            "Count": "Count", "Avg_sal": "Avg Salience"})
        cls._table_style(ax_b, rel_df, col_widths=[5, 4, 2, 4], row_colors=ROW_B)

        # Panel C — Span Ontology
        ax_c = fig.add_subplot(gs[0, 2])
        ax_c.set_facecolor(PANEL)
        ax_c.set_title("C · Span Ontology  (all sentences)", color=FG,
                       fontsize=9.5, fontweight="bold", pad=6)
        span_df = df[["index", "relation", "role", "salience", "preview"]].copy()
        span_df["salience"] = span_df["salience"].map("{:.5f}".format)
        span_df.columns     = ["#", "Relation", "Role", "Salience", "Preview"]
        cls._table_style(ax_c, span_df, col_widths=[1, 4, 3, 3, 10], row_colors=ROW_C)


# ═══════════════════════════════════════════════════════════════════════
# 5b.  STATISTICS ANALYSER
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StatResult:
    descriptive:   pd.DataFrame   # per-relation descriptive stats
    role_means:    pd.DataFrame   # nucleus vs satellite means
    normality:     pd.DataFrame   # Shapiro-Wilk per relation
    kruskal:       dict           # Kruskal-Wallis H across relations
    mannwhitney:   dict           # Mann-Whitney U nucleus vs satellite
    anova:         dict           # one-way ANOVA + eta-squared
    spearman:      dict           # rank correlation index↔salience
    pivot_mean:    pd.DataFrame   # relation × role mean salience (heatmap)
    pairwise:      pd.DataFrame   # pairwise Mann-Whitney between relations
    group_means:   pd.DataFrame   # clean summary table for display


class StatisticsAnalyzer:
    """
    Full statistical battery on salience scores.

    Tests run:
      · Descriptive stats per relation (mean, median, std, var, skew, kurtosis, CV)
      · Shapiro-Wilk normality test per relation (W, p-value, normal?)
      · One-way ANOVA across relations + η² effect size
      · Kruskal-Wallis H-test across relations (non-parametric alternative)
      · Mann-Whitney U: nucleus vs satellite
      · Spearman ρ: sentence index vs salience (order effect)
      · Pairwise Mann-Whitney between every pair of relations
    """

    @classmethod
    def analyze(cls, df: pd.DataFrame) -> StatResult:
        from scipy import stats as sp

        sal      = df["salience"].values.astype(float)
        groups   = {r: g["salience"].values.astype(float)
                    for r, g in df.groupby("relation") if len(g) >= 2}
        role_nuc = df[df["role"] == "nucleus"]["salience"].astype(float).values
        role_sat = df[df["role"] == "satellite"]["salience"].astype(float).values

        # ── descriptive stats ────────────────────────────────────────
        rows = []
        for rel, vals in groups.items():
            rows.append({
                "Relation":  rel,
                "N":         len(vals),
                "Mean":      np.mean(vals),
                "Median":    np.median(vals),
                "Std":       np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                "Variance":  np.var(vals,  ddof=1) if len(vals) > 1 else 0.0,
                "Min":       np.min(vals),
                "Max":       np.max(vals),
                "Skewness":  float(sp.skew(vals))     if len(vals) > 2 else np.nan,
                "Kurtosis":  float(sp.kurtosis(vals)) if len(vals) > 2 else np.nan,
                "CV (%)":    (np.std(vals, ddof=1) / np.mean(vals) * 100)
                              if np.mean(vals) != 0 and len(vals) > 1 else np.nan,
            })
        descriptive = pd.DataFrame(rows).sort_values("Mean", ascending=False)

        # ── role means ────────────────────────────────────────────────
        role_rows = []
        for role_lbl, vals in [("nucleus", role_nuc), ("satellite", role_sat)]:
            if len(vals):
                role_rows.append({
                    "Role":     role_lbl,
                    "N":        len(vals),
                    "Mean":     np.mean(vals),
                    "Median":   np.median(vals),
                    "Std":      np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                    "Min":      np.min(vals),
                    "Max":      np.max(vals),
                })
        role_means = pd.DataFrame(role_rows)

        # ── Shapiro-Wilk normality test (need ≥ 3 samples) ───────────
        norm_rows = []
        for rel, vals in groups.items():
            if len(vals) >= 3:
                w, p = sp.shapiro(vals)
                norm_rows.append({
                    "Relation": rel,
                    "W":        round(w, 5),
                    "p-value":  round(p, 5),
                    "Normal?":  "Yes" if p > 0.05 else "No",
                })
        normality = pd.DataFrame(norm_rows).sort_values("p-value")

        # ── one-way ANOVA ─────────────────────────────────────────────
        g_lists = [v for v in groups.values() if len(v) >= 1]
        anova   = {"f": np.nan, "p": np.nan, "eta2": np.nan, "note": ""}
        if len(g_lists) >= 2:
            try:
                f_stat, p_anova = sp.f_oneway(*g_lists)
                # η² = SS_between / SS_total
                grand_mean = sal.mean()
                ss_total   = np.sum((sal - grand_mean) ** 2)
                ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2
                                 for v in g_lists)
                eta2       = ss_between / ss_total if ss_total > 0 else np.nan
                anova      = {"f": round(f_stat, 4),
                              "p": round(p_anova, 6),
                              "eta2": round(eta2, 4),
                              "note": ("Significant (p<0.05)" if p_anova < 0.05
                                       else "Not significant")}
            except Exception as e:
                anova["note"] = str(e)

        # ── Kruskal-Wallis ────────────────────────────────────────────
        kruskal = {"h": np.nan, "p": np.nan, "note": ""}
        if len(g_lists) >= 2:
            try:
                h_stat, p_kw = sp.kruskal(*g_lists)
                kruskal = {"h": round(h_stat, 4),
                           "p": round(p_kw, 6),
                           "note": ("Significant (p<0.05)" if p_kw < 0.05
                                    else "Not significant")}
            except Exception as e:
                kruskal["note"] = str(e)

        # ── Mann-Whitney U: nucleus vs satellite ──────────────────────
        mannwhitney = {"u": np.nan, "p": np.nan, "r_effect": np.nan, "note": ""}
        if len(role_nuc) >= 1 and len(role_sat) >= 1:
            try:
                u_stat, p_mw = sp.mannwhitneyu(role_nuc, role_sat,
                                                alternative="two-sided")
                n1, n2       = len(role_nuc), len(role_sat)
                r_effect     = 1 - 2 * u_stat / (n1 * n2)
                mannwhitney  = {
                    "u": round(u_stat, 2),
                    "p": round(p_mw, 6),
                    "r_effect": round(r_effect, 4),
                    "note": ("Nucleus ≠ Satellite (p<0.05)"
                             if p_mw < 0.05 else "No significant difference"),
                }
            except Exception as e:
                mannwhitney["note"] = str(e)

        # ── Spearman rank correlation ─────────────────────────────────
        spearman = {"rho": np.nan, "p": np.nan, "note": ""}
        if len(sal) >= 3:
            try:
                rho, p_sp = sp.spearmanr(df["index"].values, sal)
                spearman  = {
                    "rho": round(rho, 4),
                    "p":   round(p_sp, 6),
                    "note": ("Significant order effect (p<0.05)"
                             if p_sp < 0.05 else "No significant order effect"),
                }
            except Exception as e:
                spearman["note"] = str(e)

        # ── pivot: mean salience by relation × role ───────────────────
        pivot_mean = (
            df.pivot_table(values="salience", index="relation",
                           columns="role", aggfunc="mean")
              .fillna(0)
              .round(5)
        )

        # ── pairwise Mann-Whitney between relations ───────────────────
        rel_names  = list(groups.keys())
        pw_rows    = []
        for i in range(len(rel_names)):
            for j in range(i + 1, len(rel_names)):
                a, b = groups[rel_names[i]], groups[rel_names[j]]
                if len(a) >= 1 and len(b) >= 1:
                    try:
                        u, p = sp.mannwhitneyu(a, b, alternative="two-sided")
                        pw_rows.append({
                            "Group A":   rel_names[i],
                            "Group B":   rel_names[j],
                            "U":         round(u, 2),
                            "p-value":   round(p, 5),
                            "Sig (p<.05)": "★" if p < 0.05 else "",
                        })
                    except Exception:
                        pass
        pairwise = pd.DataFrame(pw_rows).sort_values("p-value") if pw_rows else pd.DataFrame()

        # ── clean summary table ───────────────────────────────────────
        group_means = descriptive[["Relation", "N", "Mean", "Median", "Std", "CV (%)"]].copy()
        group_means = group_means.round(6)

        return StatResult(
            descriptive  = descriptive,
            role_means   = role_means,
            normality    = normality,
            kruskal      = kruskal,
            mannwhitney  = mannwhitney,
            anova        = anova,
            spearman     = spearman,
            pivot_mean   = pivot_mean,
            pairwise     = pairwise,
            group_means  = group_means,
        )


# ═══════════════════════════════════════════════════════════════════════
# 5c.  STATISTICS VISUALISER
# ═══════════════════════════════════════════════════════════════════════

class StatVisualizer:
    """Four-panel publication figure: heatmap, distributions, box, correlation."""

    # ── colour helpers ───────────────────────────────────────────────

    @staticmethod
    def _ax_style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=FG, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=SUB, fontsize=8)
        ax.set_ylabel(ylabel, color=SUB, fontsize=8)
        ax.tick_params(colors=SUB, labelsize=7.5)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)

    # ── Figure 3: full stats dashboard ──────────────────────────────

    @classmethod
    def stats_figure(cls, df: pd.DataFrame, sr: StatResult) -> plt.Figure:
        fig = plt.figure(figsize=(22, 18))
        fig.patch.set_facecolor(BG)
        fig.suptitle("Salience · Statistical Validation Dashboard",
                     fontsize=15, fontweight="bold", color=FG, y=0.995)

        gs = GridSpec(3, 3, figure=fig,
                      left=0.06, right=0.97,
                      top=0.96, bottom=0.05,
                      hspace=0.45, wspace=0.38)

        cls._panel_heatmap(fig.add_subplot(gs[0, :2]), sr)
        cls._panel_role_bar(fig.add_subplot(gs[0, 2]),  sr)
        cls._panel_violin(fig.add_subplot(gs[1, :]),    df)
        cls._panel_kde(fig.add_subplot(gs[2, 0]),       df)
        cls._panel_scatter(fig.add_subplot(gs[2, 1]),   df, sr)
        cls._panel_pvalue_bar(fig.add_subplot(gs[2, 2]),sr)

        return fig

    # ── Panel 1: Heatmap (relation × role → mean salience) ──────────

    @classmethod
    def _panel_heatmap(cls, ax: plt.Axes, sr: StatResult) -> None:
        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.ticker as mticker

        pivot = sr.pivot_mean
        if pivot.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", color=SUB)
            return

        cmap = LinearSegmentedColormap.from_list(
            "sal", ["#0F1117", "#1A3A5C", "#2E6DA4", "#FFD700", "#FF6B35"])

        data = pivot.values
        im   = ax.imshow(data, cmap=cmap, aspect="auto", interpolation="nearest")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, color=FG, fontsize=9, fontweight="bold")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, color=FG, fontsize=8)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                txt = f"{val:.4f}" if val > 0 else "—"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="#000000" if val > data.max() * 0.6 else FG)

        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.ax.tick_params(colors=SUB, labelsize=7)
        cbar.set_label("Mean Salience", color=SUB, fontsize=8)

        cls._ax_style(ax, "Heatmap · Mean Salience  (Relation × Role)",
                      "Role", "Rhetorical Relation")
        ax.tick_params(colors=FG)

    # ── Panel 2: Role mean bar chart ─────────────────────────────────

    @classmethod
    def _panel_role_bar(cls, ax: plt.Axes, sr: StatResult) -> None:
        rm = sr.role_means
        if rm.empty:
            return
        roles  = rm["Role"].tolist()
        means  = rm["Mean"].tolist()
        stds   = rm["Std"].tolist()
        colors = ["#4C72B0", "#DD8452"]

        bars = ax.bar(roles, means, color=colors[:len(roles)],
                      yerr=stds, capsize=6, error_kw={"ecolor": "#AAAAAA", "lw": 1.5},
                      width=0.5, edgecolor=GRID, linewidth=0.8, alpha=0.88)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05,
                    f"μ={mean:.4f}\nσ={std:.4f}",
                    ha="center", va="bottom", fontsize=7.5, color=FG)

        cls._ax_style(ax, "Mean Salience by Role\n(± 1 SD)",
                      "Role", "Mean Salience")
        ax.set_facecolor(PANEL)

        # Add Mann-Whitney annotation
        mw   = sr.mannwhitney
        note = f"Mann-Whitney U={mw['u']}\np={mw['p']}  r={mw['r_effect']}"
        ax.text(0.98, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=7, color="#FFD700",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#252836",
                          edgecolor="#444", alpha=0.9))

    # ── Panel 3: Violin per relation ─────────────────────────────────

    @classmethod
    def _panel_violin(cls, ax: plt.Axes, df: pd.DataFrame) -> None:
        rel_order = (df.groupby("relation")["salience"]
                       .mean().sort_values(ascending=False).index.tolist())
        groups    = [df[df["relation"] == r]["salience"].values for r in rel_order]
        colors    = [RELATION_COLOR.get(r, "#888888") for r in rel_order]

        parts = ax.violinplot(groups, positions=range(len(rel_order)),
                              showmedians=True, showextrema=True, widths=0.7)

        for i, (pc, col) in enumerate(zip(parts["bodies"], colors)):
            pc.set_facecolor(col); pc.set_alpha(0.65); pc.set_edgecolor(col)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color(FG); parts[key].set_linewidth(1.2)

        # Overlay jittered points
        for i, (grp, col) in enumerate(zip(groups, colors)):
            jitter = np.random.uniform(-0.12, 0.12, size=len(grp))
            ax.scatter(i + jitter, grp, color=col, alpha=0.5, s=18, zorder=3)

        ax.set_xticks(range(len(rel_order)))
        ax.set_xticklabels(rel_order, rotation=30, ha="right", color=FG, fontsize=8)
        cls._ax_style(ax, "Violin · Salience Distribution by Rhetorical Relation",
                      "Relation", "Salience Score")

    # ── Panel 4: KDE + histogram ─────────────────────────────────────

    @classmethod
    def _panel_kde(cls, ax: plt.Axes, df: pd.DataFrame) -> None:
        from scipy.stats import gaussian_kde

        for role, col in [("nucleus", "#4C72B0"), ("satellite", "#DD8452")]:
            vals = df[df["role"] == role]["salience"].astype(float).values
            if len(vals) < 2:
                continue
            ax.hist(vals, bins=12, color=col, alpha=0.28,
                    density=True, edgecolor=GRID, linewidth=0.5)
            xs = np.linspace(vals.min() - 1e-6, vals.max() + 1e-6, 300)
            try:
                kde = gaussian_kde(vals, bw_method="silverman")
                ax.plot(xs, kde(xs), color=col, lw=2, label=f"{role} (n={len(vals)})")
            except Exception:
                pass

        ax.axvline(df["salience"].mean(), color="#FFD700", lw=1.5,
                   linestyle="--", label=f"Global mean={df['salience'].mean():.4f}")
        ax.legend(fontsize=7, facecolor="#1E1E2E", edgecolor="#444",
                  labelcolor=FG, framealpha=0.8)
        cls._ax_style(ax, "KDE + Histogram · Salience by Role",
                      "Salience Score", "Density")

    # ── Panel 5: Scatter index vs salience + Spearman ────────────────

    @classmethod
    def _panel_scatter(cls, ax: plt.Axes, df: pd.DataFrame,
                       sr: StatResult) -> None:
        colors = [RELATION_COLOR.get(r, "#888") for r in df["relation"]]
        ax.scatter(df["index"], df["salience"], c=colors, s=55,
                   alpha=0.80, edgecolors=GRID, linewidths=0.4, zorder=3)

        # Trend line (OLS)
        x = df["index"].values.astype(float)
        y = df["salience"].values.astype(float)
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs   = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, m * xs + b, color="#FFD700", lw=1.8,
                    linestyle="--", label=f"OLS slope={m:.5f}")

        sp   = sr.spearman
        note = f"Spearman ρ={sp['rho']}\np={sp['p']}"
        ax.text(0.97, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="#FFD700",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#252836",
                          edgecolor="#444", alpha=0.9))
        ax.legend(fontsize=7, facecolor="#1E1E2E", edgecolor="#444",
                  labelcolor=FG, framealpha=0.8)
        cls._ax_style(ax, "Sentence Index vs Salience\n(Spearman rank correlation)",
                      "Sentence Index", "Salience Score")

    # ── Panel 6: p-value bar chart (pairwise MW) ─────────────────────

    @classmethod
    def _panel_pvalue_bar(cls, ax: plt.Axes, sr: StatResult) -> None:
        pw = sr.pairwise
        if pw.empty:
            ax.text(0.5, 0.5, "Need ≥2 groups\nfor pairwise test",
                    ha="center", va="center", color=SUB, fontsize=9,
                    transform=ax.transAxes)
            cls._ax_style(ax, "Pairwise p-values (Mann-Whitney)")
            return

        labels = [f"{r['Group A'][:6]}↔{r['Group B'][:6]}"
                  for _, r in pw.iterrows()]
        pvals  = pw["p-value"].values
        colors = ["#C44E52" if p < 0.05 else "#4C72B0" for p in pvals]

        y_pos  = range(len(labels))
        ax.barh(list(y_pos), pvals, color=colors, alpha=0.82,
                edgecolor=GRID, linewidth=0.5)
        ax.axvline(0.05, color="#FFD700", lw=1.5, linestyle="--",
                   label="α = 0.05")
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, color=FG, fontsize=7)
        ax.legend(fontsize=7, facecolor="#1E1E2E", edgecolor="#444",
                  labelcolor=FG, framealpha=0.8)
        cls._ax_style(ax, "Pairwise p-values · Mann-Whitney U\n(red = significant)",
                      "p-value", "Relation pair")


# ═══════════════════════════════════════════════════════════════════════
# 6.  PIPELINE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

class DiscourseService:
    """
    Coordinates the full pipeline:
      upload → cache check → NLP → persist → visualise
    """

    def __init__(self, cache: CacheManager, db: DatabaseManager):
        self._cache = cache
        self._db    = db

    def process(self, text: str, filename: str) -> AnalysisResult:
        key = CacheManager.make_key(text)

        # 1. cache hit?
        cached = self._cache.get(key)
        if cached:
            return cached

        # 2. analyse
        result = DiscourseAnalyzer.analyze(text)
        job_id = key

        # 3. persist to SQLite
        record = JobRecord(
            job_id     = job_id,
            filename   = filename,
            char_count = len(text),
            sent_count = len(result.df),
            created_at = time.time(),
            status     = "done",
        )
        self._db.upsert_job(record)
        self._db.save_spans(job_id, result.df)
        self._db.save_tfidf(job_id, result.prep.tfidf, result.prep.top_terms)

        # 4. store in cache
        self._cache.set(key, result)
        return result

    def fig_to_png(self, fig: plt.Figure) -> bytes:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=180,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()

    def df_to_csv(self, df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode()


# ═══════════════════════════════════════════════════════════════════════
# 7.  STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    for k, v in [
        ("result",      None),
        ("filename",    ""),
        ("processing",  False),
    ]:
        if k not in st.session_state:
            st.session_state[k] = v


@st.cache_resource
def _get_service() -> DiscourseService:
    return DiscourseService(CacheManager(), DatabaseManager())


def _sidebar(service: DiscourseService) -> None:
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        cache_be = service._cache.backend
        icon     = "🟢" if cache_be == "Redis" else "🟡"
        st.markdown(f"**Cache backend:** {icon} {cache_be}")
        st.markdown("**Storage:** 🗄️ SQLite")

        st.divider()
        st.markdown("### 📜 Job History")
        jobs = service._db.list_jobs()
        if not jobs:
            st.caption("No jobs yet.")
        for j in jobs[:10]:
            ts  = time.strftime("%m/%d %H:%M", time.localtime(j["created_at"]))
            lbl = f"{ts}  ·  {j['filename']}  ({j['sent_count']} sents)"
            if st.button(lbl, key=f"job_{j['job_id']}", use_container_width=True):
                df_loaded   = service._db.load_spans(j["job_id"])
                tfidf_loaded= service._db.load_tfidf(j["job_id"])
                if df_loaded is not None:
                    # rebuild a minimal PrepResult
                    top = sorted(tfidf_loaded, key=tfidf_loaded.get, reverse=True)[:30]
                    prep = PrepResult(sentences=[], tfidf=tfidf_loaded,
                                     token_count=0, vocab_size=0, top_terms=top)
                    st.session_state["result"]   = AnalysisResult(df=df_loaded, prep=prep)
                    st.session_state["filename"] = j["filename"]
                    st.rerun()

        st.divider()
        if st.button("🗑️ Flush cache", use_container_width=True):
            service._cache.flush()
            st.success("Cache cleared.")


def _upload_section(service: DiscourseService) -> None:
    st.markdown("### 📂 Upload Document")
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "PDF or TXT / MD — up to 200 K characters",
            type=["pdf", "txt", "md"],
            label_visibility="collapsed",
        )

    with col2:
        use_demo = st.button("▶ Run Demo Text", use_container_width=True)

    demo_text = (
        "Enterprise Data Solutions Proposal\n\n"
        "This document outlines our approach to data modernization. However, legacy "
        "systems present significant challenges. Because many organizations rely on "
        "outdated infrastructure, migration must be handled carefully. For example, "
        "phased rollouts reduce risk substantially. Therefore, we recommend a "
        "three-stage plan. In addition, ongoing training ensures adoption. Finally, "
        "post-migration support guarantees long-term success. In summary, this "
        "proposal delivers measurable ROI within twelve months."
    )

    text, filename = "", ""

    if use_demo:
        text, filename = demo_text, "demo.txt"
    elif uploaded:
        with st.spinner("Reading file…"):
            text     = TextProcessor.read_upload(uploaded)
            filename = uploaded.name
        if len(text) > MAX_CHARS:
            st.warning(f"Truncating to {MAX_CHARS:,} characters.")
            text = text[:MAX_CHARS]

    if text:
        st.info(f"**{filename}** — {len(text):,} chars")
        if st.button("🔍 Analyse", type="primary", use_container_width=True):
            with st.spinner("Analysing discourse structure…"):
                try:
                    result = service.process(text, filename)
                    st.session_state["result"]   = result
                    st.session_state["filename"] = filename
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")


def _fmt_p(p) -> str:
    """Format p-value with significance stars."""
    if np.isnan(p):
        return "n/a"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{p:.5f} {stars}"


def _results_section(service: DiscourseService) -> None:
    result: AnalysisResult = st.session_state["result"]
    fname:  str            = st.session_state["filename"]
    df, prep               = result.df, result.prep

    # ── Metrics strip ────────────────────────────────────────────────
    sal   = df["salience"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Sentences",      len(df))
    c2.metric("Nuclei",         int((df["role"] == "nucleus").sum()))
    c3.metric("Satellites",     int((df["role"] == "satellite").sum()))
    c4.metric("Mean Salience",  f"{sal.mean():.5f}")
    c5.metric("Max Salience",   f"{sal.max():.5f}")
    c6.metric("Unique terms",   prep.vocab_size)

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "🌳 Salience Tree",
        "📊 Ontology Tables",
        "📈 Statistics & Heatmap",
        "📋 Raw Data",
    ])

    with t1:
        with st.spinner("Rendering tree…"):
            fig1 = Visualizer.salience_tree(df)
        st.pyplot(fig1, use_container_width=True)
        png1 = service.fig_to_png(fig1)
        st.download_button(
            "⬇ Download Tree (PNG)", data=png1,
            file_name=f"{Path(fname).stem}_salience_tree.png",
            mime="image/png", use_container_width=True,
        )

    with t2:
        with st.spinner("Rendering tables…"):
            fig2 = Visualizer.ontology_tables(df, prep)
        st.pyplot(fig2, use_container_width=True)
        png2 = service.fig_to_png(fig2)
        st.download_button(
            "⬇ Download Ontology (PNG)", data=png2,
            file_name=f"{Path(fname).stem}_ontology_tables.png",
            mime="image/png", use_container_width=True,
        )

    # ── Statistics tab ────────────────────────────────────────────────
    with t3:
        with st.spinner("Running statistical tests…"):
            sr = StatisticsAnalyzer.analyze(df)

        # ── Headline test results ─────────────────────────────────
        st.markdown("### 🔬 Hypothesis Tests")
        ca, cb, cc = st.columns(3)

        with ca:
            st.markdown("**One-way ANOVA**")
            st.markdown(f"F = `{sr.anova['f']}`")
            st.markdown(f"p = `{_fmt_p(sr.anova['p'])}`")
            st.markdown(f"η² = `{sr.anova['eta2']}`")
            st.caption(sr.anova["note"])

        with cb:
            st.markdown("**Kruskal-Wallis H** *(non-parametric)*")
            st.markdown(f"H = `{sr.kruskal['h']}`")
            st.markdown(f"p = `{_fmt_p(sr.kruskal['p'])}`")
            st.caption(sr.kruskal["note"])

        with cc:
            st.markdown("**Mann-Whitney U** *(nucleus vs satellite)*")
            st.markdown(f"U = `{sr.mannwhitney['u']}`")
            st.markdown(f"p = `{_fmt_p(sr.mannwhitney['p'])}`")
            st.markdown(f"r = `{sr.mannwhitney['r_effect']}`")
            st.caption(sr.mannwhitney["note"])

        st.divider()

        # Spearman row
        sp = sr.spearman
        st.markdown(
            f"**Spearman ρ** (sentence order → salience): "
            f"`ρ = {sp['rho']}`, `p = {_fmt_p(sp['p'])}`  — {sp['note']}"
        )

        st.divider()

        # ── Stats dashboard figure ────────────────────────────────
        st.markdown("### 📊 Validation Dashboard")
        with st.spinner("Rendering statistical figures…"):
            fig3 = StatVisualizer.stats_figure(df, sr)
        st.pyplot(fig3, use_container_width=True)
        png3 = service.fig_to_png(fig3)
        st.download_button(
            "⬇ Download Stats Dashboard (PNG)", data=png3,
            file_name=f"{Path(fname).stem}_stats_dashboard.png",
            mime="image/png", use_container_width=True,
        )

        st.divider()

        # ── Mean values tables ────────────────────────────────────
        st.markdown("### 📋 Mean Values from Tables")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("#### By Rhetorical Relation")
            disp_desc = sr.descriptive[[
                "Relation","N","Mean","Median","Std","Skewness","Kurtosis","CV (%)"
            ]].copy()
            for col in ["Mean","Median","Std","Skewness","Kurtosis","CV (%)"]:
                disp_desc[col] = disp_desc[col].apply(
                    lambda v: f"{v:.5f}" if pd.notna(v) else "—")
            st.dataframe(disp_desc, use_container_width=True, hide_index=True)
            csv_desc = service.df_to_csv(sr.descriptive)
            st.download_button("⬇ CSV", data=csv_desc,
                               file_name=f"{Path(fname).stem}_descriptive.csv",
                               mime="text/csv")

        with col_r:
            st.markdown("#### By Role (Nucleus / Satellite)")
            disp_role = sr.role_means.copy()
            for col in ["Mean","Median","Std","Min","Max"]:
                if col in disp_role.columns:
                    disp_role[col] = disp_role[col].apply(lambda v: f"{v:.5f}")
            st.dataframe(disp_role, use_container_width=True, hide_index=True)

        st.markdown("#### Normality Tests (Shapiro-Wilk per relation)")
        if not sr.normality.empty:
            st.dataframe(sr.normality, use_container_width=True, hide_index=True)
        else:
            st.caption("Need ≥ 3 samples per relation.")

        st.markdown("#### Pairwise Mann-Whitney U between Relations")
        if not sr.pairwise.empty:
            st.dataframe(sr.pairwise, use_container_width=True, hide_index=True)
            csv_pw = service.df_to_csv(sr.pairwise)
            st.download_button("⬇ Pairwise CSV", data=csv_pw,
                               file_name=f"{Path(fname).stem}_pairwise.csv",
                               mime="text/csv")
        else:
            st.caption("Need ≥ 2 groups with ≥ 1 sample each.")

        st.markdown("#### Heatmap Data (Relation × Role → Mean Salience)")
        st.dataframe(sr.pivot_mean.style.background_gradient(
            cmap="YlOrRd", axis=None), use_container_width=True)
        csv_pivot = service.df_to_csv(sr.pivot_mean.reset_index())
        st.download_button("⬇ Heatmap CSV", data=csv_pivot,
                           file_name=f"{Path(fname).stem}_heatmap.csv",
                           mime="text/csv")

    # ── Raw data tab ─────────────────────────────────────────────────
    with t4:
        st.markdown("#### Discourse Spans")
        display_df = df[["index", "relation", "role", "salience", "preview"]].copy()
        st.dataframe(display_df, use_container_width=True)
        csv_spans = service.df_to_csv(df.drop(columns=["preview"]))
        st.download_button(
            "⬇ Download Spans (CSV)", data=csv_spans,
            file_name=f"{Path(fname).stem}_discourse.csv",
            mime="text/csv", use_container_width=True,
        )

        st.markdown("#### TF-IDF Terms (top 30)")
        tfidf_df = pd.DataFrame({
            "Rank":  range(1, len(prep.top_terms) + 1),
            "Term":  prep.top_terms,
            "Score": [round(prep.tfidf[w], 6) for w in prep.top_terms],
        })
        st.dataframe(tfidf_df, use_container_width=True)
        csv_tfidf = service.df_to_csv(tfidf_df)
        st.download_button(
            "⬇ Download TF-IDF (CSV)", data=csv_tfidf,
            file_name=f"{Path(fname).stem}_tfidf.csv",
            mime="text/csv", use_container_width=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Discourse Analyser",
        page_icon="🌳",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Minimal dark-mode CSS tweaks
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0F1117; }
    [data-testid="stSidebar"]          { background-color: #1A1D27; }
    h1, h2, h3, h4 { color: #E8E8E8; }
    .stMetricValue  { font-size: 1.6rem; }
    </style>""", unsafe_allow_html=True)

    st.title("🌳 RST Discourse Analyser")
    st.caption("Upload a PDF or text file to generate a Salience Tree + Ontology Tables.")

    _init_state()
    service = _get_service()
    _sidebar(service)
    _upload_section(service)

    if st.session_state["result"] is not None:
        st.divider()
        _results_section(service)


if __name__ == "__main__":
    main()
