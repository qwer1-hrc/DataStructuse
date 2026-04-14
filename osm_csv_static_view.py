#!/usr/bin/env python3
"""
读取 osm_export_csv 下的导出结果，做 OSM 静态可视化：
- 仅显示路网边、仓库、充电站
- 不显示车辆与任务
- 可选窗口大小，可切换场景
"""

from __future__ import annotations

import argparse
import csv
import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk
from typing import Dict, List, Optional, Tuple


DEFAULT_EXPORT_DIR = "osm_export_csv"
SIZE_PRESETS: Dict[str, Tuple[int, int]] = {
    "small": (900, 620),
    "medium": (1200, 820),
    "large": (1500, 980),
}


@dataclass
class StaticScenario:
    name: str
    nodes: Dict[int, Tuple[float, float]]
    edges: List[Tuple[int, int, float]]  # (u, v, congest_base)
    depot: Optional[Tuple[int, float, float]]
    chargers: List[Tuple[int, int, float, float, int]]  # (sid, node, lon, lat, slots)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _detect_scenarios(export_dir: str) -> List[str]:
    names: List[str] = []
    for fn in os.listdir(export_dir):
        if fn.endswith("_map_nodes.csv"):
            names.append(fn[: -len("_map_nodes.csv")])
    return sorted(names)


def load_scenario(export_dir: str, scenario: str) -> StaticScenario:
    p_nodes = os.path.join(export_dir, f"{scenario}_map_nodes.csv")
    p_edges = os.path.join(export_dir, f"{scenario}_map_edges.csv")
    p_sites = os.path.join(export_dir, f"{scenario}_sites.csv")
    missing = [p for p in (p_nodes, p_edges, p_sites) if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError("缺少 CSV 文件: " + ", ".join(missing))

    nodes: Dict[int, Tuple[float, float]] = {}
    for row in _read_csv_rows(p_nodes):
        n = int(row["node"])
        lon = float(row["lon"])
        lat = float(row["lat"])
        nodes[n] = (lon, lat)

    edges: List[Tuple[int, int, float]] = []
    for row in _read_csv_rows(p_edges):
        u = int(row["u"])
        v = int(row["v"])
        congest = float(row.get("congest_base", "0") or 0.0)
        edges.append((u, v, congest))

    depot: Optional[Tuple[int, float, float]] = None
    chargers: List[Tuple[int, int, float, float, int]] = []
    for row in _read_csv_rows(p_sites):
        site_type = row.get("site_type", "").strip().lower()
        sid = int(row.get("sid", "-1") or -1)
        node = int(row["node"])
        lon = float(row["lon"])
        lat = float(row["lat"])
        slots_raw = row.get("slots", "").strip()
        slots = int(slots_raw) if slots_raw else 0
        if site_type == "depot":
            depot = (node, lon, lat)
        elif site_type == "charger":
            chargers.append((sid, node, lon, lat, slots))

    return StaticScenario(
        name=scenario,
        nodes=nodes,
        edges=edges,
        depot=depot,
        chargers=chargers,
    )


def _lerp_rgb(c0: Tuple[int, int, int], c1: Tuple[int, int, int], t: float) -> str:
    t = max(0.0, min(1.0, t))
    r = int(c0[0] + (c1[0] - c0[0]) * t + 0.5)
    g = int(c0[1] + (c1[1] - c0[1]) * t + 0.5)
    b = int(c0[2] + (c1[2] - c0[2]) * t + 0.5)
    return f"#{r:02x}{g:02x}{b:02x}"


class OSMStaticCSVApp:
    def __init__(self, root: tk.Tk, export_dir: str, scenario_names: List[str], initial: str) -> None:
        self.root = root
        self.export_dir = export_dir
        self.scenario_names = scenario_names
        self._colors = {
            "bg": "#16161e",
            "depot": "#e0af68",
            "charger": "#73daca",
            "text": "#a9b1d6",
            "panel": "#1a1b26",
        }

        top = ttk.Frame(root, padding=6)
        top.pack(fill=tk.X)
        ttk.Label(top, text="场景").pack(side=tk.LEFT, padx=(0, 4))
        self.combo = ttk.Combobox(top, state="readonly", values=scenario_names, width=16)
        self.combo.pack(side=tk.LEFT, padx=4)
        self.combo.set(initial)
        self.combo.bind("<<ComboboxSelected>>", self._on_select)
        self.lbl = ttk.Label(top, text="")
        self.lbl.pack(side=tk.LEFT, padx=12)

        self.canvas = tk.Canvas(root, bg=self._colors["bg"], highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.scenario = load_scenario(self.export_dir, initial)
        self.root.bind("<Configure>", self._on_resize)
        self._draw()

    def _on_resize(self, _evt=None) -> None:
        self._draw()

    def _on_select(self, _evt=None) -> None:
        name = self.combo.get().strip()
        if not name:
            return
        self.scenario = load_scenario(self.export_dir, name)
        self._draw()

    def _compute_bounds(self) -> Tuple[float, float, float, float]:
        lonlat = list(self.scenario.nodes.values())
        lons = [x for x, _ in lonlat]
        lats = [y for _, y in lonlat]
        w, e = min(lons), max(lons)
        s, n = min(lats), max(lats)
        dx = (e - w) * 0.06 + 1e-9
        dy = (n - s) * 0.06 + 1e-9
        return w - dx, s - dy, e + dx, n + dy

    def _project(
        self, lon: float, lat: float, b: Tuple[float, float, float, float], ox: float, oy: float, pw: float, ph: float
    ) -> Tuple[float, float]:
        w0, s0, e0, n0 = b
        x = ox + (lon - w0) / (e0 - w0) * pw
        y = oy + (n0 - lat) / (n0 - s0) * ph
        return x, y

    def _draw(self) -> None:
        self.canvas.delete("all")
        W = max(640, int(self.canvas.winfo_width()))
        H = max(480, int(self.canvas.winfo_height()))
        pad = 12
        panel_h = 54
        pw = W - 2 * pad
        ph = H - panel_h - 2 * pad
        ox, oy = pad, pad

        b = self._compute_bounds()
        n_nodes = len(self.scenario.nodes)
        n_edges = len(self.scenario.edges)
        n_ch = len(self.scenario.chargers)
        self.lbl.config(text=f"节点 {n_nodes}  边 {n_edges}  充电站 {n_ch}")

        for u, v, congest in self.scenario.edges:
            p0 = self.scenario.nodes.get(u)
            p1 = self.scenario.nodes.get(v)
            if p0 is None or p1 is None:
                continue
            x0, y0 = self._project(p0[0], p0[1], b, ox, oy, pw, ph)
            x1, y1 = self._project(p1[0], p1[1], b, ox, oy, pw, ph)
            col = _lerp_rgb((0x3B, 0x42, 0x61), (0xF7, 0x66, 0x6E), congest)
            self.canvas.create_line(x0, y0, x1, y1, fill=col, width=1)

        if self.scenario.depot is not None:
            _node, lon, lat = self.scenario.depot
            x, y = self._project(lon, lat, b, ox, oy, pw, ph)
            self.canvas.create_rectangle(
                x - 8, y - 8, x + 8, y + 8, fill=self._colors["depot"], outline="#1a1b26", width=2
            )

        for sid, _node, lon, lat, _slots in self.scenario.chargers:
            x, y = self._project(lon, lat, b, ox, oy, pw, ph)
            r = 6
            self.canvas.create_polygon(
                x, y - r, x + r, y, x, y + r, x - r, y, fill=self._colors["charger"], outline="#1a1f2e"
            )
            if sid < 10:
                self.canvas.create_text(x + 9, y - 9, text=str(sid), fill="#7aa2f7", font=("Microsoft YaHei UI", 8))

        ly = H - panel_h + 2
        self.canvas.create_rectangle(0, ly, W, H, fill=self._colors["panel"], outline="#292e42")
        self.canvas.create_text(
            14,
            ly + 12,
            text="■ 仓库   ◆ 充电站   路网颜色：拥挤度低->高",
            fill=self._colors["text"],
            anchor=tk.W,
            font=("Microsoft YaHei UI", 9),
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="静态查看 osm_export_csv（地图/仓库/充电站）")
    ap.add_argument("--dir", default=DEFAULT_EXPORT_DIR, help="CSV 导出目录（默认 osm_export_csv）")
    ap.add_argument("--scenario", default=None, help="初始场景名（例如 OSM_SMALL）")
    ap.add_argument(
        "--size",
        choices=sorted(SIZE_PRESETS.keys()),
        default="medium",
        help="窗口大小预设：small/medium/large",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not os.path.isdir(args.dir):
        print(f"目录不存在: {os.path.abspath(args.dir)}")
        return 1
    names = _detect_scenarios(args.dir)
    if not names:
        print(f"未在目录中发现 *_map_nodes.csv: {os.path.abspath(args.dir)}")
        return 1
    initial = args.scenario if args.scenario in names else names[0]
    w, h = SIZE_PRESETS[args.size]
    root = tk.Tk()
    root.title("OSM 导出 CSV 静态可视化")
    root.geometry(f"{w}x{h}")
    OSMStaticCSVApp(root, args.dir, names, initial)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

