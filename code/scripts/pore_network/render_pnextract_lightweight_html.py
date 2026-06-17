#!/usr/bin/env python3
"""Export a lightweight interactive HTML view for large pnextract networks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_lightweight_payload(
    pores: pd.DataFrame,
    throats: pd.DataFrame,
    *,
    voxel_size_m: float,
    sphere_radius_scale: float = 1.0,
    tube_radius_scale: float = 1.0,
    min_tube_radius_vox: float = 0.35,
) -> dict:
    pores = pores.copy()
    coords_vox = pores[["pore_center_x_m", "pore_center_y_m", "pore_center_z_m"]].to_numpy(dtype=float) / voxel_size_m
    radii_vox = pores["pore_radius_m"].to_numpy(dtype=float) / voxel_size_m * sphere_radius_scale
    pores[["x_vox", "y_vox", "z_vox"]] = coords_vox
    pores["radius_vox"] = radii_vox

    pore_ids = pores["pore_id"].astype(int).to_list()
    id_to_index = {pore_id: index for index, pore_id in enumerate(pore_ids)}
    internal = throats[(throats["pore1_id"] > 0) & (throats["pore2_id"] > 0)].copy()

    edges: list[list[float]] = []
    for row in internal.itertuples(index=False):
        pore1_id = int(row.pore1_id)
        pore2_id = int(row.pore2_id)
        if pore1_id not in id_to_index or pore2_id not in id_to_index:
            continue
        radius_vox = max(float(row.throat_radius_m) / voxel_size_m * tube_radius_scale, min_tube_radius_vox)
        edges.append([id_to_index[pore1_id], id_to_index[pore2_id], round(radius_vox, 6)])

    mins = coords_vox.min(axis=0)
    maxs = coords_vox.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = float(np.max(maxs - mins))
    return {
        "renderer": "canvas-2d-lightweight",
        "pores": np.round(np.column_stack([coords_vox, radii_vox]), 6).tolist(),
        "edges": edges,
        "bounds": {
            "min": np.round(mins, 6).tolist(),
            "max": np.round(maxs, 6).tolist(),
            "center": np.round(center, 6).tolist(),
            "extent": extent,
        },
        "counts": {
            "pores": int(len(pores)),
            "throats_total": int(len(throats)),
            "throats_rendered": int(len(edges)),
        },
        "metadata": {
            "coordinate_units": "voxel",
            "voxel_size_m": float(voxel_size_m),
            "sphere_radius_scale": float(sphere_radius_scale),
            "tube_radius_scale": float(tube_radius_scale),
            "min_tube_radius_vox": float(min_tube_radius_vox),
            "note": (
                "Large-network browser preview. It keeps all pore nodes and internal throats, "
                "but renders them as projected canvas circles/lines instead of PyVista sphere/tube meshes."
            ),
        },
    }


def html_template(payload_json: str) -> str:
    return dedent(
        f"""\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>pnextract Lightweight Pore Network</title>
          <style>
            html, body {{
              margin: 0;
              height: 100%;
              overflow: hidden;
              background: #ffffff;
              color: #111827;
              font-family: Arial, Helvetica, sans-serif;
            }}
            #networkCanvas {{
              display: block;
              width: 100vw;
              height: 100vh;
              cursor: grab;
              background: #ffffff;
            }}
            #networkCanvas:active {{
              cursor: grabbing;
            }}
            .panel {{
              position: fixed;
              top: 14px;
              right: 14px;
              z-index: 10;
              padding: 10px 12px;
              border: 1px solid rgba(15, 23, 42, 0.14);
              border-radius: 8px;
              background: rgba(255, 255, 255, 0.9);
              box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
              font-size: 13px;
              line-height: 1.4;
              pointer-events: none;
            }}
            .panel strong {{
              display: block;
              margin-bottom: 2px;
            }}
            .toolbar {{
              position: fixed;
              left: 14px;
              bottom: 14px;
              z-index: 11;
              display: flex;
              gap: 6px;
              padding: 7px;
              border: 1px solid rgba(15, 23, 42, 0.14);
              border-radius: 8px;
              background: rgba(255, 255, 255, 0.9);
              box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
            }}
            button {{
              height: 30px;
              min-width: 32px;
              border: 1px solid rgba(15, 23, 42, 0.22);
              border-radius: 6px;
              background: #ffffff;
              color: #111827;
              font-size: 13px;
              cursor: pointer;
            }}
          </style>
        </head>
        <body>
          <canvas id="networkCanvas"></canvas>
          <div class="panel">
            <strong>Corrected pnextract V2 network</strong>
            <div id="counts"></div>
            <div id="viewState"></div>
          </div>
          <div class="toolbar">
            <button id="zoomIn" type="button" title="Zoom in">+</button>
            <button id="zoomOut" type="button" title="Zoom out">-</button>
            <button id="resetView" type="button" title="Reset view">Reset</button>
          </div>
          <script>
          const NETWORK_PAYLOAD = {payload_json};
          const canvas = document.getElementById('networkCanvas');
          const ctx = canvas.getContext('2d');
          const pores = NETWORK_PAYLOAD.pores;
          const edges = NETWORK_PAYLOAD.edges;
          const center = NETWORK_PAYLOAD.bounds.center;
          const extent = Math.max(NETWORK_PAYLOAD.bounds.extent, 1);
          const counts = NETWORK_PAYLOAD.counts;
          let rx = -0.55;
          let ry = 0.95;
          let zoom = 1.0;
          let dragging = false;
          let lastX = 0;
          let lastY = 0;
          const projected = new Array(pores.length);
          document.getElementById('counts').textContent =
            counts.pores.toLocaleString() + ' pores / ' +
            counts.throats_rendered.toLocaleString() + ' internal throats';

          function resize() {{
            const ratio = window.devicePixelRatio || 1;
            canvas.width = Math.floor(window.innerWidth * ratio);
            canvas.height = Math.floor(window.innerHeight * ratio);
            canvas.style.width = window.innerWidth + 'px';
            canvas.style.height = window.innerHeight + 'px';
            ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
            draw();
          }}

          function projectPoint(p) {{
            let x = (p[0] - center[0]) / extent;
            let y = (p[1] - center[1]) / extent;
            let z = (p[2] - center[2]) / extent;
            const cy = Math.cos(ry);
            const sy = Math.sin(ry);
            const cx = Math.cos(rx);
            const sx = Math.sin(rx);
            const x1 = x * cy + z * sy;
            const z1 = -x * sy + z * cy;
            const y1 = y * cx - z1 * sx;
            const z2 = y * sx + z1 * cx;
            const scale = Math.min(window.innerWidth, window.innerHeight) * 1.42 * zoom;
            return {{
              x: window.innerWidth / 2 + x1 * scale,
              y: window.innerHeight / 2 - y1 * scale,
              z: z2,
              r: Math.max(1.2, Math.min(7.0, p[3] * scale / extent * 0.18))
            }};
          }}

          function draw() {{
            ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
            ctx.lineCap = 'round';
            ctx.strokeStyle = 'rgba(0, 76, 255, 0.28)';
            for (let i = 0; i < pores.length; i += 1) projected[i] = projectPoint(pores[i]);
            for (const edge of edges) {{
              const a = projected[edge[0]];
              const b = projected[edge[1]];
              if (!a || !b) continue;
              ctx.lineWidth = Math.max(0.45, Math.min(2.2, edge[2] * 0.28 * zoom));
              ctx.beginPath();
              ctx.moveTo(a.x, a.y);
              ctx.lineTo(b.x, b.y);
              ctx.stroke();
            }}
            const order = projected.map((p, i) => [p.z, i]).sort((a, b) => a[0] - b[0]);
            for (const item of order) {{
              const p = projected[item[1]];
              const shade = Math.max(0, Math.min(1, 0.58 + p.z * 0.5));
              ctx.fillStyle = 'rgba(' + Math.round(210 + shade * 45) + ', 0, 0, 0.72)';
              ctx.beginPath();
              ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
              ctx.fill();
            }}
            document.getElementById('viewState').textContent = 'Zoom ' + zoom.toFixed(2);
          }}

          canvas.addEventListener('mousedown', event => {{
            dragging = true;
            lastX = event.clientX;
            lastY = event.clientY;
          }});
          window.addEventListener('mouseup', () => {{ dragging = false; }});
          window.addEventListener('mousemove', event => {{
            if (!dragging) return;
            ry += (event.clientX - lastX) * 0.008;
            rx += (event.clientY - lastY) * 0.008;
            lastX = event.clientX;
            lastY = event.clientY;
            draw();
          }});
          canvas.addEventListener('wheel', event => {{
            event.preventDefault();
            zoom *= event.deltaY < 0 ? 1.12 : 1 / 1.12;
            zoom = Math.max(0.25, Math.min(6.0, zoom));
            draw();
          }}, {{ passive: false }});
          document.getElementById('zoomIn').addEventListener('click', () => {{ zoom = Math.min(6.0, zoom * 1.18); draw(); }});
          document.getElementById('zoomOut').addEventListener('click', () => {{ zoom = Math.max(0.25, zoom / 1.18); draw(); }});
          document.getElementById('resetView').addEventListener('click', () => {{ rx = -0.55; ry = 0.95; zoom = 1.0; draw(); }});
          window.addEventListener('resize', resize);
          resize();
          </script>
        </body>
        </html>
        """
    )


def write_lightweight_html(
    payload: dict,
    *,
    html_out: Path,
    metadata_out: Path,
    source_pores_csv: Path,
    source_throats_csv: Path,
) -> dict:
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    html_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.write_text(html_template(payload_json), encoding="utf-8")

    metadata = {
        "output_html": str(html_out),
        "renderer": payload["renderer"],
        "input_pores_csv": str(source_pores_csv),
        "input_throats_csv": str(source_throats_csv),
        **payload["metadata"],
        **payload["counts"],
        "html_size_bytes": int(html_out.stat().st_size),
        "note": "Openable full-topology preview; not a PyVista sphere/tube mesh export.",
    }
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pores", required=True)
    parser.add_argument("--throats", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--voxel-size-m", type=float, default=2.8e-6)
    parser.add_argument("--sphere-radius-scale", type=float, default=1.0)
    parser.add_argument("--tube-radius-scale", type=float, default=1.0)
    parser.add_argument("--min-tube-radius-vox", type=float, default=0.35)
    args = parser.parse_args()

    pores_path = Path(args.pores)
    throats_path = Path(args.throats)
    payload = build_lightweight_payload(
        pd.read_csv(pores_path),
        pd.read_csv(throats_path),
        voxel_size_m=args.voxel_size_m,
        sphere_radius_scale=args.sphere_radius_scale,
        tube_radius_scale=args.tube_radius_scale,
        min_tube_radius_vox=args.min_tube_radius_vox,
    )
    metadata = write_lightweight_html(
        payload,
        html_out=Path(args.out),
        metadata_out=Path(args.metadata_out),
        source_pores_csv=pores_path,
        source_throats_csv=throats_path,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
