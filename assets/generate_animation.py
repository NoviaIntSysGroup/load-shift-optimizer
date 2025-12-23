#!/usr/bin/env python3
"""
Animate TikZ by sweeping the value of \newcommand{\controlHoursStart}{...}.
Workflow: LaTeX -> PDF -> PNG (pdftoppm or gs) -> GIF (Pillow).
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

from pdf_utils import find_rasterizer, pdf_to_png
from PIL import Image

PATTERN = r'(\\newcommand\s*\{\\controlHoursStart\}\s*\{)(.*?)(\})'


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def substitute_control_hours_start(tex: str, value: str) -> str:
    if re.search(PATTERN, tex):
        return re.sub(PATTERN, r'\g<1>' + str(value) + r'\g<3>', tex)
    sys.exit("[ERROR] Couldn't find \\newcommand{\\controlHoursStart}{...} in template")

def make_gif(pngs: list[Path], out_gif: Path, fps: float, loop: int) -> None:
    if not pngs:
        sys.exit("[ERROR] No PNGs to animate")
    dur = max(1, int(1000 / fps))
    frames = [Image.open(p).convert("RGBA") for p in pngs]
    frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                   duration=dur, loop=loop, disposal=2)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("template", type=Path)
    ap.add_argument(
        "--values",
        required=True,
        help="Comma-separated values for controlHoursStart",
    )
    ap.add_argument("--outdir", type=Path, default=Path("build"))
    ap.add_argument("--gif", type=Path, default=Path("animation.gif"))
    ap.add_argument("--pdflatex", default="pdflatex")
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--fps", type=float, default=1)
    ap.add_argument("--loop", type=int, default=0)
    args = ap.parse_args()

    if not shutil.which(args.pdflatex):
        sys.exit(f"[ERROR] '{args.pdflatex}' not found")

    raster_kind, raster_exec = find_rasterizer()
    print(f"[info] Using {raster_kind} at {raster_exec}")

    tex_src = args.template.read_text(encoding="utf-8")
    values = [v.strip() for v in args.values.split(",") if v.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)

    pngs: list[Path] = []
    for i, val in enumerate(values):
        tex = substitute_control_hours_start(tex_src, val)
        tex_path = args.outdir / f"frame_{i:04d}.tex"
        pdf_path = args.outdir / f"frame_{i:04d}.pdf"
        prefix  = args.outdir / f"frame_{i:04d}"
        tex_path.write_text(tex, encoding="utf-8")

        run([args.pdflatex, "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(args.outdir), str(tex_path)])

        # Convert PDF to PNG and store the path
        png_path = prefix.parent / f"{prefix.name}-1.png"
        pdf_to_png(
            pdf_path,
            png_path,
            dpi=args.dpi,
            tool=raster_kind,
            exe_path=raster_exec,
        )
        pngs.append(png_path)

    make_gif(pngs, args.gif, fps=args.fps, loop=args.loop)
    print(f"Done. Frames: {len(pngs)}  GIF: {args.gif.resolve()}")

if __name__ == "__main__":
    main()
