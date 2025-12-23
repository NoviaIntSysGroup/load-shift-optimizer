"""Utilities for converting PDF to raster images using pdftoppm or Ghostscript."""

import shutil
import subprocess
import sys
from pathlib import Path


def find_rasterizer() -> tuple[str, str]:
    """
    Detect available PDF rasterization tool.

    Searches for pdftoppm first (preferred), then falls back to Ghostscript
    variants including Windows-specific executables.

    Returns:
        Tuple of (tool_name, executable_path) where tool_name is either
        "pdftoppm" or "gs"

    Raises:
        SystemExit: If no suitable rasterizer is found
    """
    # Try pdftoppm first (preferred, usually faster)
    if pdftoppm_path := shutil.which("pdftoppm"):
        return ("pdftoppm", pdftoppm_path)

    # Fall back to Ghostscript (check Windows variants too)
    for gs_variant in ("gs", "gswin64c", "gswin32c", "mgs"):
        if gs_path := shutil.which(gs_variant):
            return ("gs", gs_path)

    sys.exit(
        "[ERROR] No PDF rasterizer found. Install 'poppler-utils' (for pdftoppm) "
        "or Ghostscript"
    )


def pdf_to_png(
    pdf_path: Path,
    output_path: Path,
    dpi: int = 600,
    tool: str | None = None,
    exe_path: str | None = None,
) -> Path:
    """
    Convert a single-page PDF to PNG.

    Args:
        pdf_path: Path to input PDF file
        output_path: Path for output PNG file (without extension for pdftoppm)
        dpi: Resolution in dots per inch
        tool: Tool to use ("pdftoppm" or "gs"). Auto-detected if None
        exe_path: Path to tool executable. Auto-detected if None

    Returns:
        Path to the generated PNG file

    Raises:
        SystemExit: If conversion fails or output file not created
    """
    # Auto-detect rasterizer if not specified
    if tool is None or exe_path is None:
        tool, exe_path = find_rasterizer()

    if tool == "pdftoppm":
        # pdftoppm adds "-1.png" suffix automatically for single page
        prefix = output_path.parent / output_path.stem
        subprocess.run(
            [exe_path, "-r", str(dpi), "-png", str(pdf_path), str(prefix)],
            check=True,
        )
        result_path = prefix.parent / f"{prefix.name}-1.png"

        # Rename to desired output path if different
        if result_path != output_path:
            result_path.rename(output_path)
            result_path = output_path

    else:  # Ghostscript
        subprocess.run(
            [
                exe_path,
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                f"-r{dpi}",
                "-sDEVICE=pngalpha",
                "-o",
                str(output_path),
                str(pdf_path),
            ],
            check=True,
        )
        result_path = output_path

    if not result_path.exists():
        sys.exit(f"[ERROR] Expected PNG not created: {result_path}")

    return result_path


def pdf_to_pngs_multipage(
    pdf_path: Path,
    output_prefix: Path,
    dpi: int = 600,
    tool: str | None = None,
    exe_path: str | None = None,
) -> list[Path]:
    """
    Convert a multi-page PDF to PNG files.

    Args:
        pdf_path: Path to input PDF file
        output_prefix: Prefix for output PNG files (tool will add page numbers)
        dpi: Resolution in dots per inch
        tool: Tool to use ("pdftoppm" or "gs"). Auto-detected if None
        exe_path: Path to tool executable. Auto-detected if None

    Returns:
        List of paths to generated PNG files

    Raises:
        SystemExit: If conversion fails
    """
    # Auto-detect rasterizer if not specified
    if tool is None or exe_path is None:
        tool, exe_path = find_rasterizer()

    if tool == "pdftoppm":
        subprocess.run(
            [exe_path, "-r", str(dpi), "-png", str(pdf_path), str(output_prefix)],
            check=True,
        )
        # pdftoppm creates files like "prefix-1.png", "prefix-2.png", etc.
        pattern = f"{output_prefix.name}-*.png"
        png_files = sorted(output_prefix.parent.glob(pattern))

    else:  # Ghostscript
        output_pattern = str(output_prefix) + "-%d.png"
        subprocess.run(
            [
                exe_path,
                "-dSAFER",
                "-dBATCH",
                "-dNOPAUSE",
                f"-r{dpi}",
                "-sDEVICE=pngalpha",
                "-o",
                output_pattern,
                str(pdf_path),
            ],
            check=True,
        )
        # Ghostscript creates files like "prefix-1.png", "prefix-2.png", etc.
        pattern = f"{output_prefix.name}-*.png"
        png_files = sorted(output_prefix.parent.glob(pattern))

    if not png_files:
        sys.exit(f"[ERROR] No PNG files created from {pdf_path}")

    return png_files
