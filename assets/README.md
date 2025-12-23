# Assets Directory

LaTeX source files and Python scripts for generating documentation images and animations.

## Contents

**LaTeX Sources:**
- `logo.tex`, `transfer_matrix.tex`, `moving_horizon.tex` - Static diagrams
- `transfer_matrix_moving_horizon.tex` - Animated diagram source
- `transfer_matrix.sty` - Shared style package

**Generation Scripts:**
- `generate_images.py` - Generates all PNG images from LaTeX
- `generate_animation.py` - Generates animated GIF
- `pdf_utils.py` - PDF to PNG conversion utilities

## Dependencies

**LaTeX** (plus pdftoppm or Ghostscript for PDF rasterization)
```bash
# Ubuntu/Debian
sudo apt-get install texlive texlive-latex-extra poppler-utils ghostscript

# macOS
brew install --cask mactex

# Windows
# https://miktex.org/download
```

**Python** (Pillow for GIF generation)
```bash
uv sync
```

## Usage

Generate all images and animation:
```bash
uv run assets/generate_images.py
```

Output saved to `images/`:
- `logo.png` (5000 DPI)
- `transfer_matrix.png`, `moving_horizon.png` (600 DPI)
- `animation.gif`

Generate animation only:
```bash
uv run assets/generate_animation.py assets/transfer_matrix_moving_horizon.tex --values "1,7,13,19,25"
```

## Notes

- Scripts auto-detect `pdftoppm` or Ghostscript (typically bundled with LaTeX)
- Temporary build files (`temp/`, `build/`) are auto-cleaned and git-ignored
- Edit `.tex` files and regenerate to update visualizations
