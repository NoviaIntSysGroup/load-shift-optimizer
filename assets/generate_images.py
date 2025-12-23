import os
import shutil
from pathlib import Path

from pdf_utils import find_rasterizer, pdf_to_pngs_multipage

# Determine project root (one level up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Define paths
TEX_DIR = SCRIPT_DIR  # assets/
IMAGES_DIR = PROJECT_ROOT / "images"
TEMP_DIR = SCRIPT_DIR / "temp"

# Ensure output directory exists
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Detect PDF rasterizer once at startup
raster_tool, raster_exe = find_rasterizer()
print(f"[info] Using {raster_tool} for PDF rasterization")

# Find all .tex files in the scripts directory
tex_files = list(TEX_DIR.glob("*.tex"))

# Compile LaTeX files and convert to PNG
for tex_file in tex_files:
    # Skip transfer_matrix_moving_horizon.tex as it will be processed separately for GIF
    if tex_file.name == "transfer_matrix_moving_horizon.tex":
        continue
    
    # Compile LaTeX to PDF (run from TEX_DIR to find .sty files)
    original_dir = os.getcwd()
    os.chdir(TEX_DIR)
    os.system(f'pdflatex -output-directory="{TEMP_DIR}" "{tex_file.name}"')
    os.chdir(original_dir)
    
    base_name = tex_file.stem
    pdf_file = TEMP_DIR / f"{base_name}.pdf"
    
    if pdf_file.exists():
        # Convert PDF to images at high resolution (600 DPI for crisp quality)
        dpi = 5000 if 'logo' in base_name else 600

        # Convert PDF to PNG(s)
        temp_prefix = TEMP_DIR / base_name
        png_files = pdf_to_pngs_multipage(
            pdf_file, temp_prefix, dpi=dpi, tool=raster_tool, exe_path=raster_exe
        )

        # Move PNGs to final location with appropriate naming
        for i, png_path in enumerate(png_files):
            if len(png_files) == 1:
                output_path = IMAGES_DIR / f"{base_name}.png"
            else:
                output_path = IMAGES_DIR / f"{base_name}_page_{i+1}.png"
            shutil.move(str(png_path), str(output_path))

        print(f"Generated: {base_name}.png")

# Clean up temporary files
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)

# Generate animated GIF
gif_tex = TEX_DIR / "transfer_matrix_moving_horizon.tex"
if gif_tex.exists():
    make_gif_script = TEX_DIR / "generate_animation.py"
    os.system(f'python "{make_gif_script}" "{gif_tex}" --values "1,7,13,19,25"')
    
    # Move animation.gif to images folder (check both root and script dir)
    animation_locations = [
        PROJECT_ROOT / "animation.gif",
        SCRIPT_DIR / "animation.gif"
    ]
    animation_dst = IMAGES_DIR / "animation.gif"
    
    for animation_src in animation_locations:
        if animation_src.exists():
            shutil.move(str(animation_src), str(animation_dst))
            print("Generated: animation.gif")
            break
    
    # Clean up build folder if it exists
    build_dir = SCRIPT_DIR / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
