"""
extract_frames.py
-----------------
Extracts all frames from a GIF and saves them as numbered PNGs
in the format required by LaTeX's animate package.

Usage:
    python extract_frames.py

Output:
    S823789_slic_1500_fused_channel0.png
    S823789_slic_1500_fused_channel1.png
    ...
    S823789_slic_1500_fused_channelN.png

Requires: Pillow  (pip install Pillow)
"""

from PIL import Image
import os

GIF_PATH  = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\Events\S830464_hierarchical_graph.gif"
BASENAME  = "S830464_slic_950_hierarchical"

def extract_frames(gif_path, basename):
    with Image.open(gif_path) as gif:
        frame_idx = 0
        while True:
            out_path = f"{basename}{frame_idx}.png"
            # Convert to RGBA first to preserve transparency, then to RGB for PNG
            frame = gif.convert("RGBA")
            # Composite onto white background (LaTeX renderers expect opaque images)
            background = Image.new("RGB", frame.size, (255, 255, 255))
            background.paste(frame, mask=frame.split()[3])
            background.save(out_path)
            print(f"  Saved {out_path}")
            frame_idx += 1
            try:
                gif.seek(frame_idx)
            except EOFError:
                break
    print(f"\nDone. Extracted {frame_idx} frames (0 to {frame_idx - 1}).")
    print(f"Update your LaTeX command to: {{0}}{{{frame_idx - 1}}}")

if __name__ == "__main__":
    if not os.path.exists(GIF_PATH):
        print(f"ERROR: {GIF_PATH} not found in current directory.")
        print(f"Current directory: {os.getcwd()}")
    else:
        print(f"Extracting frames from {GIF_PATH} ...")
        extract_frames(GIF_PATH, BASENAME)