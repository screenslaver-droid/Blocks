import h5py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import os
import shutil
import math
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- CONFIGURATION ---
BASE_PATH    = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\2019"
CATALOG_PATH = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"
EVENTS_DIR   = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\Events"

os.makedirs(EVENTS_DIR, exist_ok=True)

# Duration of each frame window in seconds.
# SEVIR standard is 5 minutes (300s), but kept as a constant so it's easy
# to adjust if working with non-standard subsets.
SECONDS_PER_FRAME = 300

SEVIR_PROJ = ccrs.LambertConformal(
    central_longitude=-98.0,
    central_latitude=38.0,
    standard_parallels=(30.0, 60.0),
    globe=ccrs.Globe(semimajor_axis=6370000, semiminor_axis=6370000)
)

# ir107/ir069 → None means: compute from real data at load time (int16, ~16k-33k range)
# VIL         → 0-255 (uint8, fixed)
# lght        → scatter coloring, vmin/vmax unused
CHANNEL_DEFS = [
    ('vil',   'Mass (VIL)',        'jet',     0,    255 ),
    ('ir107', 'Cloud Top (IR107)', 'gray_r',  None, None),
    ('ir069', 'Moisture (IR069)',  'viridis', None, None),
    ('lght',  'Lightning',         'hot',     None, None),
]

# --------------------------------------------------------------------------- #
# FILE HELPERS
# --------------------------------------------------------------------------- #
def get_local_filename(event_id, img_type, catalog):
    row = catalog[(catalog['id'] == event_id) & (catalog['img_type'] == img_type)]
    if row.empty: return None
    return os.path.join(BASE_PATH, img_type, os.path.basename(row.iloc[0]['file_name']))

def read_data(filepath, event_id, img_type):
    if filepath is None or not os.path.exists(filepath): return None
    with h5py.File(filepath, 'r') as f:
        if event_id in f:
            return f[event_id][:]
        if 'id' in f:
            ids = f['id'][:]
            if len(ids) > 0 and isinstance(ids[0], bytes):
                ids = [x.decode('utf-8') for x in ids]
            if event_id in ids:
                idx = np.where(np.array(ids) == event_id)[0][0]
                if img_type in f:
                    return f[img_type][idx]
    return None

def load_event_data(event_id, catalog):
    data_store = {}
    found_any  = False
    for ch in ['vil', 'ir107', 'ir069', 'lght']:
        path = get_local_filename(event_id, ch, catalog)
        if path:
            data_store[ch] = read_data(path, event_id, ch)
            if data_store[ch] is not None:
                found_any = True
        else:
            data_store[ch] = None
    return data_store if found_any else {}

# --------------------------------------------------------------------------- #
# DYNAMIC FRAME COUNT
# --------------------------------------------------------------------------- #
def infer_frame_count(data_dict: dict) -> int:
    """
    Determine the true number of frames from whichever channels are loaded.

    Image channels (vil, ir107, ir069):
        shape = (H, W, T)  →  T is axis 2, read directly.

    Lightning (lght):
        shape = (N_flashes, 5), col 0 = time_offset in seconds.
        Number of windows = ceil(max_time_offset / SECONDS_PER_FRAME) + 1.
        The +1 accounts for the window that *contains* the last flash
        (e.g. a flash at t=14399s falls in window 47 of 0-indexed 48 windows).

    Strategy: collect a candidate frame count from every available channel,
    then take the maximum so no channel is clipped short. Falls back to the
    SEVIR standard of 49 if nothing can be inferred (all channels missing).
    """
    candidates = []

    for key, data in data_dict.items():
        if data is None:
            continue
        if key == 'lght':
            if data.ndim == 2 and data.shape[1] >= 1 and len(data) > 0:
                max_t = float(data[:, 0].max())
                # Which 5-min window does the last flash fall in?
                n_windows = int(math.floor(max_t / SECONDS_PER_FRAME)) + 1
                candidates.append(n_windows)
        else:
            # Image channel: shape = (H, W, T)
            if data.ndim == 3:
                candidates.append(data.shape[2])

    if not candidates:
        print("  [Warning] Could not infer frame count from data — defaulting to 49.")
        return 49

    frame_count = max(candidates)
    total_min   = frame_count * SECONDS_PER_FRAME // 60
    print(f"  Inferred frame count: {frame_count} frames "
          f"({total_min} min, {SECONDS_PER_FRAME}s/frame)")
    return frame_count

# --------------------------------------------------------------------------- #
# COLOUR SCALE HELPER
# --------------------------------------------------------------------------- #
def _resolve_vmin_vmax(key, data, vmin, vmax):
    """Compute global min/max for IR channels (int16, ~16k-33k) from real data."""
    if key == 'lght' or data is None:
        return vmin, vmax
    if vmin is None or vmax is None:
        valid = data[data != 0]
        if valid.size == 0:
            return 0, 1
        return float(valid.min()), float(valid.max())
    return vmin, vmax

# --------------------------------------------------------------------------- #
# STATIC PLOT (LAST FRAME)
# --------------------------------------------------------------------------- #
def plot_event(event_id, data_dict, catalog):
    subset = catalog[catalog['id'] == event_id]
    if subset.empty:
        print(f"Error: Event ID {event_id} not found in catalog.")
        return

    row    = subset.iloc[0]
    extent = [row['llcrnrlon'], row['urcrnrlon'], row['llcrnrlat'], row['urcrnrlat']]

    frame_count = infer_frame_count(data_dict)
    total_min   = frame_count * SECONDS_PER_FRAME // 60
    print(f"\nPlotting {event_id}  |  Bounds: {extent}")

    fig = plt.figure(figsize=(20, 5))
    for i, (key, title, cmap, vmin, vmax) in enumerate(CHANNEL_DEFS):
        ax   = fig.add_subplot(1, 4, i + 1, projection=SEVIR_PROJ)
        data = data_dict.get(key)
        rv, rx = _resolve_vmin_vmax(key, data, vmin, vmax)
        _draw_channel_static(ax, data, key, cmap, extent, vmin=rv, vmax=rx, fig=fig)
        ax.set_title(title)

    plt.suptitle(
        f"Event {event_id}: Last Frame  "
        f"(T+{total_min} min, {frame_count} frames @ {SECONDS_PER_FRAME}s each)",
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

def _draw_channel_static(ax, data, key, cmap, extent, vmin, vmax, fig=None):
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES,  linewidth=1.5, edgecolor='black', zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=2.0, edgecolor='black', zorder=10)

    if data is None:
        ax.text(0.5, 0.5, "MISSING", transform=ax.transAxes, ha='center', color='red')
        return

    if key == 'lght':
        lats, lons = data[:, 1], data[:, 2]
        energy = data[:, 3].astype(np.float32)
        bbox   = ((lats >= extent[2]) & (lats <= extent[3]) &
                  (lons >= extent[0]) & (lons <= extent[1]))
        if np.sum(bbox) > 0:
            log_e = np.log1p(np.clip(energy[bbox], 0, None))
            norm  = mcolors.Normalize(vmin=log_e.min(), vmax=log_e.max() + 1e-6)
            sc = ax.scatter(lons[bbox], lats[bbox], c=log_e, cmap='hot', norm=norm,
                            s=30, marker='+', transform=ccrs.PlateCarree(), zorder=5)
            if fig: plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label='log(Energy+1)')
        else:
            ax.text(0.5, 0.5, "No Strikes", transform=ax.transAxes,
                    ha='center', fontsize=9, color='gray')
    else:
        img = data[..., -1].astype(np.float32)
        im  = ax.imshow(img, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax,
                        extent=extent, transform=ccrs.PlateCarree())
        if fig: plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# --------------------------------------------------------------------------- #
# WRITER DETECTION
# --------------------------------------------------------------------------- #
def _get_writer():
    if shutil.which("ffmpeg") is not None:
        print("  ffmpeg detected — saving as .mp4")
        return animation.FFMpegWriter(fps=5, bitrate=1800), ".mp4"
    else:
        print("  ffmpeg NOT found on PATH — falling back to Pillow GIF writer.")
        print("  Install ffmpeg from https://ffmpeg.org/download.html and add to PATH for .mp4")
        return animation.PillowWriter(fps=5), ".gif"

# --------------------------------------------------------------------------- #
# FAST ANIMATION — dynamic frame count
# --------------------------------------------------------------------------- #
def animate_event(event_id, data_dict, catalog):
    """
    Animate the full event duration.

    Frame count is inferred from the data (image channel T-axis or lightning
    time offsets) rather than hardcoded, so events shorter or longer than the
    SEVIR 49-frame standard are handled correctly.

    Per-frame rendering uses in-place artist updates (im.set_data / scatter.remove)
    so Cartopy map features are drawn only once, not on every frame.
    """
    subset = catalog[catalog['id'] == event_id]
    if subset.empty:
        print(f"Error: Event ID {event_id} not found in catalog.")
        return

    row         = subset.iloc[0]
    extent      = [row['llcrnrlon'], row['urcrnrlon'], row['llcrnrlat'], row['urcrnrlat']]
    frame_count = infer_frame_count(data_dict)
    total_min   = frame_count * SECONDS_PER_FRAME // 60

    writer, ext = _get_writer()
    dpi      = 80 if ext == ".gif" else 120
    out_path = os.path.join(EVENTS_DIR, f"{event_id}{ext}")

    print(f"\nBuilding animation for {event_id}")
    print(f"  {frame_count} frames  |  {total_min} min total  |  dpi={dpi}")
    print(f"  Output → {out_path}")

    # Stable colour bounds computed once from full data volume
    resolved = {
        key: _resolve_vmin_vmax(key, data_dict.get(key), vmin, vmax)
        for key, _, __, vmin, vmax in CHANNEL_DEFS
    }

    # ------------------------------------------------------------------ #
    # BUILD FIGURE ONCE — map features drawn here, never again
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(20, 5))
    axes = [fig.add_subplot(1, 4, i + 1, projection=SEVIR_PROJ) for i in range(4)]
    time_text = fig.text(0.5, 0.01, '', ha='center', fontsize=11)

    artist_map: dict = {}   # key → AxesImage or scatter PathCollection or None

    for ax, (key, title, cmap, _, __) in zip(axes, CHANNEL_DEFS):
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.STATES,  linewidth=1.5, edgecolor='black', zorder=10)
        ax.add_feature(cfeature.BORDERS, linewidth=2.0, edgecolor='black', zorder=10)
        ax.set_title(title, fontsize=10)

        data   = data_dict.get(key)
        rv, rx = resolved[key]

        if key == 'lght':
            artist_map[key] = None   # scatter replaced each frame

        elif data is not None:
            # Guard: some events may have fewer frames than the inferred max —
            # use the last available frame index to seed imshow safely.
            t0  = min(0, data.shape[2] - 1)
            im  = ax.imshow(data[..., t0].astype(np.float32),
                            origin='lower', cmap=cmap, vmin=rv, vmax=rx,
                            extent=extent, transform=ccrs.PlateCarree(),
                            animated=True)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            artist_map[key] = im
        else:
            ax.text(0.5, 0.5, "MISSING", transform=ax.transAxes,
                    ha='center', color='red')
            artist_map[key] = None

    plt.tight_layout()

    # ------------------------------------------------------------------ #
    # PER-FRAME UPDATE — only data changes, map layer untouched
    # ------------------------------------------------------------------ #
    def _update(t):
        updated = []

        for ax, (key, title, cmap, _, __) in zip(axes, CHANNEL_DEFS):
            data = data_dict.get(key)
            if data is None:
                continue

            if key == 'lght':
                prev = artist_map[key]
                if prev is not None:
                    prev.remove()

                t_start = t * SECONDS_PER_FRAME
                t_end   = t_start + SECONDS_PER_FRAME
                mask    = (data[:, 0] >= t_start) & (data[:, 0] < t_end)
                lats, lons = data[mask, 1], data[mask, 2]
                energy = data[mask, 3].astype(np.float32)
                bbox   = ((lats >= extent[2]) & (lats <= extent[3]) &
                          (lons >= extent[0]) & (lons <= extent[1]))

                if np.sum(bbox) > 0:
                    log_e = np.log1p(np.clip(energy[bbox], 0, None))
                    norm  = mcolors.Normalize(vmin=log_e.min(), vmax=log_e.max() + 1e-6)
                    sc = ax.scatter(lons[bbox], lats[bbox], c=log_e, cmap='hot',
                                    norm=norm, s=30, marker='+',
                                    transform=ccrs.PlateCarree(), zorder=5,
                                    animated=True)
                    artist_map[key] = sc
                    updated.append(sc)
                else:
                    artist_map[key] = None

            else:
                im = artist_map[key]
                if im is not None:
                    # Clamp t to the channel's actual T-axis length.
                    # If lightning pushes frame_count beyond what an image
                    # channel has (e.g. a late stray flash), hold the last frame.
                    t_safe = min(t, data.shape[2] - 1)
                    im.set_data(data[..., t_safe].astype(np.float32))
                    updated.append(im)

        elapsed_min = t * SECONDS_PER_FRAME // 60
        time_text.set_text(
            f"Event {event_id}  |  T+{elapsed_min:03d} min  "
            f"(frame {t + 1}/{frame_count}  —  {total_min} min total)"
        )
        updated.append(time_text)
        return updated

    anim = animation.FuncAnimation(
        fig, _update,
        frames=frame_count,
        interval=200,
        blit=False   # Cartopy transforms prevent blit=True
    )

    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not os.path.exists(CATALOG_PATH):
        print("Catalog not found!")
        exit()

    print("Loading Catalog...")
    catalog = pd.read_csv(CATALOG_PATH, parse_dates=['time_utc'], low_memory=False)

    while True:
        print("\n" + "=" * 40)
        user_input = input("Enter Event ID (or 'q' to quit): ").strip()

        if user_input.lower() == 'q':
            break
        if user_input == "":
            continue

        print(f"Searching for {user_input}...")
        data_store = load_event_data(user_input, catalog)

        if not data_store:
            print("  [Error] No data found for this ID. Check if you downloaded it.")
            continue

        plot_event(user_input, data_store, catalog)

        save = input("Save animation? [y/n]: ").strip().lower()
        if save == 'y':
            animate_event(user_input, data_store, catalog)