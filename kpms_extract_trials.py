import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re

def get_port_coords(vid_path, port_order):
    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError("Couldn't read the video file! Double check your file path and ensure the video isn't corrupted.")

    coords = []
    fig, ax = plt.subplots()
    ax.set_title(f"Click 6 port locations in order: {port_order}")
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def onclick(event):
        if event.xdata and event.ydata:
            coords.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
            if len(coords) == 6:
                plt.close()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)

    if len(coords) < 6:
        raise ValueError(f"You only clicked {len(coords)} ports instead of 6. Run the script again and click all of them.")

    vid_h, vid_w = frame.shape[0], frame.shape[1]
    return coords, vid_w, vid_h

def is_far_from_ports(row, port_coords, w, h, threshold):
    # Drops frames where the mouse is too close to a port, usually to isolate pure locomotion
    for px, py in port_coords:
        if np.hypot(row['NoseX'] - px, row['NoseY'] - py) < threshold * max(w, h):
            return False
    return True

def calc_rel_distances(metadata, port_coords, labels, w, h):
    max_dim = max(w, h)
    for lbl, (px, py) in zip(labels, port_coords):
        metadata[f"RelDist_Port_{lbl}"] = np.hypot(metadata['NoseX'] - px, metadata['NoseY'] - py) / max_dim
    return metadata

def parse_trial_type(raw_type, target_types):
    stripped = ''.join(re.findall(r'[A-Z]', str(raw_type).upper()))
    for lbl in target_types:
        if all(char in stripped for char in list(lbl.upper())):
            return lbl
    return None

def filter_by_condition(trials, metadata, field, condition):
    if field and field not in metadata.columns:
        raise KeyError(f"Looks like '{field}' isn't in your spreadsheet columns. Check for typos.")
    
    if condition == "first":
        return [trials[0]]
    elif condition == "last":
        return [trials[-1]]
    elif condition == "highest":
        return [metadata.loc[metadata[field].idxmax()]["TrialNumber"]]
    elif condition == "lowest":
        return [metadata.loc[metadata[field].idxmin()]["TrialNumber"]]
    elif condition == "all":
        return list(trials)
    
    raise ValueError("That slicing condition doesn't exist. Use 'first', 'last', 'highest', 'lowest', or 'all'.")

def run_extraction(config):
    os.makedirs(config['export_dir'], exist_ok=True)
    
    trial_meta = pd.read_csv(config['overlay_file'])
    pose_data = pd.read_csv(config['kpms_file'])
    
    type_metrics = pd.read_csv(config['type_metrics_file']) if config.get('type_metrics_file') else None
    port_time = pd.read_csv(config['port_time_file']) if config.get('port_time_file') else None
    session_id = os.path.basename(config['overlay_file']).split("_overlay")[0]

    port_coords, vid_w, vid_h = get_port_coords(config['video_file'], config['port_order'])
    trial_meta = calc_rel_distances(trial_meta, port_coords, list(config['port_order']), vid_w, vid_h)

    clean_meta = trial_meta.dropna(subset=["TrialNumber"])
    clean_meta = clean_meta[clean_meta["TrialOutcome"].isin(config['allowed_outcomes'])]
    clean_meta = clean_meta[clean_meta["TrialType"].apply(lambda x: parse_trial_type(x, config['allowed_categories']) is not None)]
    
    if clean_meta.empty:
        raise ValueError("No trials survived the filtering! Check if your outcome or category settings are too strict.")

    valid_trials = clean_meta["TrialNumber"].unique()
    target_trials = {}

    if config['slice_trials']:
        for cond in config['slice_trials']:
            if cond == "first" and len(valid_trials) > 0:
                target_trials[int(valid_trials[0])] = "first"
            elif cond == "last" and len(valid_trials) > 0:
                target_trials[int(valid_trials[-1])] = "last"

    if config['slice_metrics']:
        for cond in config['slice_metrics']:
            if config['metric_field'] and type_metrics is not None:
                valid_nums = set(clean_meta["TrialNumber"].unique())
                filt_metrics = type_metrics[type_metrics["TrialNumber"].isin(valid_nums)]
                for tr in filter_by_condition(filt_metrics["TrialNumber"].unique(), filt_metrics, config['metric_field'], cond):
                    target_trials[int(tr)] = f"{cond}{config['metric_field']}"
            elif config['port_metric_field'] and port_time is not None:
                valid_nums = set(clean_meta["TrialNumber"].unique())
                filt_ports = port_time[port_time["TrialNumber"].isin(valid_nums)]
                for tr in filter_by_condition(filt_ports["TrialNumber"].unique(), filt_ports, config['port_metric_field'], cond):
                    target_trials[int(tr)] = f"{cond}{config['port_metric_field']}"

    if not target_trials:
        for tr in valid_trials:
            tr = int(tr)
            t_rows = trial_meta[trial_meta["TrialNumber"] == tr]
            if t_rows.empty:
                continue

            if config['naming'] == "allowed_categories":
                reason = ''.join(filter(str.isalpha, str(t_rows.iloc[0]["TrialType"])))
            elif config['naming'] == "allowed_outcomes":
                reason = str(t_rows.iloc[0]["TrialOutcome"]).lower()  
            elif config['naming'] == "mice_group":
                reason = config['mice_group']
            else:
                reason = "all"
            target_trials[tr] = reason

    for t_num, reason in target_trials.items():
        t_rows = trial_meta[trial_meta["TrialNumber"] == t_num]
        t_type = t_rows.iloc[0]["TrialType"]
        t_outcome = t_rows.iloc[0]["TrialOutcome"]

        if t_outcome not in config['allowed_outcomes'] or not parse_trial_type(t_type, config['allowed_categories']):
            continue

        start_idx = t_rows.index.min()
        end_idx = t_rows.index.max() + 1

        sliced_pose = pose_data.iloc[start_idx:end_idx].copy()
        sliced_meta = t_rows.copy()

        if config['require_away']:
            sliced_meta = sliced_meta[sliced_meta.apply(is_far_from_ports, axis=1, port_coords=port_coords, w=vid_w, h=vid_h, threshold=config['away_pct'])]
            sliced_pose = sliced_pose.loc[sliced_meta.index]

        sliced_pose = sliced_pose.join(sliced_meta[[f"RelDist_Port_{lbl}" for lbl in config['port_order']]])

        parts = [reason, session_id, f"Trial{t_num}"]
        if config['require_away']:
            parts.append("awayfromports")

        out_path = os.path.join(config['export_dir'], "_".join(parts) + ".csv")
        sliced_pose.to_csv(out_path, index=False)


if __name__ == "__main__":
    extraction_config = {
        "overlay_file": "/path/to/your/data/session_overlay.csv",
        "kpms_file": "/path/to/your/data/session_kpms.csv",
        "video_file": "/path/to/your/data/session_video.mp4",
        "type_metrics_file": None,
        "port_time_file": None,
        "export_dir": "/path/to/your/output/sliced_trials",
        
        "allowed_categories": ["AB", "CD", "BD", "BC", "DE", "AE"],
        "allowed_outcomes": ["Correct"],
        "slice_trials": None,
        "slice_metrics": None,
        "metric_field": None,
        "port_metric_field": None,
        
        "naming": "allowed_categories",
        "mice_group": "male",
        "port_order": "IECABD",
        "require_away": False,
        "away_pct": 0.15
    }
    
    run_extraction(extraction_config)