import os
import sys
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- Configuration ---
OVERLAY_FILE = "data/session_overlay.csv"
KPMS_FILE = "data/moseq_features.csv"
VIDEO_FILE = "data/session_video.mp4"

METRICS_FILE = None 
PORT_TIME_FILE = None 

VALID_CATEGORIES = ["AB", "CD", "BD", "BC", "DE", "AE"]  
VALID_OUTCOMES = ["Correct"] 

SLICE_TRIALS = None 
SLICE_METRICS = None 
METRIC_FIELD = None 
PORT_METRIC_FIELD = None 

PORT_CLICK_ORDER = "IECABD"  
AWAY_ONLY = False 
AWAY_RADIUS = 0.15  

NAMING_MODE = "allowed_categories"  
ANIMAL_GROUP = "male"  
OUT_DIR = "sliced_trials"

# --- Coordinate Acquisition ---
def get_port_coords(video_path, label_order):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    if not success:
        raise IOError("Couldn't read the session video file.")

    coords = []
    fig, ax = plt.subplots()
    ax.set_title(f"Click 6 ports in order: {label_order}")
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
        raise ValueError(f"Missing ports! Only got {len(coords)} of 6 clicks.")

    return coords, frame.shape[1], frame.shape[0]

# --- Helper Logic ---
def check_port_distance(row, coords, w, h, radius_pct):
    max_dim = max(w, h)
    for (px, py) in coords:
        if np.hypot(row['NoseX'] - px, row['NoseY'] - py) < (radius_pct * max_dim):
            return False
    return True

def parse_trial_type(trial_str, valid_types):
    letters = ''.join(re.findall(r'[A-Z]', str(trial_str).upper()))
    for t in valid_types:
        if all(char in letters for char in list(t.upper())):
            return t
    return None

def fetch_target_trials(trial_list, metric_df, field, mode):
    if field and field not in metric_df.columns:
        raise KeyError(f"The metric field '{field}' isn't in your data columns.")
    if mode == "first":
        return [trial_list[0]]
    elif mode == "last":
        return [trial_list[-1]]
    elif mode == "highest":
        return [metric_df.loc[metric_df[field].idxmax()]["TrialNumber"]]
    elif mode == "lowest":
        return [metric_df.loc[metric_df[field].idxmin()]["TrialNumber"]]
    return list(trial_list)

# --- Main Slicing Pipeline ---
def slice_session_data():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    tracking_df = pd.read_csv(OVERLAY_FILE)
    moseq_df = pd.read_csv(KPMS_FILE)
    base_name = os.path.basename(OVERLAY_FILE).split("_overlay")[0]

    # Pull behavioral hardware configurations interactively
    ports, vid_w, vid_h = get_port_coords(VIDEO_FILE, PORT_CLICK_ORDER)
    max_dim = max(vid_w, vid_h)
    
    port_labels = ["Port_I", "Port_E", "Port_A", "Port_C", "Port_B", "Port_D"]
    for lbl, (px, py) in zip(port_labels, ports):
        tracking_df[f"RelDist_{lbl}"] = np.sqrt((tracking_df["NoseX"] - px) ** 2 + (tracking_df["NoseY"] - py) ** 2) / max_dim

    clean_track = tracking_df.dropna(subset=["TrialNumber"])
    clean_track = clean_track[clean_track["TrialOutcome"].isin(VALID_OUTCOMES)]
    clean_track = clean_track[clean_track["TrialType"].apply(lambda x: parse_trial_type(x, VALID_CATEGORIES) is not None)]
    
    unique_trials = clean_track["TrialNumber"].unique()
    targets = {}

    if SLICE_TRIALS:
        for mode in SLICE_TRIALS:
            if mode == "first":
                targets[int(unique_trials[0])] = "first"
            elif mode == "last":
                targets[int(unique_trials[-1])] = "last"

    if SLICE_METRICS:
        for mode in SLICE_METRICS:
            if METRIC_FIELD:
                metrics_df = pd.read_csv(METRICS_FILE)
                sub_df = metrics_df[metrics_df["TrialNumber"].isin(set(unique_trials))]
                for t in fetch_target_trials(sub_df["TrialNumber"].unique(), sub_df, METRIC_FIELD, mode):
                    targets[int(t)] = f"{mode}{METRIC_FIELD}"
            elif PORT_METRIC_FIELD:
                port_df = pd.read_csv(PORT_TIME_FILE)
                sub_df = port_df[port_df["TrialNumber"].isin(set(unique_trials))]
                for t in fetch_target_trials(sub_df["TrialNumber"].unique(), sub_df, PORT_METRIC_FIELD, mode):
                    targets[int(t)] = f"{mode}{PORT_METRIC_FIELD}"

    if not targets:
        for t in unique_trials:
            t = int(t)
            t_rows = tracking_df[tracking_df["TrialNumber"] == t]
            if t_rows.empty:
                continue
                
            if NAMING_MODE == "allowed_categories":
                label = ''.join(filter(str.isalpha, str(t_rows.iloc[0]["TrialType"])))
            elif NAMING_MODE == "allowed_outcomes":
                label = str(t_rows.iloc[0]["TrialOutcome"]).lower()
            elif NAMING_MODE == "mice_group":
                label = ANIMAL_GROUP
            else:
                label = "all"
            targets[t] = label

    # Execute extraction loop
    for t_num, tag in targets.items():
        t_rows = tracking_df[tracking_df["TrialNumber"] == t_num]
        if t_rows.empty or (t_rows.iloc[0]["TrialOutcome"] not in VALID_OUTCOMES) or not parse_trial_type(t_rows.iloc[0]["TrialType"], VALID_CATEGORIES):
            continue

        t_start, t_end = t_rows.index.min(), t_rows.index.max() + 1
        sliced_moseq = moseq_df.iloc[t_start:t_end].copy()
        sliced_track = t_rows.copy()

        if AWAY_ONLY:
            sliced_track = sliced_track[sliced_track.apply(check_port_distance, axis=1, coords=ports, w=vid_w, h=vid_h, radius_pct=AWAY_RADIUS)]
            sliced_moseq = sliced_moseq.loc[sliced_track.index]

        # Calculate normalized geometry values
        for lbl, (px, py) in zip(list(PORT_CLICK_ORDER), ports):
            sliced_track[f"RelDist_Port_{lbl}"] = np.hypot(sliced_track['NoseX'] - px, sliced_track['NoseY'] - py) / max_dim
        
        sliced_moseq = sliced_moseq.join(sliced_track[[f"RelDist_Port_{lbl}" for lbl in PORT_CLICK_ORDER]])

        fn_elements = [tag, base_name, f"Trial{t_num}"]
        if AWAY_ONLY:
            fn_elements.append("awayfromports")

        out_path = os.path.join(OUT_DIR, "_".join(fn_elements) + ".csv")
        sliced_moseq.to_csv(out_path, index=False)

if __name__ == "__main__":
    slice_session_data()