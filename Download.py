import pandas as pd
import os
import random

# --- CONFIGURATION ---
# Where to save the downloaded files (relative to this script)
OUTPUT_DIR = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR"
CATALOG_FILE = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR\CATALOG.csv"
# Number of events to download. 
# Set to 5-10 first to test, then increase to 25-50 once confirmed working.
NUM_EVENTS = 1

def generate_batch_script():
    print(f"Loading {CATALOG_FILE}...")
    df = pd.read_csv(CATALOG_FILE, parse_dates=['time_utc'], low_memory=False)

    # 1. Filter for the 4 types we need
    required_types = {'vil', 'ir107', 'ir069', 'lght'}
    df = df[df['img_type'].isin(required_types)]
    
    # Filter for low missing data (high quality)
    df = df[df['pct_missing'] == 0.0]

    # 2. Group by Event ID to find events that have ALL 4 types
    print("Grouping by Event ID to find complete sets...")
    # This checks if an ID has all 4 required image types associated with it
    valid_ids = df.groupby('id').filter(lambda x: required_types.issubset(set(x['img_type'])))['id'].unique()
    
    print(f"Found {len(valid_ids)} events with all 4 channels active.")

    if len(valid_ids) == 0:
        print("No valid events found! Check catalog or criteria.")
        return

    # 3. Randomly select our "Golden Subset"
    selected_ids = random.sample(list(valid_ids), min(NUM_EVENTS, len(valid_ids)))
    print(f"Selected {len(selected_ids)} random events for download.")

    # 4. Generate the AWS Commands
    batch_commands = []
    
    # Filter dataframe to only our selected events
    subset = df[df['id'].isin(selected_ids)]

    for index, row in subset.iterrows():
        # The 'file_name' column usually looks like: 'vil/2018/SEVIR_VIL_STORMEVENTS_...h5'
        raw_filename = row['file_name']
        
        # CORRECTED S3 PATH: Just append the raw filename to the data root
        s3_path = f"s3://sevir/data/{raw_filename}"
        
        # Parse the raw filename to reorganize locally as Year/Type/File
        # raw_filename parts: [Type, Year, Filename]
        parts = raw_filename.replace('\\', '/').split('/')
        
        if len(parts) == 3:
            img_type = parts[0]
            year = parts[1]
            fname = parts[2]
            
            # Construct local path: .../SEVIR/2018/vil/filename.h5
            local_folder = os.path.join(OUTPUT_DIR, year, img_type)
            local_path = os.path.join(local_folder, fname)
            
            # Create directory command (using distinct to avoid duplicates in batch file)
            # (We will just assume the batch execution handles existing dirs gracefully or we make them here)
            # For a batch file, it's safer to ensure dir exists
            batch_commands.append(f'if not exist "{local_folder}" mkdir "{local_folder}"')
            
            # Add the download command
            cmd = f'aws s3 cp "{s3_path}" "{local_path}" --no-sign-request'
            batch_commands.append(cmd)
        else:
            print(f"Skipping weird filename format: {raw_filename}")

    # Remove duplicate mkdir commands to clean up the script
    # (Optional, but makes the batch file readable)
    unique_commands = []
    seen = set()
    for cmd in batch_commands:
        if cmd not in seen or "aws s3" in cmd: # Always keep copy commands, dedup mkdirs
            unique_commands.append(cmd)
            if "mkdir" in cmd:
                seen.add(cmd)

    # 5. Save to a batch file
    with open('download_golden.bat', 'w') as f:
        f.write("@echo off\n")
        f.write("echo Starting Golden Subset Download (Corrected)...\n")
        for cmd in unique_commands:
            f.write(cmd + "\n")
        f.write("echo Download Complete.\n")
        f.write("pause\n")

    print(f"Done! 'download_golden.bat' created with {len(unique_commands)} commands.")

def generate_resume_script():
    # 1. Load the previous batch file to see what we PLANNED to download
    # (Or we can just regenerate the list if you haven't changed the random seed/logic)
    # Since we used random.sample, we can't easily reproduce the EXACT list unless
    # you saved the list of IDs.
    
    # CRITICAL: Did you save the list of IDs?
    # If not, we have to scan the directory to see what IS there, and only fix the broken ones.
    # A better approach for "Resuming" without knowing the original random seed is:
    # "Scan the batch file itself."
    
    if not os.path.exists('download_golden.bat'):
        print("Error: Original 'download_golden.bat' not found. Cannot resume.")
        return

    print("Scanning 'download_golden.bat' to find missing files...")
    
    new_commands = []
    
    with open('download_golden.bat', 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # We only care about lines that actually download stuff
        if line.startswith("aws s3 cp"):
            # Extract the local path from the command
            # Format: aws s3 cp "s3://..." "C:\Local\Path..." --no-sign-request
            parts = line.split('"')
            if len(parts) >= 4:
                local_path = parts[3] # The second quoted string is the destination
                
                # CHECK: Does this file exist?
                if os.path.exists(local_path):
                    # Check if it's empty/corrupt (Size 0)
                    if os.path.getsize(local_path) == 0:
                        print(f"Found empty file (re-downloading): {os.path.basename(local_path)}")
                        new_commands.append(line)
                    else:
                        print(f"Skipping existing file: {os.path.basename(local_path)}")
                else:
                    # File doesn't exist at all -> Download it
                    new_commands.append(line)
    
    if not new_commands:
        print("All files appear to be downloaded successfully!")
        return

    # Write the new "Resume" batch file
    with open('resume_download.bat', 'w') as f:
        f.write("@echo off\n")
        f.write("echo Resuming Download (Skipping existing files)...\n")
        for cmd in new_commands:
            f.write(cmd + "\n")
        f.write("echo Resume Complete.\n")
        f.write("pause\n")

    print(f"\nCreated 'resume_download.bat' with {len(new_commands)} remaining files.")
    print("Run 'resume_download.bat' to finish.")

if __name__ == "__main__":
    generate_batch_script()
    #generate_resume_script()