import pandas as pd
from skyfield.api import Loader, EarthSatellite, utc, wgs84
from datetime import datetime, timedelta
import os

# Set up Skyfield's data loader
load = Loader('./data') 
ts = load.timescale()

# Change working directory to where your input files are located
os.chdir(r'C:\Users\陆尧\Desktop\source code')

def read_tle(file_path, common_satellite_names):
    """
    Reads TLE file and returns a dictionary of EarthSatellite objects for common satellites.
    """
    satellites = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 3):
            name = lines[i].strip()
            if name in common_satellite_names:
                line1 = lines[i+1].strip()
                line2 = lines[i+2].strip()
                satellite = EarthSatellite(line1, line2, name, ts)
                satellites[name] = satellite
    return satellites

def compute_satellite_positions(args):
    """
    Calculates subpoints (longitude, latitude) of a satellite for each time in the list.
    """
    satellite, times = args
    subpoints = []
    for t in times:
        subpoint = satellite.at(t).subpoint()
        lat_deg = subpoint.latitude.degrees
        lon_deg = subpoint.longitude.degrees
        subpoints.append((lon_deg, lat_deg))
    return satellite.name, subpoints

def main():
    tle_file_path = 'starlink_2024-08-19.txt'
    common_satellites_file_path = 'common_starlink_satellites.csv'

    # Load common satellite names from CSV
    common_satellites_df = pd.read_csv(common_satellites_file_path)
    common_satellite_names = set(common_satellites_df['Satellite Name'].str.strip())

    print("Read TLE data...")
    satellites = read_tle(tle_file_path, common_satellite_names)
    satellite_names = list(satellites.keys())

    # Define time range: From 00:00 to 23:59 of the same day at 1-minute intervals
    start_time = datetime(2024, 8, 19, 0, 0, 0, tzinfo=utc)
    end_time = datetime(2024, 8, 19, 23, 59, 0, tzinfo=utc)
    time_step = timedelta(minutes=1)

    # Generate time list
    times = []
    current_time = start_time
    while current_time <= end_time:
        t = ts.from_datetime(current_time)
        times.append(t)
        current_time += time_step

    # Prepare arguments for computation
    args = [(satellites[name], times) for name in satellite_names]

    print("Start calculating satellite positions...")
    results = [compute_satellite_positions(arg) for arg in args]

    print("Building results matrix...")
    data = {}
    for name, subpoints in results:
        lon_lat_list = []
        for (lon_deg, lat_deg) in subpoints:
            lat_abs = abs(lat_deg)
            lon_str = f"{lon_deg:.2f}E" if lon_deg >= 0 else f"{-lon_deg:.2f}W"
            lat_str = f"{lat_abs:.2f}N" if lat_deg >= 0 else f"{lat_abs:.2f}S"
            lon_lat_list.append(f"{lon_str},{lat_str}")
        data[name] = lon_lat_list

    # Create a DataFrame with times as index and satellite names as columns
    time_strings = [dt.utc_strftime('%H:%M') for dt in times]
    df = pd.DataFrame(data, index=time_strings)
    df.index.name = 'Time'
    df = df[satellite_names]  # Ensure column order matches satellite_names

    output_file = 'satellite_positions.csv'
    df.to_csv(output_file, encoding='utf-8')

    print(df.head())

if __name__ == '__main__':
    main()
