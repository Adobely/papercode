import pandas as pd
import numpy as np
import os
import concurrent.futures
import time
import multiprocessing
from pyscipopt import Model

# If needed, set your working directory
os.chdir()

def read_common_satellites(common_satellites_file):
    df = pd.read_csv(common_satellites_file)
    satellite_names = df['Satellite Name'].astype(str).str.strip().tolist()
    return satellite_names

def read_satellite_positions(sat_positions_file, selected_satellites, time_step_interval=1):
    df = pd.read_csv(sat_positions_file)
    df.set_index('Time', inplace=True)
    # Sample every 'time_step_interval' rows
    df = df.iloc[::time_step_interval, :]
    satellite_names = [sat for sat in df.columns if sat in selected_satellites]
    df = df[satellite_names]
    time_steps = df.index.tolist()
    return time_steps, satellite_names, df

def read_observation_points(obs_points_file):
    """
    Reads observation points from a CSV file containing:
      latitude, longitude, is_high_seas
    You can change whether to limit the attack nodes to high sea or not.
    """
    df = pd.read_csv(obs_points_file)
    
    # Filter only the high-seas observation points, if you don't want to, just comment it out.
    if 'is_high_seas' in df.columns:
        df = df[df['is_high_seas'] == 1]
    
    # Convert to list of (latitude, longitude) pairs
    observation_points = list(zip(df['latitude'], df['longitude']))
    return observation_points

def preprocess_satellite_positions(sat_positions_df):
    sat_positions_df_reset = sat_positions_df.reset_index()
    melted_df = sat_positions_df_reset.melt(id_vars=['Time'], var_name='Satellite Name', value_name='Position')
    melted_df = melted_df.dropna(subset=['Position'])
    # Extract latitude and longitude
    extracted = melted_df['Position'].str.extract(r'([-\d.]+)([EW]),([-\d.]+)([NS])')
    extracted.columns = ['Lon', 'Lon_dir', 'Lat', 'Lat_dir']
    extracted['Lon'] = extracted['Lon'].astype(float)
    extracted['Lat'] = extracted['Lat'].astype(float)
    # Adjust signs based on direction
    extracted.loc[extracted['Lon_dir'] == 'W', 'Lon'] *= -1
    extracted.loc[extracted['Lat_dir'] == 'S', 'Lat'] *= -1
    melted_df = pd.concat([melted_df[['Time', 'Satellite Name']], extracted[['Lat', 'Lon']]], axis=1)
    return melted_df

def compute_cover_for_time(args):
    time, group, obs_lats, obs_lons, coverage_radius_km = args
    sat_lats = group['Lat'].values
    sat_lons = group['Lon'].values
    sat_names = group['Satellite Name'].values

    if len(sat_lats) == 0:
        return []

    # Convert degrees to radians
    sat_lats_rad = np.radians(sat_lats)[:, np.newaxis]  # Shape (N_sats, 1)
    sat_lons_rad = np.radians(sat_lons)[:, np.newaxis]
    obs_lats_rad = np.radians(obs_lats)  # Shape (N_obs,)
    obs_lons_rad = np.radians(obs_lons)

    # Compute differences
    dlat = obs_lats_rad - sat_lats_rad  # Shape (N_sats, N_obs)
    dlon = obs_lons_rad - sat_lons_rad

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(sat_lats_rad) * np.cos(obs_lats_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    earth_radius_km = 6371
    distances = earth_radius_km * c  # Shape (N_sats, N_obs)

    # Find where distances <= coverage_radius_km
    sat_indices, obs_indices = np.where(distances <= coverage_radius_km)

    cover_list = []
    for idx in range(len(sat_indices)):
        sat_idx = sat_indices[idx]
        obs_idx = obs_indices[idx]
        sat_name = sat_names[sat_idx]
        cover_list.append((sat_name, obs_idx, time))

    return cover_list

def compute_cover_matrix_parallel(observation_points, melted_df, time_steps,
                                  coverage_radius_km=1000, max_time_index=None):
    obs_lats = np.array([obs_lat for obs_lat, obs_lon in observation_points])
    obs_lons = np.array([obs_lon for obs_lat, obs_lon in observation_points])

    # Limit to time steps up to max_time_index if specified
    if max_time_index is not None:
        time_steps_limited = time_steps[:max_time_index+1]
        grouped = [(time, group) for time, group in melted_df.groupby('Time') if time in time_steps_limited]
    else:
        grouped = list(melted_df.groupby('Time'))

    args_list = [(time, group, obs_lats, obs_lons, coverage_radius_km) for time, group in grouped]

    cover_list = []

    num_cores = multiprocessing.cpu_count()
    print(f"Number of CPU cores available for data processing: {num_cores}")

    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(compute_cover_for_time, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            cover_list.extend(result)
    return cover_list

def minimize_observation_points(cover_list, satellite_names, observation_points, time_steps):
    # Create SCIP model
    model = Model("Satellite_Coverage_Problem")
    model.hideOutput()  # Hide solver output

    num_obs = len(observation_points)

    # Binary decision variables for observation points
    x_vars = {j: model.addVar(vtype="B", name=f"x_{j}") for j in range(num_obs)}

    # Build coverage dictionary: satellite -> set of observation points
    sat_to_obs = {sat_name: set() for sat_name in satellite_names}
    for sat_name, obs_idx, _ in cover_list:
        sat_to_obs[sat_name].add(obs_idx)

    # Remove satellites that cannot be covered
    satellites_with_no_coverage = [sat for sat in satellite_names if not sat_to_obs[sat]]
    if satellites_with_no_coverage:
        print(f"The following satellites cannot be covered: {satellites_with_no_coverage}")
        return False, None, None

    # Constraints: Each satellite must be covered by at least one selected observation point
    for sat_name in satellite_names:
        obs_indices = sat_to_obs[sat_name]
        model.addCons(sum(x_vars[obs_idx] for obs_idx in obs_indices) >= 1,
                      f"Cover_{sat_name}")

    # Objective: Minimize the number of observation points
    model.setObjective(sum(x_vars[j] for j in range(num_obs)), "minimize")

    # Solve the model
    print("Solving the model using SCIP to minimize the number of observation points...")
    model.optimize()

    if model.getStatus() == "optimal":
        # Extract selected observation points
        selected_points = [j for j in range(num_obs) if model.getVal(x_vars[j]) > 0.5]
        min_k = len(selected_points)
        return True, min_k, selected_points
    else:
        return False, None, None

def main():
    sat_positions_file = 'satellite_positions.csv'
    obs_points_file = 'circle_centers.csv' # possible attack points
    common_satellites_file = 'common_starlink_satellites.csv'
    output_file = 'result.csv'

    print("Reading satellite names...")
    common_starlink_satellites = read_common_satellites(common_satellites_file)
    print(f"Total satellites available: {len(common_starlink_satellites)}")

    # Use all satellites
    satellite_names = common_starlink_satellites

    print("Reading satellite positions...")
    time_step_interval = 1  # Every minute
    time_steps, satellite_names, sat_positions_df = read_satellite_positions(
        sat_positions_file, satellite_names, time_step_interval=time_step_interval
    )
    print(f"Loaded {len(satellite_names)} satellites, {len(time_steps)} time steps.")

    if not satellite_names:
        print("No matching satellites found in the position matrix. Please check the files or satellite names.")
        return

    print("Reading observation points (only high-seas-based circle centers)...")
    observation_points = read_observation_points(obs_points_file)
    print(f"Loaded {len(observation_points)} observation points in high seas.")

    print("Preprocessing satellite positions...")
    melted_df = preprocess_satellite_positions(sat_positions_df)

    # Initialize results list
    results = []

    # Loop through T_hours from 1 to 5 in 0.5-hour increments
    for T_hours in np.arange(5, 6, 0.5):
        print(f"\nComputing coverage matrix for T = {T_hours} hours...")
        time_step_interval_minutes = 1  # 1-minute intervals
        steps_per_hour = 60 // time_step_interval_minutes
        T_steps = int(T_hours * steps_per_hour)

        # Limit time steps up to T_steps
        max_time_index = T_steps - 1  # zero-based index

        cover_list = compute_cover_matrix_parallel(
            observation_points, melted_df, time_steps,
            coverage_radius_km=1000, max_time_index=max_time_index
        )

        if not cover_list:
            print(f"No coverage possible within T = {T_hours} hours.")
            continue

        feasible, min_k, selected_points = minimize_observation_points(
            cover_list, satellite_names, observation_points, time_steps
        )
        if feasible:
            print(f"Feasible solution found with T = {T_hours} hours")
            print(f"Minimum number of observation points required: {min_k}")
            # Extract selected observation points and their coordinates
            selected_observation_points = [(idx + 1, observation_points[idx]) for idx in selected_points]
            for idx, (obs_idx, (lat, lon)) in enumerate(selected_observation_points):
                print(f"Point {idx+1}: Index {obs_idx}, Latitude {lat}, Longitude {lon}")

            # Append results
            for idx, (obs_idx, (lat, lon)) in enumerate(selected_observation_points):
                results.append({
                    'T_hours': T_hours,
                    'Observation_Point_Index': obs_idx,  # Adjusted to 1-based index
                    'Latitude': lat,
                    'Longitude': lon,
                    'Observation_Points': min_k
                })
        else:
            print(f"No feasible solution found with T = {T_hours} hours.")

    # Save all results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"Results saved to '{output_file}'")


if __name__ == '__main__':
    main()
