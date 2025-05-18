import pandas as pd
import aqi # python-aqi library
import os

def calculate_pollutant_aqi(row, pollutant_code, csv_column_name, input_unit, target_unit_for_lib):
    """
    Calculates the AQI for a single pollutant using the python-aqi library.
    Handles unit conversion if necessary.
    """
    try:
        val_str = str(row[csv_column_name]).strip()
        if not val_str or val_str.lower() == 'nan' or val_str.lower() == 'na':
            return None 
        
        val = float(val_str)
        
        converted_val = val
        # Approximate conversions (standard conditions 25C, 1atm)
        # Molar Volume at 25C, 1atm = 24.45 L/mol
        # Molar Masses: O3=48.00, CO=28.01, SO2=64.07, NO2=46.01 g/mol
        if input_unit == 'ug/m3' and target_unit_for_lib == 'ppb':
            if pollutant_code == 'o3': converted_val = val * (24.45 / 48.00)
            elif pollutant_code == 'so2': converted_val = val * (24.45 / 64.07)
            elif pollutant_code == 'no2': converted_val = val * (24.45 / 46.01)
        elif input_unit == 'mg/m3' and target_unit_for_lib == 'ppm':
            if pollutant_code == 'co': converted_val = val * (24.45 / 28.01)
        
        pollutant_aqi_val = aqi.to_iaqi(pollutant_code, str(converted_val), algo=aqi.ALGO_EPA)
        return int(pollutant_aqi_val)
    except ValueError:
        # print(f"Warning: Could not convert value '{row[csv_column_name]}' for {csv_column_name} to float.")
        return None
    except Exception as e:
        # print(f"Warning: Could not calculate AQI for {csv_column_name} with value '{row[csv_column_name]}': {e}")
        return None

def process_city_csv(input_csv_path, output_csv_path, 
                     city_name_for_output, state_name_for_output, country_name_for_output):
    """
    Processes a single city's air quality CSV file:
    1. Renames columns.
    2. Calculates US EPA AQI and adds it as 'aqi_us'.
    3. Adds 'city', 'state', 'country' columns.
    4. Saves the processed data.
    """
    print(f"\nProcessing file: {input_csv_path} for city: {city_name_for_output}")
    try:
        # Attempt to detect encoding, common ones are utf-8 and latin1
        try:
            df = pd.read_csv(input_csv_path, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            print(f"UTF-8 decoding failed for {input_csv_path}, trying latin1...")
            df = pd.read_csv(input_csv_path, low_memory=False, encoding='latin1')
        print(f"Successfully read {input_csv_path} with {len(df)} rows.")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_csv_path}'")
        return False
    except Exception as e:
        print(f"Error reading CSV file '{input_csv_path}': {e}")
        return False

    # 1. Rename 'From Date' to 'timestamp'
    # Also check for common variations like 'Date' or 'datetime' if 'From Date' is not found
    timestamp_col_found = False
    if 'From Date' in df.columns:
        df.rename(columns={'From Date': 'timestamp'}, inplace=True)
        print("Renamed 'From Date' to 'timestamp'.")
        timestamp_col_found = True
    elif 'Date' in df.columns: # Common alternative
        df.rename(columns={'Date': 'timestamp'}, inplace=True)
        print("Renamed 'Date' to 'timestamp'.")
        timestamp_col_found = True
    elif 'datetime' in df.columns: # Another common alternative
        df.rename(columns={'datetime': 'timestamp'}, inplace=True)
        print("Renamed 'datetime' to 'timestamp'.")
        timestamp_col_found = True
    elif 'timestamp' in df.columns: # Already correctly named
        print("'timestamp' column already exists.")
        timestamp_col_found = True
    else:
        print(f"Error: A recognizable timestamp column ('From Date', 'Date', 'datetime', 'timestamp') not found in {input_csv_path}. Please check your CSV header.")
        return False


    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        original_len = len(df)
        df.dropna(subset=['timestamp'], inplace=True)
        if len(df) < original_len:
            print(f"Dropped {original_len - len(df)} rows due to unparseable timestamps.")
    else: 
        print("Error: 'timestamp' column is required. Aborting processing for this file.")
        return False

    # 2. Rename other feature columns
    column_rename_map = {
        'Temp (degree C)': 'temperature_celsius', 'Temp': 'temperature_celsius', 'Temperature': 'temperature_celsius',
        'RH (%)': 'humidity_percent', 'RH': 'humidity_percent', 'Relative Humidity': 'humidity_percent',
        'WS (m/s)': 'wind_speed_mps', 'WS': 'wind_speed_mps', 'Wind Speed': 'wind_speed_mps',
        'PM2.5 (ug/m3)': 'pm25_conc_ug_m3', 'PM2.5': 'pm25_conc_ug_m3',
        'PM10 (ug/m3)': 'pm10_conc_ug_m3', 'PM10': 'pm10_conc_ug_m3',
        'Ozone (ug/m3)': 'o3_conc_ug_m3', 'Ozone': 'o3_conc_ug_m3', 'O3 (ug/m3)': 'o3_conc_ug_m3',
        'CO (mg/m3)': 'co_conc_mg_m3', 'CO': 'co_conc_mg_m3',
        'SO2 (ug/m3)': 'so2_conc_ug_m3', 'SO2': 'so2_conc_ug_m3',
        'NO2 (ug/m3)': 'no2_conc_ug_m3', 'NO2': 'no2_conc_ug_m3'
    }
    actual_renames = {k: v for k, v in column_rename_map.items() if k in df.columns}
    df.rename(columns=actual_renames, inplace=True)
    if actual_renames:
        print(f"Renamed columns: {actual_renames}")

    # 3. Calculate 'aqi_us'
    pollutants_for_aqi = [
        ('pm25_conc_ug_m3', 'pm2.5', 'ug/m3', 'ug/m3'),
        ('pm10_conc_ug_m3', 'pm10', 'ug/m3', 'ug/m3'),
        ('o3_conc_ug_m3', 'o3', 'ug/m3', 'ppb'),
        ('co_conc_mg_m3', 'co', 'mg/m3', 'ppm'),
        ('so2_conc_ug_m3', 'so2', 'ug/m3', 'ppb'),
        ('no2_conc_ug_m3', 'no2', 'ug/m3', 'ppb')
    ]

    print("Calculating AQI sub-indices...")
    aqi_us_column_data = []
    for i, row in df.iterrows():
        row_aqi_values = []
        for csv_col, code, in_unit, target_unit in pollutants_for_aqi:
            if csv_col in df.columns: # Check if the (potentially renamed) column exists
                sub_aqi = calculate_pollutant_aqi(row, code, csv_col, in_unit, target_unit)
                if sub_aqi is not None:
                    row_aqi_values.append(sub_aqi)
        
        if row_aqi_values:
            aqi_us_column_data.append(max(row_aqi_values))
        else:
            aqi_us_column_data.append(None) 
    
    df['aqi_us'] = aqi_us_column_data
    print("Overall 'aqi_us' calculated.")
    
    # 4. Add/Overwrite city, state, country columns
    df['city'] = city_name_for_output
    df['state'] = state_name_for_output
    df['country'] = country_name_for_output
    print(f"Added/Set location columns: city='{city_name_for_output}', state='{state_name_for_output}', country='{country_name_for_output}'.")

    # 5. Select and reorder columns for the final output
    final_expected_cols = [
        'timestamp', 'city', 'state', 'country', 'aqi_us', 
        'temperature_celsius', 'humidity_percent', 'wind_speed_mps',
        'pm25_conc_ug_m3', 'pm10_conc_ug_m3', 'o3_conc_ug_m3', 
        'co_conc_mg_m3', 'so2_conc_ug_m3', 'no2_conc_ug_m3'
    ]
    
    cols_to_keep_in_output = [col for col in final_expected_cols if col in df.columns]
    # Add any other columns from the original df that were not in final_expected_cols but you might want to keep
    # For this script, we'll stick to the defined expected columns for simplicity in training
    df_processed = df[cols_to_keep_in_output].copy()

    missing_aqi_count = df_processed['aqi_us'].isnull().sum()
    if missing_aqi_count > 0:
        print(f"Warning: {missing_aqi_count} rows have no 'aqi_us' calculated in the final output for this city.")

    if os.path.isdir(output_csv_path):
        print(f"Error: Output path '{output_csv_path}' is a directory. Please provide a full file name.")
        return False
        
    try:
        df_processed.to_csv(output_csv_path, index=False)
        print(f"Successfully processed data for {city_name_for_output} saved to '{output_csv_path}'")
        return True
    except Exception as e:
        print(f"Error saving processed CSV file '{output_csv_path}': {e}")
        return False

if __name__ == "__main__":
    # Define your city CSV files and their corresponding location details
    # IMPORTANT: Update these input_path values to match YOUR actual filenames
    city_files_to_process = [
        {
            "input_path": "DL001.csv", # User provided filename
            "output_path": "delhi_for_training.csv",
            "city": "Delhi", "state": "Delhi", "country": "India"
        },
        {
            "input_path": "MH002.csv", # User provided filename
            "output_path": "mumbai_for_training.csv",
            "city": "Mumbai", "state": "Maharashtra", "country": "India"
        },
        {
            "input_path": "KA001.csv", # User provided filename
            "output_path": "bengaluru_for_training.csv",
            "city": "Bengaluru", "state": "Karnataka", "country": "India"
        },
        {
            "input_path": "TN001.csv", # User provided filename
            "output_path": "chennai_for_training.csv",
            "city": "Chennai", "state": "Tamil Nadu", "country": "India"
        },
        {
            "input_path": "WB002.csv", # User provided filename
            "output_path": "kolkata_for_training.csv",
            "city": "Kolkata", "state": "West Bengal", "country": "India"
        }
    ]

    all_successful = True
    processed_files_info = [] # To store info for main.py

    for city_spec in city_files_to_process:
        if not os.path.exists(city_spec["input_path"]):
            print(f"Skipping {city_spec['input_path']}: File not found.")
            continue 

        success = process_city_csv(
            input_csv_path=city_spec["input_path"],
            output_csv_path=city_spec["output_path"],
            city_name_for_output=city_spec["city"],
            state_name_for_output=city_spec["state"],
            country_name_for_output=city_spec["country"]
        )
        if not success:
            all_successful = False
        else:
            processed_files_info.append({
                "filepath": city_spec["output_path"],
                "city": city_spec["city"],
                "state": city_spec["state"],
                "country": city_spec["country"]
            })

    
    if all_successful and processed_files_info:
        print("\nAll specified CSV files processed successfully (if found).")
    elif processed_files_info: # Some processed successfully
        print("\nSome CSV files were processed successfully. Others may have failed or were not found. Please check logs.")
    else:
        print("\nNo CSV files were processed successfully. Please check the logs and file paths.")
    
    print("\nNext Steps:")
    print("1. Ensure the output CSV files (e.g., 'delhi_for_training.csv', etc.) are in the same directory as your main.py backend script.")
    print("2. Update the 'PREVIOUS_CITY_DATA_FILES' list in your 'main.py' backend script to look like this (using your output filenames):")
    print("   PREVIOUS_CITY_DATA_FILES = [")
    for info in processed_files_info:
        print(f"       {{")
        print(f"           \"filepath\": \"{info['filepath']}\",")
        print(f"           \"city\": \"{info['city']}\", \"state\": \"{info['state']}\", \"country\": \"{info['country']}\"")
        print(f"       }},")
    print("   ]")
    print("3. After updating main.py, restart your backend and trigger model retraining via the API.")

