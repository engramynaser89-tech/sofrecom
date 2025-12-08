import os
import re
import pandas as pd
import logging

# === Logging Configuration ===
def setup_logger(log_path):
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler())

# === Coordinate Conversion ===
def to_decimal_degrees(coord_str):
    if isinstance(coord_str, (int, float)):
        return float(coord_str)
    coord_str = str(coord_str).strip()
    if not coord_str:
        raise ValueError("Empty coordinate")
    coord_str = coord_str.replace("°", "").replace("º", "")
    coord_str = re.sub(r'[\t\s]+', ' ', coord_str).strip()
    if re.fullmatch(r'-?\d+(\.\d+)?', coord_str):
        return float(coord_str)
    match = re.fullmatch(r'(-?\d+)\s+(\d+(\.\d+)?)', coord_str)
    if match:
        degrees = float(match.group(1))
        minutes = float(match.group(2))
        return degrees + (minutes / 60)
    raise ValueError(f"Unsupported coordinate format: '{coord_str}'")

# === Header Detection ===
def find_column_indices(header_row, target_columns):
    col_map = {}
    for idx, val in enumerate(header_row):
        val_str = str(val).strip().lower()
        for target in target_columns:
            if target.lower() in val_str:
                col_map[target] = idx
                break
    return col_map

# === Metadata Extraction ===
def extract_metadata(df):
    metadata = {"Operator Name": None, "Place ID": None, "District": None, "Place": None}
    try:
        metadata["Operator Name"] = str(df.iat[1, 2]).strip() if not pd.isna(df.iat[1, 2]) else None
        metadata["Place ID"] = str(df.iat[13, 1]).strip() if not pd.isna(df.iat[13, 1]) else None
        metadata["District"] = str(df.iat[13, 2]).strip() if not pd.isna(df.iat[13, 2]) else None
        metadata["Place"] = str(df.iat[13, 5]).strip() if not pd.isna(df.iat[13, 5]) else None
    except Exception as e:
        logging.warning(f"Metadata extraction failed: {e}")
    return metadata

# === Row Extraction ===
def extract_rows_by_indices(df, header_row_idx, col_indices, file_name, metadata, work_order, operator, place_name_idx=2, max_rows=100):
    records = []
    empty_count = 0

    for i in range(header_row_idx + 1, len(df)):
        row = df.iloc[i]
        record = {}

        for col, idx in col_indices.items():
            if idx < len(row):
                record[col] = row.iloc[idx]

        place = row.iloc[place_name_idx] if place_name_idx < len(row) else ""
        if not isinstance(place, str) or not place.strip():
            empty_count += 1
            if empty_count >= 5:
                break
            continue

        record["Place Name"] = place.strip()
        record["File Name"] = file_name
        record["Work Order"] = work_order
        record["Operator"] = operator

        # Add metadata
        for k, v in metadata.items():
            record[k] = v

        try:
            lon = to_decimal_degrees(record.get("Longitude", ""))
            lat = to_decimal_degrees(record.get("Latitude", ""))
        except Exception as e:
            logging.warning(f"Coordinate parse error in {file_name}: {e}")
            empty_count += 1
            if empty_count >= 5:
                break
            continue

        if not any([record.get("Date"), record.get("Time"), record.get("Serving ID")]):
            empty_count += 1
            if empty_count >= 5:
                break
            continue

        record["Longitude"] = lon
        record["Latitude"] = lat
        records.append(record)
        empty_count = 0

        if len(records) >= max_rows:
            break

    return pd.DataFrame(records)

# === Main Recursive Search ===
def search_all_wo_folders(root_folder, search_segment):
    combined_data = []
    log_path = os.path.join(root_folder, "extraction_log.txt")
    setup_logger(log_path)

    logging.info(f"Starting extraction under root: {root_folder}")
    target_cols = ['Longitude', 'Latitude', 'Date', 'Time', 'Serving ID']

    # Identify folders like "WO1_STC"
    wo_folders = [
        f for f in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, f)) and re.match(r'^WO\d+_[A-Za-z]+$', f)
    ]

    if not wo_folders:
        logging.error("No valid Work Order folders found.")
        return

    for folder_name in wo_folders:
        work_order, operator = folder_name.split("_", 1)
        folder_path = os.path.join(root_folder, folder_name)
        logging.info(f"Processing folder: {folder_name} (Work Order={work_order}, Operator={operator})")

        # Walk recursively
        for dirpath, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.xls', '.xlsx', '.xlsm', '.xlsb')) and not file.startswith('~$'):
                    file_path = os.path.join(dirpath, file)
                    try:
                        xls = pd.ExcelFile(file_path)
                        for sheet_name in xls.sheet_names:
                            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

                            # Look for sheet containing "Selected Test Points"
                            mask = df.apply(lambda row: search_segment in ' '.join(row.dropna().astype(str)), axis=1)
                            if not mask.any():
                                continue

                            header_row_idx = None
                            col_indices = {}
                            for i in range(min(20, len(df))):
                                row = df.iloc[i].astype(str).str.lower()
                                if any(tc.lower() in ' '.join(row.values) for tc in target_cols):
                                    col_indices = find_column_indices(row, target_cols)
                                    if len(col_indices) >= 3:
                                        header_row_idx = i
                                        break

                            if header_row_idx is not None:
                                metadata = extract_metadata(df)
                                extracted_df = extract_rows_by_indices(
                                    df,
                                    header_row_idx,
                                    col_indices,
                                    file_name=file,
                                    metadata=metadata,
                                    work_order=work_order,
                                    operator=operator
                                )
                                if not extracted_df.empty:
                                    combined_data.append(extracted_df)
                                    logging.info(f"✔ Extracted {len(extracted_df)} rows from {file_path} ({sheet_name})")
                            else:
                                logging.warning(f"No valid header found in {file_path} ({sheet_name})")

                    except Exception as e:
                        logging.error(f"Error processing {file_path}: {e}")

    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        final_df.dropna(subset=["Longitude", "Latitude", "Place Name"], inplace=True)
        final_df = final_df[
            final_df["Longitude"].apply(lambda x: isinstance(x, float)) &
            final_df["Latitude"].apply(lambda x: isinstance(x, float))
        ]

        output_path = os.path.join(root_folder, "test_points_all_v3.csv")
        final_df.to_csv(output_path, index=False)
        logging.info(f"\n✅ Exported {len(final_df)} rows to: {output_path}")
    else:
        logging.warning("🚫 No valid data extracted from any folder.")

# === Configuration ===
search_text = "Selected Test Points"
root_folder = r"C:\operators\Zain\submissions\WO3\FMR"  # adjust as needed

if __name__ == "__main__":
    search_all_wo_folders(root_folder, search_text)