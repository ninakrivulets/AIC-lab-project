import pickle
import pandas as pd
import os
import pyopenms as oms
import logging
import uuid
import base64

import shutil

# folder for saving
output_folder = '/blob/dda/PXD028806/training_data/'
#output_folder = '/home/ninak/train_data_del'
#exp_dir = '/home/ninak/train_data_logs_del'
exp_dir = '/blob/dda/PXD028806/train_data_logs'
os.makedirs(exp_dir, exist_ok=True)

os.makedirs(output_folder, exist_ok=True)
#file_path = '/home/ninak/tailor.assign-confidence.filtered2.txt'
file_path = '/blob/dda/PXD028806/tailor.assign-confidence.filtered.txt' 
df = pd.read_csv(file_path, sep='\t')

df = df[df['tdc q-value'] < 0.05]

# Logger setup
'''

def get_logger(exp_id, log_path):
    logger = logging.getLogger(f"Experiment_{exp_id}")
    logger.setLevel(logging.INFO)
    # creating handlers
    file_handler = logging.FileHandler(f"{log_path}/experiment_{exp_id}.log")
    console_handler = logging.StreamHandler()
    # Set formatter for handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
'''
def get_logger(exp_id, log_path):
    #logger_name = f"Experiment_{exp_id}"
    #logger = logging.getLogger(logger_name)
    logger = logging.getLogger('main')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f"{log_path}/experiment_{exp_id}.log")
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def extract_scannum(native_id, plogger):
    try:
        parts = native_id.split('scan=')
        if len(parts) > 1:
            return int(parts[1])
    except Exception as e:
        print(f"Error extracting scannum from NativeID {native_id}: {e}")
        plogger.info(f"Error extracting scannum from NativeID {native_id}: {e}")
    return None

def parse_protein_id(protein_id_str, seq_len):
    protein_dict = {}
    proteins = protein_id_str.split(',')  # split into several ids
    for protein in proteins:
        try:
            key, start = protein.split('(')
            start = int(start[:-1])  # Remove the closing brackets and convert to int
            protein_dict[key] = {
                'start': start,
                'end': start + seq_len
            }
        except Exception as e:
            print(f"Error parsing protein ID {protein}: {e}")
    return protein_dict

def save_partial_data(data, file_counter, plogger):
    output_file = os.path.join(output_folder, f"PXD028806_tailor_{file_counter}.pkl")
    try:
        with open(output_file, "wb") as f:
            data = pd.DataFrame(data)
            pickle.dump(data, f)
        print(f"Saved {len(data)} rows to {output_file}")
        plogger.info(f"Saved {len(data)} rows to {output_file}") 
    except Exception as e:
        print(f"Error while saving to {output_file}: {e}")
        plogger.info(f"Error while saving to {output_file}: {e}") 


def generate_uuid():
    """Generates a uuid 4 string, in this context for tracking each run of the experiment
    Returns:
        an ascii friendly uuid4 string.
    """
    return base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")

def get_mz_and_intensities(df, spectra_per_file=1000):
    file_groups = df.groupby('file')['scan'].apply(list).to_dict()

    processed_spectra = 0  # Counter for total spectra
    file_counter = 1       # Counter for file saving
    results = []           # Temporary storage for results
    exp_id = generate_uuid()
    plogger = get_logger(exp_id=exp_id, log_path=exp_dir)
    plogger.info(f"Experiment ID: {exp_id}") #remove this to get rid of exp id on every line
    plogger.info(f"Input file: {file_path}")
    for mzml_file, scans in file_groups.items():
        # exp_id = generate_uuid
        # plogger = get_logger(exp_id=exp_id, log_path=exp_dir)
        #plogger.info(f"Experiment ID: {exp_id}")
        
        # Checking if the file exists
        if os.path.exists(mzml_file):
            print(f"Processing file: {mzml_file}, scan numbers: {scans}") 
            plogger.info(f"Processing file: {mzml_file}, scan numbers: {scans}")
            
            exp = oms.MSExperiment()
            oms.MzMLFile().load(mzml_file, exp)
            spectra = exp.getSpectra()
            
            for spec in spectra:
                #plogger.info(f"still working for {spec}") 
                native_id = spec.getNativeID()
                scannum = extract_scannum(native_id, plogger)
                if scannum is not None and scannum in scans:
                    row = df[(df['file'] == mzml_file) & (df['scan'] == scannum)]
                    # did not test this part
                    if 'charge' in df.columns:
                        row = row.drop_duplicates(subset=['file', 'scan', 'charge'])

                    if len(row) != 1:
                        plogger.info(f"Skipping scan {scannum} in file {mzml_file} due to non-unique rows.")
                        continue
                    # end of the part
                    if not row.empty:
                        spectrum_neutral_mass = row['spectrum neutral mass'].values[0]
                        peptide_mass = row['peptide mass'].values[0]
                        sequence = row['sequence'].values[0]
                        seq_len = len(sequence)

                        protein_id = row['protein id'].values[0]
                        protein_dict = parse_protein_id(protein_id, seq_len)
                        mz_values = spec.get_peaks()[0]
                        intensities = spec.get_peaks()[1]
                        max_intensity = spec.getMaxIntensity()
                        mask = intensities >= 0.01 * max_intensity
                        filtered_intensities = intensities[mask]
                        intensities_scaled = 100 * filtered_intensities / filtered_intensities.max()  # Rescale intensities
                        mask_2 = intensities_scaled >= 1
                        intensities_scaled_masked = intensities_scaled[mask_2] # remove those less than 1
                        results.append({
                            'file': mzml_file,
                            'scan': scannum,
                            'mz_values': list(mz_values),
                            'intensity': list(intensities_scaled_masked),
                            'spectrum_neutral_mass': spectrum_neutral_mass,
                            'peptide_mass': peptide_mass,
                            'sequence': sequence,
                            'protein_ids': protein_dict
                        })
                        processed_spectra += 1
                        
                        # Save after every `spectra_per_file` rows
                        if len(results) >= spectra_per_file:
                            save_partial_data(results, file_counter, plogger)
                            file_counter += 1
                            results = []  # Clear results for next batch

                           
        else:
            plogger.info(f"File {mzml_file} not found")
            print(f"File {mzml_file} not found")
    
    # Save remaining results
    if results:
        save_partial_data(results, file_counter, plogger)
    plogger.info(f"Processed {processed_spectra} spectra in total.")
    print(f"Processed {processed_spectra} spectra in total.")

get_mz_and_intensities(df, spectra_per_file=1000)


