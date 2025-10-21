import pyarrow as pa
import pyarrow.ipc as ipc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def process_single_file(file_path):
    """Processes a single arrow file and returns its local sum and count."""
    local_sum = 0
    local_count = 0
    try:
        with pa.memory_map(file_path, 'r') as source:
            try:
                reader = ipc.open_stream(source)
            except pa.ArrowInvalid:
                source.seek(0)
                reader = ipc.open_file(source)
            
            # Check if column exists in the schema before iterating
            if 'nonpad_tokens' not in reader.schema.names:
                # print(f"Warning: 'nonpad_tokens' column not found in {os.path.basename(file_path)}")
                return 0, 0

            for batch in reader:
                local_sum += batch.column('nonpad_tokens').sum().as_py()
                local_count += len(batch.column('nonpad_tokens'))
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return 0, 0
        
    return local_sum, local_count

def calculate_mean_nonpad_tokens_v2(directory):
    """
    Calculates the mean by processing files in parallel.
    """
    arrow_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.arrow')]
    if not arrow_files:
        print(f"No .arrow files found in directory: {directory}")
        return 0.0

    print(f"Found {len(arrow_files)} .arrow files. Processing in parallel...")

    total_sum = 0
    total_count = 0

    # Use a process pool to parallelize file processing
    with ProcessPoolExecutor() as executor:
        # Create a future for each file processing task
        futures = [executor.submit(process_single_file, fp) for fp in arrow_files]
        
        # Use tqdm to show progress as tasks complete
        for future in tqdm(as_completed(futures), total=len(arrow_files), desc="Processing files"):
            local_sum, local_count = future.result()
            total_sum += local_sum
            total_count += local_count

    if total_count == 0:
        print("No 'nonpad_tokens' data found in any of the files.")
        return 0.0

    mean_value = total_sum / total_count
    return mean_value


directory = '/data/yqw/bd3lms-alpha/data/DrugLikeSMILSE_packed1024_butina_s055_blocks10k/train/'
calculate_mean_nonpad_tokens_v2(directory)