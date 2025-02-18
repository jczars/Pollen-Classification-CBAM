import logging
import os
import psutil
import nvidia_smi
import gc
import tensorflow as tf
from tensorflow.keras.backend import clear_session

def initialize_tf_gpus():
    """
    Configures TensorFlow to enable memory growth for GPUs before any operation.
    Must be called immediately after importing TensorFlow.
    """
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)  # Enable memory growth
                print(f"Enabled memory growth for device: {device.name}")
            except RuntimeError as e:
                print(f"Could not set memory growth for device {device.name}: {e}")
    else:
        print("No GPU devices found.")

def get_cpu_memory_usage():
    """Retrieves CPU memory usage information."""
    memory_info = psutil.virtual_memory()
    return memory_info.used / (1024 ** 3), memory_info.available / (1024 ** 3)  # Returns in GB

def get_gpu_memory_usage():
    """Retrieves GPU memory usage information using nvidia-smi."""
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using GPU 0
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    used_memory = info.used / (1024 ** 2)  # Convert to MB
    free_memory = info.free / (1024 ** 2)  # Convert to MB
    total_memory = info.total / (1024 ** 2)  # Convert to MB
    return used_memory, free_memory, total_memory

def reset_keras():
    """Clears Keras session and performs garbage collection."""
    clear_session()  # Clears the Keras session
    print("Cleared Keras session.")
    gc.collect()  # Forces garbage collection
    print("Garbage collection complete.")

def monitor_memory_and_run():
    """
    Monitors memory usage, logs information, and performs actions accordingly.
    Ensures sufficient GPU memory is available before proceeding.
    """
    used_gpu_mem, free_gpu_mem, total_gpu_mem = get_gpu_memory_usage()
    used_cpu_mem, free_cpu_mem = get_cpu_memory_usage()

    logging.info(f"CPU Memory - Used: {used_cpu_mem:.2f} GB, Available: {free_cpu_mem:.2f} GB")
    logging.info(f"GPU Memory - Used: {used_gpu_mem:.2f} MB, Free: {free_gpu_mem:.2f} MB, Total: {total_gpu_mem:.2f} MB")

    print('\n')
    print("*"*90)
    print(f"CPU Memory - Used: {used_cpu_mem:.2f} GB, Available: {free_cpu_mem:.2f} GB")
    print(f"GPU Memory - Used: {used_gpu_mem:.2f} MB, Free: {free_gpu_mem:.2f} MB, Total: {total_gpu_mem:.2f} MB")

    # Check if GPU memory is at least 60% free
    if free_gpu_mem / total_gpu_mem * 100 < 60:
        logging.info("GPU memory usage is above threshold. Attempting to clear memory.")
        print("GPU memory usage is above the threshold. Attempting to clear memory.")

        reset_keras()  # Clears session and attempts garbage collection

        # Recheck GPU memory usage after cleanup
        used_gpu_mem, free_gpu_mem, total_gpu_mem = get_gpu_memory_usage()
        logging.info(f"Post-Cleanup GPU Memory - Used: {used_gpu_mem:.2f} MB, Free: {free_gpu_mem:.2f} MB")
        print(f"Post-Cleanup GPU Memory - Used: {used_gpu_mem:.2f} MB, Free: {free_gpu_mem:.2f} MB")

        if free_gpu_mem / total_gpu_mem * 100 < 60:
            logging.error("Insufficient GPU memory available after cleanup. Exiting.")
            print("Insufficient GPU memory available after cleanup. Exiting.")
            print("*"*90)
            print('\n')
            
            return

    logging.info("Sufficient GPU memory available. Proceeding with program execution.")
    print("Sufficient GPU memory available. Proceeding with program execution.")
    print("*"*90)
    print('\n')

def log_memory_usage(variable_name):
    """
    Log the memory usage before and after removing a variable.
    
    Args:
        variable_name (str): The name of the variable being deleted.
    """
    # Get memory usage before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
    
    logging.info(f"[INFO] Memory usage before deleting {variable_name}: {mem_before:.2f} MB")
    print(f"[INFO] Memory usage before deleting {variable_name}: {mem_before:.2f} MB")
    
    # Garbage collection and memory cleanup
    gc.collect()
    
    # Get memory usage after
    mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
    logging.info(f"[INFO] Memory usage after deleting {variable_name}: {mem_after:.2f} MB")
    print(f"[INFO] Memory usage after deleting {variable_name}: {mem_after:.2f} MB")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='memory_log.log', filemode='w')

if __name__ == "__main__":
    # Initialize GPU settings before any TensorFlow operation
    initialize_tf_gpus()
    monitor_memory_and_run()
