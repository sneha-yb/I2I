import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv
from scipy.signal import find_peaks

data_structure = []

os.environ["CUDA_VISIBLE_DEVICES"]="0"

gpu_device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fft_with_window_gpu(mat, axis=-1, shift_flag=False): # axis on adc samples
    mat_shape = mat.shape
    dim = np.ones_like(mat_shape, dtype=int)
    dim[axis] = mat_shape[axis]
    wd = torch.hamming_window(mat_shape[axis], periodic=False, alpha=0.54, beta=0.46, dtype=torch.float32).reshape(tuple(dim)).to(gpu_device)
    # print('mat*wd', mat * wd)
    result = torch.fft.fft(mat * wd, dim=axis)
    if shift_flag:
        result = torch.fft.fftshift(result, dim=axis)
    # print('result', result)
    return result


def fft_with_padding(mat, dim, size, shift_flag=False):
    result=torch.fft.fft(mat, dim=dim, n=size)
    if shift_flag:
        result=torch.fft.fftshift(result, dim=dim)
    return result


def get_time_range(data, trial):
    length_size, tx_size, rx_size, chirp_loops, adc_samples = data.shape #length size is frame size

    data = torch.from_numpy(np.asarray(data))
    print(data.shape)

    data = data.view(length_size, tx_size, rx_size, chirp_loops, adc_samples)
    data = torch.permute(data, (1, 2, 0, 3, 4)).view(tx_size * rx_size, length_size, chirp_loops, adc_samples)[0:1] # 1st tx-rx only, work as [0] but remain the original dimentions
    with torch.no_grad():
        data = data.cfloat().to(gpu_device)
        frames_tensor = data
        frames_range = fft_with_window_gpu(frames_tensor, axis=-1, shift_flag=False)
        
        # frames range shape: torch.Size([1, 50, 128, 256]), each range bin contains specific amplitude information
        
        fs = 256/(59*10e-6)
        freq_slope = 60.012 * 1e+12
        freq_resolution = fs/256
        delta_r_cm = 100*(freq_resolution * 299792458 / (2 * freq_slope))
        
        
        #print(delta_r_cm)
        
        range_real = np.arange(frames_range.numpy().shape[-1]) * delta_r_cm
        avg_amplitude = np.mean(np.abs(frames_range.numpy()), axis=(0,1,2))

        
        range_plt = []
        amplitude_plt = []
        for i in range(len(range_real)):
            if range_real[i] <= 20:
                range_plt.append(range_real[i])
                amplitude_plt.append(avg_amplitude[i])
        
        '''
        sf="./FFT_figure"
        if not os.path.isdir(sf):
            os.mkdir(sf)
        tpath = '{}/{}.png'.format(sf,trial)
        plt.figure(figsize=(8, 5))
        plt.plot(range_plt, amplitude_plt, label="Avg Magnitude")
        #plt.plot(range_real, avg_amplitude, label="Avg Magnitude")
        plt.xlabel("Distance (cm)")
        plt.ylabel("Magnitude")
        plt.title("Distance vs Magnitude_{}".format(trial))
        plt.legend()
        plt.grid(True)
        plt.savefig(tpath)
        #plt.show()
        '''

        range_filtered = []
        amplitude_filtered = []
        for i in range(len(range_real)):
            if 11 < range_real[i] < 20:
                range_filtered.append(range_real[i])
                amplitude_filtered.append(avg_amplitude[i])
       

        return range_filtered, amplitude_filtered


def load_sync_list(infilename):
    sync_list = []
    csv_file = pd.read_csv(infilename, header=0)
    print(infilename)
    for row_index in range(csv_file.shape[0]):
        trail_name = csv_file["trial_name"][row_index]
        sync_list.append(trail_name)
    return sync_list

def compute_frequency_entropy(frames_range, num_freq_bins=10):
    """
    Computes spectral entropy across frequency bands (distance bins).

    Args:
        frames_range (torch.Tensor): FFT magnitude data (Shape: [bins, time])
        num_freq_bins (int): Number of frequency bins to divide the spectrum.

    Returns:
        frequency_entropy (np.ndarray): Entropy values per frequency bin.
    """
    print("Input frames_range shape:", frames_range.shape)
    power_spectrum = torch.abs(frames_range) ** 2
    prob_distribution = power_spectrum / (torch.sum(power_spectrum, dim=-1, keepdim=True) + 1e-10)
    spectral_entropy = -torch.sum(prob_distribution * torch.log2(prob_distribution + 1e-10), dim=-1)

    freq_bin_size = spectral_entropy.shape[0] // num_freq_bins
    frequency_entropy = [
        spectral_entropy[i * freq_bin_size: (i + 1) * freq_bin_size].mean().item()
        for i in range(num_freq_bins)
    ]

    return np.array(frequency_entropy)

def extract_frequency_entropy_features(frames_range, trial, brix_value, num_freq_bins=10):
    """
    Extracts only std_entropy and std_power features and saves them as a CSV.
    """
    frequency_entropy = compute_frequency_entropy(frames_range, num_freq_bins=num_freq_bins)
    std_entropy_freq = np.std(frequency_entropy)

    # Power-based features
    all_selected_frames = frames_range.unsqueeze(0) if len(frames_range.shape) == 2 else frames_range
    avg_power_per_bin = torch.mean(torch.abs(all_selected_frames), dim=(0, 1))
    std_power = np.std(avg_power_per_bin.cpu().numpy())

    # Feature dictionary with only the 2 features
    feature_dict = {
        "trial": trial,
        "brix_value": brix_value,
        "std_entropy": std_entropy_freq,
        "std_power": std_power
    }

    return feature_dict


def compute_and_plot_frequency_entropy(frames_range, trial, brix_value, num_freq_bins=10):
    """
    Computes and plots spectral entropy across the full frequency domain (all distance bins).

    Args:
        frames_range (torch.Tensor): FFT data [tx_rx, time, range_bins]
        trial (str): Trial name
        brix_value (float): Brix value for labeling and saving
        num_freq_bins (int): Number of frequency bins (optional grouping)
    """
    # Use full range
    valid_indices = np.arange(frames_range.shape[-1])
    selected_frames = frames_range[..., valid_indices]

    # Noise reduction: Zero out low-power bins
    magnitude = torch.abs(selected_frames)
    threshold = torch.quantile(magnitude, 0.2)
    selected_frames = torch.where(magnitude < threshold, torch.tensor(0.0, device=magnitude.device), selected_frames)

    # Compute entropy
    frequency_entropy = compute_frequency_entropy(selected_frames[0], num_freq_bins=num_freq_bins)

    '''
    # Plot: use Frequency Bin Index
    bin_indices = np.arange(num_freq_bins)
    bin_labels = [f"Bin {i+1}" for i in bin_indices]
    
    plt.figure(figsize=(8, 5))
    plt.plot(bin_indices, frequency_entropy, marker='o')
    plt.xticks(ticks=bin_indices, labels=bin_labels)
    plt.xlabel("Frequency Bin Index")
    plt.ylabel("Entropy")
    plt.title(f"Spectral Entropy Across Full Range Bins ({trial})")
    plt.grid(True)

    # Save the plot
    sf = "/Users/gau/Downloads/Fruit Ripeness Files/Frequency Plots/Underripe Plots"
    os.makedirs(sf, exist_ok=True)
    plt.savefig(f"{sf}/{trial}_frequency_entropy_fullrange.png")
    # plt.show()
    '''
    # Save features
    extract_frequency_entropy_features(selected_frames[0], trial, brix_value, num_freq_bins=num_freq_bins)


if __name__=='__main__':
    root_loc="./raw_radar_frames/"
    save_folder="./hm_dra_modified_length_0-100"
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)


    trial_list=load_sync_list("./sync_file.csv")


    start_frame=0
    frame_size=50 # num frame
    #csv_filename = 'fruit_features.csv'
    csv_filename = 'demo_test.csv'

    # Open the CSV file in write mode
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow([
            "Trial",
            #"Max_Magnitude",
            #"Range_At_Max",
            "Mean_Magnitude",
            #"Std_Deviation_Magnitude",
            #"AUC",
            #"1st_Peak_Mag",
            #"1st_Peak_Mag_Range",
            "2nd_Peak_Mag",
            #"2nd_Peak_Mag_Range",
            "3rd_Peak_Mag",
            #"3rd_Peak_Mag_Range",
            "std_entropy",
            "std_power"
        ])

        for trial in trial_list:
            inloc = os.path.join(root_loc, trial)
            outfolder = os.path.join(save_folder, trial)
           
            if not os.path.isdir(outfolder):
                os.mkdir(outfolder)
            data = [np.load(os.path.join(inloc, '%04d.npy' % i)) for i in range(start_frame, start_frame + frame_size)]
            
            data = np.asarray(data)

            range_filtered, amp_filtered = get_time_range(data, trial)
            range_filtered = np.array(range_filtered)
            amp_filtered = np.array(amp_filtered)

            max_mag = np.max(amp_filtered)
            range_at_max_mag = range_filtered[np.argmax(amp_filtered)]
            mean_mag = np.mean(amp_filtered)
            std_mag = np.std(amp_filtered)

            # Area under curve
            auc = np.trapezoid(amp_filtered, range_filtered)

            # peak magnitudes and ranges
            peaks, _ = find_peaks(amp_filtered)
            peak_magnitudes = amp_filtered[peaks]
            peak_ranges = range_filtered[peaks]
            sorted_indices = np.argsort(peak_magnitudes)[::-1]
            top_n = 3
            top_peaks_mags = np.zeros(top_n)
            top_peaks_ranges = np.zeros(top_n)


            for i in range(min(top_n, len(sorted_indices))):
                idx = sorted_indices[i]
                top_peaks_mags[i] = peak_magnitudes[idx]
                top_peaks_ranges[i] = peak_ranges[idx]

            data_tensor = torch.from_numpy(data).to(gpu_device).cfloat()  # shape [50, 3, 4, 128, 256]
            data_tensor = data_tensor.view(frame_size, 3, 4, 128, 256)
            data_tensor = torch.permute(data_tensor, (1, 2, 0, 3, 4)).view(3 * 4, frame_size, 128, 256)[0:1]  # [1, 50, 128, 256]
            frames_range = fft_with_window_gpu(data_tensor, axis=-1, shift_flag=False)  # [1, 50, 128, 256]
            frames_range_for_entropy = torch.mean(frames_range[0], dim=0).transpose(0, 1)
            entropy_features = extract_frequency_entropy_features(frames_range_for_entropy, trial, brix_value=0.0, num_freq_bins=10)

            # Write the features to the CSV file
            writer.writerow([
                trial,
                #max_mag,
                #range_at_max_mag,
                mean_mag,
                #std_mag,
                #auc,
                #top_peaks_mags[0],
                #top_peaks_ranges[0],
                top_peaks_mags[1],
                #top_peaks_ranges[1],
                top_peaks_mags[2],
                #top_peaks_ranges[2],
                entropy_features["std_entropy"],
                entropy_features["std_power"]
            ])