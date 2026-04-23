import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.interpolate import interp1d

def extract_time_series(file_path, frame_size=2048, hop_length=512):
    """Extracts time-series features from an audio file"""
    try:
        y, sr = librosa.load(file_path)
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None
    
    # Extract features
    centroid_series = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
    rms_series = librosa.feature.rms(y=y, frame_length=frame_size, hop_length=hop_length)[0]
    
    # Harmonic-Percussive Separation for noisiness
    y_harm, y_perc = librosa.effects.hpss(y)
    noise_rms_series = librosa.feature.rms(y=y_perc, frame_length=frame_size, hop_length=hop_length)[0]
    noisiness_series = noise_rms_series / (rms_series + 1e-6)
    
    times = librosa.frames_to_time(np.arange(len(centroid_series)), sr=sr, hop_length=hop_length)
    
    return times, centroid_series, rms_series, noisiness_series

def find_peak_index(series):
    """Find index of the highest peak"""
    return np.argmax(series)

def align_to_peak(times, series, peak_idx, pad_before=100, pad_after=200):
    """Align series to its peak and pad to fixed length"""
    # Pad the series if needed
    padded = np.pad(series, (pad_before, pad_after), mode='constant')
    peak_idx_padded = peak_idx + pad_before
    
    # Crop around the peak
    start = peak_idx_padded - pad_before
    end = peak_idx_padded + pad_after + 1
    aligned = padded[start:end]
    
    return aligned

def calculate_area_under_curve(series):
    """Calculate normalized area under the curve"""
    normalized = series / (np.max(series) + 1e-6)
    return np.sum(normalized)

def calculate_overlap_percent(series1, series2):
    """Calculate overlap percentage between two series"""
    min_series = np.minimum(series1, series2)
    max_series = np.maximum(series1, series2)
    
    intersection = np.sum(min_series)
    union = np.sum(max_series)
    
    if union == 0:
        return 0.0
    return (intersection / union) * 100

def build_overlap_matrix(data_dict):
    """Build overlap percentage matrix for all pairs"""
    n_samples = len(data_dict)
    overlap_matrix = np.zeros((n_samples, n_samples))
    
    keys = list(data_dict.keys())
    for i in range(n_samples):
        for j in range(n_samples):
            if i == j:
                overlap_matrix[i, j] = 100.0
            else:
                # Average overlap across all three features
                overlaps = []
                for feature in ['centroid', 'energy', 'noisiness']:
                    overlaps.append(calculate_overlap_percent(
                        data_dict[keys[i]][feature],
                        data_dict[keys[j]][feature]
                    ))
                overlap_matrix[i, j] = np.mean(overlaps)
    
    return overlap_matrix, keys

def plot_aligned_samples(file1_path, file2_path, output_filename="aligned_comparison.png"):
    """Plot two aligned samples for comparison"""
    data1 = extract_time_series(file1_path)
    data2 = extract_time_series(file2_path)
    
    if not data1 or not data2:
        print("Could not load one or both files")
        return
    
    times1, centroid1, energy1, noisiness1 = data1
    times2, centroid2, energy2, noisiness2 = data2
    
    # Find peaks in energy
    peak_idx1 = find_peak_index(energy1)
    peak_idx2 = find_peak_index(energy2)
    
    # Align
    pad_before, pad_after = 50, 100
    aligned_centroid1 = align_to_peak(times1, centroid1, peak_idx1, pad_before, pad_after)
    aligned_centroid2 = align_to_peak(times2, centroid2, peak_idx2, pad_before, pad_after)
    aligned_energy1 = align_to_peak(times1, energy1, peak_idx1, pad_before, pad_after)
    aligned_energy2 = align_to_peak(times2, energy2, peak_idx2, pad_before, pad_after)
    aligned_noisiness1 = align_to_peak(times1, noisiness1, peak_idx1, pad_before, pad_after)
    aligned_noisiness2 = align_to_peak(times2, noisiness2, peak_idx2, pad_before, pad_after)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Centroid
    axes[0].plot(aligned_centroid1, color='red', label=os.path.basename(file1_path))
    axes[0].plot(aligned_centroid2, color='blue', label=os.path.basename(file2_path))
    axes[0].set_ylabel('Spectral Centroid (Hz)')
    axes[0].set_title('Aligned Spectral Centroid')
    axes[0].legend()
    axes[0].grid(True)
    
    # Energy
    axes[1].plot(aligned_energy1, color='red', label=os.path.basename(file1_path))
    axes[1].plot(aligned_energy2, color='blue', label=os.path.basename(file2_path))
    axes[1].set_ylabel('Temporal Energy')
    axes[1].set_title('Aligned Temporal Energy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Noisiness
    axes[2].plot(aligned_noisiness1, color='red', label=os.path.basename(file1_path))
    axes[2].plot(aligned_noisiness2, color='blue', label=os.path.basename(file2_path))
    axes[2].set_ylabel('Periodicity (Noisiness)')
    axes[2].set_title('Aligned Periodicity (Noisiness)')
    axes[2].set_xlabel('Frame Index (Aligned)')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved aligned comparison chart: {output_filename}")
    plt.close()

def plot_clustering_visualization(data_dict, file_names, true_labels, cluster_labels, cluster_to_sound, output_filename="clustering_visualization.png"):
    """Plot comprehensive clustering visualization with all dimensions"""
    # Extract summary features for each sample
    centroid_medians = []
    energy_medians = []
    noisiness_medians = []
    
    for file in file_names:
        data = data_dict[file]
        centroid_medians.append(np.median(data['centroid']))
        energy_medians.append(np.median(data['energy']))
        noisiness_medians.append(np.median(data['noisiness']))
    
    centroid_medians = np.array(centroid_medians)
    energy_medians = np.array(energy_medians)
    noisiness_medians = np.array(noisiness_medians)
    
    # Define colors for clusters
    cluster_colors = ['red', 'blue', 'green', 'orange']
    cluster_names = [f'Cluster {c} ({cluster_to_sound.get(c, "?")})' for c in range(4)]
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 2D Plots
    ax1 = fig.add_subplot(gs[0, 0])
    for cluster in range(4):
        mask = cluster_labels == cluster
        ax1.scatter(centroid_medians[mask], energy_medians[mask], 
                   c=cluster_colors[cluster], label=cluster_names[cluster], 
                   alpha=0.7, s=100, edgecolors='black', linewidth=1.5)
    ax1.set_xlabel('Spectral Centroid (Hz)', fontsize=12)
    ax1.set_ylabel('Temporal Energy', fontsize=12)
    ax1.set_title('Clustering: Spectral Centroid vs Temporal Energy', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 1])
    for cluster in range(4):
        mask = cluster_labels == cluster
        ax2.scatter(centroid_medians[mask], noisiness_medians[mask], 
                   c=cluster_colors[cluster], label=cluster_names[cluster], 
                   alpha=0.7, s=100, edgecolors='black', linewidth=1.5)
    ax2.set_xlabel('Spectral Centroid (Hz)', fontsize=12)
    ax2.set_ylabel('Periodicity (Noisiness)', fontsize=12)
    ax2.set_title('Clustering: Spectral Centroid vs Periodicity', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(gs[1, 0])
    for cluster in range(4):
        mask = cluster_labels == cluster
        ax3.scatter(energy_medians[mask], noisiness_medians[mask], 
                   c=cluster_colors[cluster], label=cluster_names[cluster], 
                   alpha=0.7, s=100, edgecolors='black', linewidth=1.5)
    ax3.set_xlabel('Temporal Energy', fontsize=12)
    ax3.set_ylabel('Periodicity (Noisiness)', fontsize=12)
    ax3.set_title('Clustering: Temporal Energy vs Periodicity', fontsize=14, fontweight='bold')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # 3D Plot
    ax4 = fig.add_subplot(gs[1, 1], projection='3d')
    for cluster in range(4):
        mask = cluster_labels == cluster
        ax4.scatter(centroid_medians[mask], energy_medians[mask], noisiness_medians[mask],
                   c=cluster_colors[cluster], label=cluster_names[cluster],
                   alpha=0.7, s=100, edgecolors='black', linewidth=1.5)
    ax4.set_xlabel('Spectral Centroid (Hz)', fontsize=10)
    ax4.set_ylabel('Temporal Energy', fontsize=10)
    ax4.set_zlabel('Periodicity (Noisiness)', fontsize=10)
    ax4.set_title('3D Clustering Visualization', fontsize=14, fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Feature distributions by cluster
    ax5 = fig.add_subplot(gs[2, 0])
    for cluster in range(4):
        mask = cluster_labels == cluster
        ax5.hist(centroid_medians[mask], alpha=0.5, label=cluster_names[cluster], bins=10)
    ax5.set_xlabel('Spectral Centroid (Hz)', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title('Spectral Centroid Distribution by Cluster', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    for cluster in range(4):
        mask = cluster_labels == cluster
        ax6.hist(noisiness_medians[mask], alpha=0.5, label=cluster_names[cluster], bins=10)
    ax6.set_xlabel('Periodicity (Noisiness)', fontsize=12)
    ax6.set_ylabel('Count', fontsize=12)
    ax6.set_title('Periodicity Distribution by Cluster', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('COMPREHENSIVE CLUSTERING VISUALIZATION', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(output_filename, bbox_inches='tight', dpi=150)
    print(f"Saved comprehensive clustering visualization: {output_filename}")
    plt.close()

def plot_wrong_classification_comparisons(data_dict, file_names, true_labels, cluster_labels, cluster_to_sound, 
                                         output_prefix="wrong_classification"):
    """Plot comparisons for wrongly classified samples"""
    # Find wrongly classified samples
    wrong_indices = []
    for i, (file_name, true_label, cluster) in enumerate(zip(file_names, true_labels, cluster_labels)):
        predicted_label = cluster_to_sound.get(cluster, "?")
        if true_label != predicted_label:
            wrong_indices.append((i, file_name, true_label, predicted_label))
    
    print(f"\nFound {len(wrong_indices)} wrongly classified samples")
    
    # Create sound to files mapping
    sound_to_files = {}
    for file_name, true_label in zip(file_names, true_labels):
        if true_label not in sound_to_files:
            sound_to_files[true_label] = []
        sound_to_files[true_label].append(file_name)
    
    # Plot each wrongly classified sample
    for idx, (sample_idx, wrong_file, true_sound, predicted_sound) in enumerate(wrong_indices[:5]):
        print(f"\nPlotting comparison for {wrong_file}...")
        
        # Get wrong sample data
        wrong_data = data_dict[wrong_file]
        
        # Get 2 samples from true category
        true_samples = [f for f in sound_to_files.get(true_sound, []) if f != wrong_file][:2]
        
        # Get 2 samples from predicted category
        predicted_samples = sound_to_files.get(predicted_sound, [])[:2]
        
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 1, hspace=0.3)
        
        colors = {
            'wrong': 'red',
            'true': 'green',
            'predicted': 'blue'
        }
        
        # Plot each metric
        metrics = [
            ('centroid', 'Spectral Centroid (Hz)', wrong_data['centroid']),
            ('energy', 'Temporal Energy', wrong_data['energy']),
            ('noisiness', 'Periodicity (Noisiness)', wrong_data['noisiness'])
        ]
        
        for metric_idx, (metric_name, metric_title, wrong_series) in enumerate(metrics):
            ax = fig.add_subplot(gs[metric_idx, 0])
            
            # Plot wrong sample
            ax.plot(wrong_series, color=colors['wrong'], linewidth=3, 
                   label=f'WRONG: {wrong_file} (True: {true_sound}, Pred: {predicted_sound})')
            
            # Plot true category samples
            for i, true_file in enumerate(true_samples):
                true_data = data_dict[true_file]
                ax.plot(true_data[metric_name], color=colors['true'], 
                       linewidth=1.5, alpha=0.7, linestyle='--',
                       label=f'True {true_sound}: {true_file}')
            
            # Plot predicted category samples
            for i, pred_file in enumerate(predicted_samples):
                pred_data = data_dict[pred_file]
                ax.plot(pred_data[metric_name], color=colors['predicted'], 
                       linewidth=1.5, alpha=0.7, linestyle=':',
                       label=f'Predicted {predicted_sound}: {pred_file}')
            
            ax.set_ylabel(metric_title, fontsize=12)
            ax.set_title(f'{metric_title} Comparison', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Frame Index (Aligned)', fontsize=11)
        
        plt.suptitle(f'WRONG CLASSIFICATION ANALYSIS: {wrong_file}\nTrue: {true_sound} → Predicted: {predicted_sound}', 
                    fontsize=16, fontweight='bold', y=0.995)
        output_filename = f"{output_prefix}_{idx+1}_{wrong_file.replace('.wav', '')}.png"
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        print(f"Saved: {output_filename}")
        plt.close()
    
    print(f"\nGenerated {min(5, len(wrong_indices))} wrong classification comparison plots")

def main():
    print("=== PEAK ALIGNMENT & OVERLAP PERCENTAGE CLUSTERING ===\n")
    
    audio_dir = "audio_data"
    valid_participants = [1, 2, 3, 4, 5, 7, 8, 10, 11]
    common_sounds = {'b', 'k', 'nu', 'psh'}
    
    # Extract and align all time-series data
    print("Extracting and aligning time-series data...")
    data_dict = {}
    true_labels = []
    file_names = []
    
    pad_before, pad_after = 50, 100
    
    for participant in valid_participants:
        phase1_path = os.path.join(audio_dir, str(participant), "Phase 1")
        if not os.path.exists(phase1_path):
            continue
        
        for file in os.listdir(phase1_path):
            if not file.endswith(".wav"):
                continue
            
            parts = file.split("-")
            if len(parts) < 2 or parts[1] not in common_sounds:
                continue
            
            file_path = os.path.join(phase1_path, file)
            data = extract_time_series(file_path)
            
            if data:
                times, centroid, energy, noisiness = data
                
                # Find peak in energy
                peak_idx = find_peak_index(energy)
                
                # Align all features to peak
                aligned_centroid = align_to_peak(times, centroid, peak_idx, pad_before, pad_after)
                aligned_energy = align_to_peak(times, energy, peak_idx, pad_before, pad_after)
                aligned_noisiness = align_to_peak(times, noisiness, peak_idx, pad_before, pad_after)
                
                data_dict[file] = {
                    'centroid': aligned_centroid,
                    'energy': aligned_energy,
                    'noisiness': aligned_noisiness,
                    'sound_type': parts[1],
                    'participant': participant
                }
                
                true_labels.append(parts[1])
                file_names.append(file)
    
    print(f"\nTotal samples: {len(data_dict)}")
    
    # Build overlap matrix
    print("\nBuilding overlap percentage matrix...")
    overlap_matrix, keys = build_overlap_matrix(data_dict)
    
    # Convert overlap to distance (100 - overlap)
    distance_matrix = 100 - overlap_matrix
    
    # Perform K-means clustering
    print("\nPerforming K-means clustering...")
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(distance_matrix)
    
    # Map clusters to sound types using majority vote
    from collections import Counter
    cluster_to_sound = {}
    for cluster in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        cluster_sounds = [true_labels[i] for i in cluster_indices]
        if cluster_sounds:
            most_common = Counter(cluster_sounds).most_common(1)[0][0]
            cluster_to_sound[cluster] = most_common
    
    # Print detailed results for each sample
    print("\n" + "="*80)
    print("DETAILED CLUSTERING RESULTS FOR EACH SAMPLE")
    print("="*80)
    print(f"{'File Name':<30} {'True Type':<12} {'Cluster':<8} {'Predicted Type':<15} {'Correct':<8}")
    print("-"*80)
    
    correct_count = 0
    for file_name, true_label, cluster in zip(file_names, true_labels, cluster_labels):
        predicted_label = cluster_to_sound.get(cluster, "?")
        is_correct = true_label == predicted_label
        if is_correct:
            correct_count += 1
        
        correct_str = "✓" if is_correct else "✗"
        print(f"{file_name:<30} {true_label:<12} {cluster:<8} {predicted_label:<15} {correct_str:<8}")
    
    # Print clustering results summary
    print("\n" + "="*80)
    print("CLUSTERING SUMMARY")
    print("="*80)
    print(f"Number of clusters: {n_clusters}")
    
    for cluster in range(n_clusters):
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster]
        cluster_sounds = [true_labels[i] for i in cluster_indices]
        sound_counts = Counter(cluster_sounds)
        
        print(f"\nCluster {cluster} (Mapped to: {cluster_to_sound.get(cluster, '?')}):")
        print(f"  Size: {len(cluster_indices)}")
        print(f"  Sounds: {sound_counts}")
    
    # Calculate and print accuracy
    accuracy = correct_count / len(file_names)
    print("\n" + "="*80)
    print("PERFORMANCE METRICS")
    print("="*80)
    print(f"Correct classifications: {correct_count}/{len(file_names)}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate Adjusted Rand Index
    if len(set(true_labels)) == n_clusters:
        ari = adjusted_rand_score(true_labels, cluster_labels)
        print(f"Adjusted Rand Index: {ari:.4f}")
    
    # Plot an example comparison
    print("\nGenerating example alignment plot...")
    example_file1 = os.path.join(audio_dir, "1", "Phase 1", "1-psh-3.wav")
    example_file2 = os.path.join(audio_dir, "7", "Phase 1", "7-psh-1.wav")
    if os.path.exists(example_file1) and os.path.exists(example_file2):
        plot_aligned_samples(example_file1, example_file2, "peak_aligned_comparison.png")
    
    # Generate comprehensive clustering visualization
    print("\nGenerating comprehensive clustering visualization...")
    plot_clustering_visualization(data_dict, file_names, true_labels, cluster_labels, 
                                   cluster_to_sound, "clustering_visualization.png")
    
    # Generate wrong classification comparisons
    print("\nGenerating wrong classification comparisons...")
    plot_wrong_classification_comparisons(data_dict, file_names, true_labels, cluster_labels, 
                                           cluster_to_sound, "wrong_classification")
    
    # Save detailed results to CSV
    print("\nSaving detailed results to CSV...")
    import csv
    with open("peak_alignment_clustering_detailed_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["File Name", "True Sound Type", "Cluster Assignment", "Predicted Sound Type", "Correct"])
        for file_name, true_label, cluster in zip(file_names, true_labels, cluster_labels):
            predicted_label = cluster_to_sound.get(cluster, "?")
            is_correct = true_label == predicted_label
            writer.writerow([file_name, true_label, cluster, predicted_label, "Yes" if is_correct else "No"])
    print("Saved: peak_alignment_clustering_detailed_results.csv")
    
    print("\n" + "="*80)
    print("=== COMPLETE ===")

if __name__ == "__main__":
    main()
