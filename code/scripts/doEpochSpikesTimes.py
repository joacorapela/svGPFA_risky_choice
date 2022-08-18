
import sys
import argparse
import configparser
import pickle
import numpy as np
# import pandas as pd
import scipy.io

def separateNeuronsSpikeTimesByTrials(neurons_spike_times, epochs_times, epochs_start_times, epochs_end_times):
    nTrials = len(epochs_times)
    nNeurons = len(neurons_spike_times)
    trials_spikes_times = []
    for r in range(nTrials):
        trial_spikes_times = []
        trial_epoch_time = epochs_times[r]
        trial_start_time = epochs_start_times[r]
        trial_end_time = epochs_end_times[r]
        for n in range(nNeurons):
            neuron_spikes_times = neurons_spike_times[n]
            trial_neuron_spikes_times = neuron_spikes_times[
                np.logical_and(trial_start_time<=neuron_spikes_times,
                               neuron_spikes_times<trial_end_time)]
            trial_spikes_times.append(trial_neuron_spikes_times-trial_epoch_time)
        trials_spikes_times.append(trial_spikes_times)
    return trials_spikes_times

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--clusters_types", help="comma-separated list of clusters types (e.g., single,multi,other)",
                        type=str, default="single")
    parser.add_argument("--spikes_times_var_name",
                        help="spikes times variable name", type=str,
                        default="neurons_ts")
    parser.add_argument("--epoch_var_name",
                        help="epoching variable namen",
                        type=str, default="event_cue")
    parser.add_argument("--clusters_labels_var_name",
                        help="clusters labels variable name",
                        type=str, default="clusters_labels")
    parser.add_argument("--lottery_magnitude_var_name",
                        help="lottery magnitude variable name",
                        type=str, default="lottery_magnitude")
    parser.add_argument("--surebet_magnitude_var_name",
                        help="surebet magnitude variable name",
                        type=str, default="surebet_magnitude")
    parser.add_argument("--choice_bino_var_name",
                        help="choice (binomial) variable name",
                        type=str, default="choice_bino")
    parser.add_argument("--current_reward_var_name",
                        help="current reweard variable name",
                        type=str, default="current_reward")
    parser.add_argument("--rt_choice_var_name",
                        help="reaction time choice variable name",
                        type=str, default="rt_choice")
    parser.add_argument("--rt_fixation_var_name",
                        help="reaction time fixation variable name",
                        type=str, default="rt_fixation")
    parser.add_argument("--epoch_start_offset",
                        help="epoch start offset (seconds)",
                        type=float, default=-1.0)
    parser.add_argument("--epoch_end_offset",
                        help="epoch end offset (seconds)",
                        type=float, default=3.0)
    parser.add_argument("--mat_data_filename_pattern",
                        help="mat data filename pattern",
                        type=str,
                        default="../../../data/risky_choice_data/risky_{:d}_unpacked_for_python.mat")
    parser.add_argument("--epoched_spikes_filename_pattern",
                        help="filename containing the epoched spikes",
                        type=str,
                        default="../../results/risky_{:d}_epoched_spikes.pickle")
    args = parser.parse_args()

    session_id = args.session_id
    clusters_types = args.clusters_types.split(",")
    spikes_times_var_name = args.spikes_times_var_name
    epoch_var_name = args.epoch_var_name
    clusters_labels_var_name = args.clusters_labels_var_name
    lottery_magnitude_var_name = args.lottery_magnitude_var_name
    surebet_magnitude_var_name = args.surebet_magnitude_var_name
    choice_bino_var_name = args.choice_bino_var_name
    current_reward_var_name = args.current_reward_var_name
    rt_choice_var_name = args.rt_choice_var_name
    rt_fixation_var_name = args.rt_fixation_var_name
    epoch_start_offset = args.epoch_start_offset
    epoch_end_offset = args.epoch_end_offset
    mat_data_filename_pattern = args.mat_data_filename_pattern
    epoched_spikes_filename_pattern = args.epoched_spikes_filename_pattern

    mat_data_filename = mat_data_filename_pattern.format(session_id)
    epoched_spikes_filename = epoched_spikes_filename_pattern.format(session_id)

    mat_data = scipy.io.loadmat(mat_data_filename)
    spikes_times = mat_data[spikes_times_var_name]
    epoch_times = mat_data[epoch_var_name]
    clusters_labels = [mat_data[clusters_labels_var_name][i,0][0] \
                       for i in range(mat_data[clusters_labels_var_name].shape[0])]
    valid_clusters_indices = [i for i in range(len(clusters_labels)) if clusters_labels[i] in clusters_types]
    valid_spikes_times = spikes_times[0, valid_clusters_indices]
    epoched_spikes_times = separateNeuronsSpikeTimesByTrials(
        neurons_spike_times=valid_spikes_times, epochs_times=epoch_times,
        epochs_start_times=epoch_times+epoch_start_offset,
        epochs_end_times=epoch_times+epoch_end_offset)
    lottery_magnitude = mat_data[lottery_magnitude_var_name]
    surebet_magnitude = mat_data[surebet_magnitude_var_name]
    choice_bino = mat_data[choice_bino_var_name]
    current_reward = mat_data[current_reward_var_name]
    rt_choice = mat_data[rt_choice_var_name]
    rt_fixation = mat_data[rt_fixation_var_name]
    results_to_save = {"spikes_times": epoched_spikes_times,
                       "clusters_indices": valid_clusters_indices,
                       "lottery_magnitude": lottery_magnitude,
                       "surebet_magnitude": surebet_magnitude,
                       "choice_bino": choice_bino,
                       "current_reward": current_reward,
                       "rt_choice": rt_choice,
                       "rt_fixation": rt_fixation,
                      }
    with open(epoched_spikes_filename, "wb") as f:
        pickle.dump(results_to_save, f)
    print("Saved results to {:s}".format(epoched_spikes_filename))
    import pdb; pdb.set_trace()


if __name__=="__main__":
    main(sys.argv)

