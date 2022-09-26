
import sys
import argparse
import pickle
import numpy as np
import scipy.io


def separateNeuronsSpikeTimesByTrials(neurons_spike_times, epochs_times,
                                      epochs_start_times, epochs_end_times):
    n_trials = len(epochs_times)
    n_neurons = len(neurons_spike_times)
    trials_spikes_times = [[] for r in range(n_trials)]
    for r in range(n_trials):
        trial_epoch_time = epochs_times[r]
        trial_start_time = epochs_start_times[r]
        trial_end_time = epochs_end_times[r]
        trial_spikes_times = [[] for n in range(n_neurons)]
        for n in range(n_neurons):
            neuron_spikes_times = neurons_spike_times[n]
            trial_neuron_spikes_times = neuron_spikes_times[
                np.logical_and(trial_start_time <= neuron_spikes_times,
                               neuron_spikes_times < trial_end_time)]
            trial_spikes_times[n] = trial_neuron_spikes_times-trial_epoch_time
        trials_spikes_times[r] = trial_spikes_times
    return trials_spikes_times


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--clusters_types",
                        help="comma-separated list of clusters types "
                             "(e.g., single,multi,other)",
                        type=str, default="single")
    parser.add_argument("--spikes_times_var_name",
                        help="spikes times variable name", type=str,
                        default="neurons_ts")
    parser.add_argument("--clusters_labels_var_name",
                        help="clusters labels variable name",
                        type=str, default="clusters_labels")
    parser.add_argument("--center_choice_var_name",
                        help="center choice variable namen",
                        type=str, default="cue_t")
    parser.add_argument("--side_choice_var_name",
                        help="side choice variable namen",
                        type=str, default="target_t")
    parser.add_argument("--rt_choice_var_name",
                        help="reaction time choice variable name",
                        type=str, default="rt_choice")
    parser.add_argument("--rt_fixation_var_name",
                        help="reaction time fixation variable name",
                        type=str, default="rt_fixation")
    parser.add_argument("--cell_id_var_name",
                        help="cell id variable name",
                        type=str, default="cell_id")
    parser.add_argument("--lottery_magnitude_var_name",
                        help="lottery magnitude variable name",
                        type=str, default="lottery_mag")
    parser.add_argument("--surebet_magnitude_var_name",
                        help="surebet magnitude variable name",
                        type=str, default="surebet_mag")
    parser.add_argument("--subject_choice_var_name",
                        help="subject choice variable name",
                        type=str, default="subject_choice")
    parser.add_argument("--current_reward_var_name",
                        help="current reweard variable name",
                        type=str, default="current_reward")
    parser.add_argument("--epoch_start_offset",
                        help="epoch start offset (seconds)",
                        type=float, default=0.1)
    parser.add_argument("--epoch_end_offset",
                        help="epoch end offset (seconds)",
                        type=float, default=0.5)
    parser.add_argument("--spike_data_filename_pattern",
                        help="spike data filename pattern",
                        type=str,
                        default="../../data/risky_choice_data/risky_{:d}_unpacked_for_python.mat")
    parser.add_argument("--exp_data_filename_pattern",
                        help="experiment data filename pattern",
                        type=str,
                        default="../../data/risky_{:d}_add_unpacked_for_python.mat")
    parser.add_argument("--epoched_spikes_filename_pattern",
                        help="filename containing the epoched spikes",
                        type=str,
                        default="../../results/risky_{:d}_epoched_spikes.pickle")
    args = parser.parse_args()

    session_id = args.session_id
    clusters_types = args.clusters_types.split(",")
    cell_id_var_name = args.cell_id_var_name
    spikes_times_var_name = args.spikes_times_var_name
    center_choice_var_name = args.center_choice_var_name
    side_choice_var_name = args.side_choice_var_name
    rt_choice_var_name = args.rt_choice_var_name
    rt_fixation_var_name = args.rt_fixation_var_name
    clusters_labels_var_name = args.clusters_labels_var_name
    lottery_magnitude_var_name = args.lottery_magnitude_var_name
    surebet_magnitude_var_name = args.surebet_magnitude_var_name
    subject_choice_var_name = args.subject_choice_var_name
    current_reward_var_name = args.current_reward_var_name
    epoch_start_offset = args.epoch_start_offset
    epoch_end_offset = args.epoch_end_offset
    spike_data_filename_pattern = args.spike_data_filename_pattern
    exp_data_filename_pattern = args.exp_data_filename_pattern
    epoched_spikes_filename_pattern = args.epoched_spikes_filename_pattern

    spike_data_filename = spike_data_filename_pattern.format(session_id)
    exp_data_filename = exp_data_filename_pattern.format(session_id)
    epoched_spikes_filename = epoched_spikes_filename_pattern.format(
        session_id)

    spike_data = scipy.io.loadmat(spike_data_filename)
    spikes_times = spike_data[spikes_times_var_name]
    cell_ids = spike_data[cell_id_var_name].squeeze()
    clusters_labels = [spike_data[clusters_labels_var_name][i, 0][0]
                       for i in range(spike_data[clusters_labels_var_name].
                                      shape[0])]

    exp_data = scipy.io.loadmat(exp_data_filename)
    center_choice_times = exp_data[center_choice_var_name]
    side_choice_times = exp_data[side_choice_var_name]
    rt_fixation = exp_data[rt_fixation_var_name]
    rt_choice = exp_data[rt_choice_var_name]
    lottery_magnitude = exp_data[lottery_magnitude_var_name]
    surebet_magnitude = exp_data[surebet_magnitude_var_name]
    subject_choice = exp_data[subject_choice_var_name]
    current_reward = exp_data[current_reward_var_name]
    side_on_times = side_choice_times - rt_choice

    valid_clusters_indices = [i for i in range(len(clusters_labels))
                              if clusters_labels[i] in clusters_types]
    valid_spikes_times = spikes_times[0, valid_clusters_indices]
    valid_cell_ids = cell_ids[valid_clusters_indices]

    epochs_times = center_choice_times
    epochs_start_times_abs = center_choice_times - rt_fixation - \
        epoch_start_offset
    epochs_end_times_abs = side_choice_times + rt_choice + epoch_end_offset

    epoched_spikes_times = separateNeuronsSpikeTimesByTrials(
        neurons_spike_times=valid_spikes_times,
        epochs_times=epochs_times,
        epochs_start_times=epochs_start_times_abs,
        epochs_end_times=epochs_end_times_abs)
    epochs_start_times_rel = epochs_start_times_abs - epochs_times
    epochs_end_times_rel = epochs_end_times_abs - epochs_times
    center_on_times_abs = center_choice_times - rt_fixation
    center_on_times_rel = center_on_times_abs - epochs_times
    side_on_times_rel = side_on_times - epochs_times
    side_choice_times_rel = side_choice_times - epochs_times
    results_to_save = {"spikes_times": epoched_spikes_times,
                       "epochs_start_times": epochs_start_times_rel,
                       "epochs_end_times": epochs_end_times_rel,
                       "center_on_times": center_on_times_rel,
                       "side_on_times": side_on_times_rel,
                       "side_choice_times": side_choice_times_rel,
                       "rt_choice": rt_choice,
                       "rt_fixation": rt_fixation,
                       "cell_ids": valid_cell_ids,
                       "lottery_magnitude": lottery_magnitude,
                       "surebet_magnitude": surebet_magnitude,
                       "subject_choice": subject_choice,
                       "current_reward": current_reward}
    with open(epoched_spikes_filename, "wb") as f:
        pickle.dump(results_to_save, f)
    print("Saved results to {:s}".format(epoched_spikes_filename))
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    main(sys.argv)
