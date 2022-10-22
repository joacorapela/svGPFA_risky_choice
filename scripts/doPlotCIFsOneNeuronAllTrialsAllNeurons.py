
import sys
import numpy as np
import torch
import pickle
import argparse

import svGPFA.plot.plotUtilsPlotly

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", help="estimation result number", type=int)
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--dt_CIF", help="neuron to plot", type=float,
                        default=0.01)
    parser.add_argument("--color_trials_surebet",
                        help="color for surebet trials",
                        default="rgba(255,0,0,0.3)")
    parser.add_argument("--color_trials_lottery",
                        help="color for lotterry trials",
                        default="rgba(0,0,255,0.3)")
    parser.add_argument("--epoched_spikes_filename_pattern",
                        help="filename containing the epoched spikes",
                        type=str,
                        default="../../results/risky_{:d}_epoched_spikes.pickle")
    parser.add_argument("--CIFs_one_neuron_all_trials_fig_filename_pattern",
                        help="figure filename for CIFs one neuron all trials plot",
                        type=str,
                        default="../../figures/{:08d}_CIFsAllTrials_neuron{:d}.{:s}")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default= "../../results/{:08d}_estimatedModel.pickle")

    args = parser.parse_args()
    est_res_number = args.est_res_number
    session_id = args.session_id
    dt_CIF = args.dt_CIF
    color_trials_surebet = args.color_trials_surebet
    color_trials_lottery = args.color_trials_lottery
    epoched_spikes_filename = args.epoched_spikes_filename_pattern.format(session_id)
    CIFs_one_neuron_all_trials_fig_filename_pattern = args.CIFs_one_neuron_all_trials_fig_filename_pattern 
    model_save_filename_pattern = args.model_save_filename_pattern 

    with open(epoched_spikes_filename, "rb") as f:
        loadRes = pickle.load(f)
    from_time = loadRes["epoch_start_offset"]
    to_time = loadRes["epoch_end_offset"]
    spikes_times = loadRes["spikes_times"]
    cell_ids = loadRes["cell_ids"]
    choice_bino = loadRes["choice_bino"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    trial_times = np.arange(from_time, to_time, dt_CIF)

    model_save_filename = model_save_filename_pattern.format(est_res_number)
    with open(model_save_filename, "rb") as f:
        estResults = pickle.load(f)
    model = estResults["model"]
    with torch.no_grad():
        e_pos_CIF_values = model.computeExpectedPosteriorCIFs(
            times=torch.from_numpy(trial_times))

    trials_colors = [color_trials_surebet \
                     if choice_bino[r]==0 else color_trials_lottery \
                     for r in range(n_trials)]
    for neuron_to_plot_index in range(n_neurons):
        neuron_to_plot = cell_ids[neuron_to_plot_index].item()
        print(f"Processing neuron {neuron_to_plot}")
        title = "Neuron {:d}".format(neuron_to_plot)
        fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
            times=trial_times, cif_values=e_pos_CIF_values,
            neuron_index=neuron_to_plot_index,
            # sort_event=sideIn_times, align_event=centerOut_times, marked_events=marked_events,
            trials_colors=trials_colors,
            title=title)
        fig.write_image(CIFs_one_neuron_all_trials_fig_filename_pattern.format(est_res_number, neuron_to_plot, "png"))
        fig.write_html(CIFs_one_neuron_all_trials_fig_filename_pattern.format(est_res_number, neuron_to_plot, "html"))

if __name__ == "__main__":
    main(sys.argv)
