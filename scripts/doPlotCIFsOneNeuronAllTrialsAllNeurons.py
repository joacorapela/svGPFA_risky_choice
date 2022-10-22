
import sys
import numpy as np
import torch
import pickle
import argparse
import plotly.express as px

import svGPFA.plot.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", help="estimation result number",
                        type=int)
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--n_time_steps_CIF", help="number of stime steps in "
                        "CIF plots", type=int, default=100)
    parser.add_argument("--color_trials_surebet",
                        help="color for surebet trials",
                        default="rgba(255,0,0,0.3)")
    parser.add_argument("--color_trials_lottery",
                        help="color for lotterry trials",
                        default="rgba(0,0,255,0.3)")
    parser.add_argument("--color_center_on",
                        help="color for center_on event marker",
                        default="rgba(255,0,255,0.7)")
    parser.add_argument("--color_center_poke",
                        help="color for center_poke event marker",
                        default="rgba(0,0,0,0.7)")
    parser.add_argument("--color_choice_on",
                        help="color for choice_on event marker",
                        default="rgba(0,255,0,0.7)")
    parser.add_argument("--color_choice_poke",
                        help="color for choice_poke event marker",
                        default="rgba(0,255,255,0.7)")
    parser.add_argument("--lm_colorscale",
                        help="color scale for lottery magnitude",
                        default="hot")
    parser.add_argument("--epoched_spikes_filename_pattern",
                        help="filename containing the epoched spikes",
                        type=str,
                        default="../results/risky_{:d}_epoched_spikes.pickle")
    parser.add_argument("--CIFs_one_neuron_all_trials_fig_filename_pattern",
                        help="figure filename for CIFs one neuron all trials plot",
                        type=str,
                        default="../figures/{:08d}_CIFsAllTrials_{:s}_neuron{:d}.{:s}")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../results/{:08d}_estimatedModel.pickle")

    args = parser.parse_args()
    est_res_number = args.est_res_number
    session_id = args.session_id
    n_time_steps_CIF = args.n_time_steps_CIF
    color_trials_surebet = args.color_trials_surebet
    color_trials_lottery = args.color_trials_lottery
    color_center_on = args.color_center_on
    color_center_poke = args.color_center_poke
    color_choice_on = args.color_choice_on
    color_choice_poke = args.color_choice_poke
    lm_colorscale = args.lm_colorscale
    epoched_spikes_filename = args.epoched_spikes_filename_pattern.format(
        session_id)
    CIFs_one_neuron_all_trials_fig_filename_pattern = \
        args.CIFs_one_neuron_all_trials_fig_filename_pattern
    model_save_filename_pattern = args.model_save_filename_pattern

    model_save_filename = model_save_filename_pattern.format(est_res_number)
    with open(model_save_filename, "rb") as f:
        estResults = pickle.load(f)
    model = estResults["model"]
    valid_trials_indices = estResults["valid_trials_indices"]
    n_trials = len(valid_trials_indices)
    with open(epoched_spikes_filename, "rb") as f:
        loadRes = pickle.load(f)

    trials_start_times = loadRes["trials_start_times"]
    trials_end_times = loadRes["trials_end_times"]
    subject_choices = loadRes["subject_choices"]
    center_on_times = loadRes["center_on_times"]
    center_poke_times = [0.0 for i in valid_trials_indices]
    choice_on_times = loadRes["choice_on_times"]
    choice_poke_times = loadRes["choice_poke_times"]
    spikes_times = loadRes["spikes_times"]
    cell_ids = loadRes["cell_ids"]
    lottery_magnitude = loadRes["lottery_magnitude"].squeeze()
    n_neurons = len(spikes_times[0])

    trials_start_times = trials_start_times[valid_trials_indices]
    trials_end_times = trials_end_times[valid_trials_indices]
    subject_choices = [subject_choices[i] for i in valid_trials_indices]
    center_on_times = [center_on_times[i] for i in valid_trials_indices]
    center_poke_times = [0.0 for i in valid_trials_indices]
    choice_on_times = [choice_on_times[i] for i in valid_trials_indices]
    choice_poke_times = [choice_poke_times[i] for i in valid_trials_indices]
    lottery_magnitude = [lottery_magnitude[i] for i in valid_trials_indices]
    unique_lottery_magnitudes = np.unique(lottery_magnitude).tolist()

    trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
        start_times=np.squeeze(trials_start_times),
        end_times=np.squeeze(trials_end_times),
        n_steps=n_time_steps_CIF)

    with torch.no_grad():
        e_pos_CIF_values = model.computeExpectedPosteriorCIFs(
            times=trials_times)

    trials_annotations = {"choice": subject_choices, 
                          "lottery mag": lottery_magnitude}
    marked_events = np.column_stack((center_on_times, center_poke_times,
                                     choice_on_times, choice_poke_times))
    marked_events_colors = [color_center_on, color_center_poke,
                            color_choice_on, color_choice_poke]
    trials_colors_c = [color_trials_surebet
                       if subject_choices[r] == "surebet"
                       else color_trials_lottery
                       for r in range(n_trials)]

    colors_lm = px.colors.sample_colorscale(
        colorscale=lm_colorscale, samplepoints=len(unique_lottery_magnitudes),
        colortype="rgb")
    trials_colors_lm = [None] * n_trials
    for r in range(n_trials):
        index = np.where(lottery_magnitude[r] ==
                         unique_lottery_magnitudes)[0][0]
        trials_colors_lm[r] = colors_lm[index]


    for neuron_to_plot_index in range(n_neurons):
        neuron_to_plot = cell_ids[neuron_to_plot_index].item()
        print(f"Processing neuron {neuron_to_plot}")
        title = "Neuron {:d}".format(neuron_to_plot)
        fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
            trials_times=trials_times.numpy(), cif_values=e_pos_CIF_values,
            neuron_index=neuron_to_plot_index,
            align_event=center_poke_times,
            marked_events=marked_events,
            marked_events_colors=marked_events_colors,
            trials_annotations=trials_annotations,
            trials_colors=trials_colors_c,
            title=title)
        fig.write_image(CIFs_one_neuron_all_trials_fig_filename_pattern.format(
            est_res_number, "choice", neuron_to_plot, "png"))
        fig.write_html(CIFs_one_neuron_all_trials_fig_filename_pattern.format(
            est_res_number, "choice", neuron_to_plot, "html"))

        fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
            trials_times=trials_times.numpy(), cif_values=e_pos_CIF_values,
            neuron_index=neuron_to_plot_index,
            align_event=center_poke_times,
            marked_events=marked_events,
            marked_events_colors=marked_events_colors,
            trials_annotations=trials_annotations,
            trials_colors=trials_colors_lm,
            title=title)
        fig.write_image(CIFs_one_neuron_all_trials_fig_filename_pattern.format(
            est_res_number, "lm", neuron_to_plot, "png"))
        fig.write_html(CIFs_one_neuron_all_trials_fig_filename_pattern.format(
            est_res_number, "lm", neuron_to_plot, "html"))


if __name__ == "__main__":
    main(sys.argv)
