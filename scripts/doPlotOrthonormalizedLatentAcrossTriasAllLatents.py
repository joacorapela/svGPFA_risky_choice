
import sys
import numpy as np
import torch
import pickle
import argparse
import plotly.express as px

import svGPFA.plot.plotUtilsPlotly

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("est_res_number", help="estimation result number", type=int)
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
    parser.add_argument("--orthonormalized_latents_fig_filename_pattern",
                        help="figure filename for an orthonormalized latent",
                        type=str,
                        default="../figures/{:08d}_orthonormalized_estimatedLatent_{:s}_latent{:03d}.{:s}")
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
    orthonormalized_latents_fig_filename_pattern = \
        args.orthonormalized_latents_fig_filename_pattern
    model_save_filename_pattern = args.model_save_filename_pattern

    with open(epoched_spikes_filename, "rb") as f:
        loadRes = pickle.load(f)
    trials_start_times = loadRes["trials_start_times"]
    trials_end_times = loadRes["trials_end_times"]
    subject_choices = loadRes["subject_choices"]
    center_on_times = loadRes["center_on_times"]
    choice_on_times = loadRes["choice_on_times"]
    choice_poke_times = loadRes["choice_poke_times"]
    lottery_magnitude = loadRes["lottery_magnitude"].squeeze()

    model_save_filename = model_save_filename_pattern.format(est_res_number)
    with open(model_save_filename, "rb") as f:
        estResults = pickle.load(f)
    model = estResults["model"]
    valid_trials_indices = estResults["valid_trials_indices"]
    n_trials = len(valid_trials_indices)
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

    test_mu_k, _ = model.predictLatents(times=trials_times)
    n_latents = test_mu_k[0].shape[1]
    test_mu_k_np = [test_mu_k[r].detach().numpy()
                    for r in range(len(test_mu_k))]
    estimated_C, estimated_d = model.getSVEmbeddingParams()
    estimatedC_np = estimated_C.detach().numpy()
    trials_indices = np.arange(n_trials)
    trials_labels = np.array([str(i) for i in trials_indices])
    trials_annotations = {"choice": subject_choices, 
                          "lottery mag": lottery_magnitude}
    marked_events = np.column_stack((center_on_times, center_poke_times,
                                     choice_on_times, choice_poke_times))
    marked_events_colors = [color_center_on, color_center_poke,
                            color_choice_on, color_choice_poke]
    align_times = np.zeros(n_trials)
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

    for latent_to_plot in range(n_latents):
        print(f"Processing latent {latent_to_plot}")
        fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
            trials_times=trials_times.numpy(), latentsMeans=test_mu_k_np,
            latentToPlot=latent_to_plot, C=estimatedC_np,
            align_event=align_times, marked_events=marked_events,
            marked_events_colors=marked_events_colors,
            trials_labels=trials_labels, trials_annotations=trials_annotations,
            trials_colors=trials_colors_c, xlabel="Time (msec)")
        fig.write_image(orthonormalized_latents_fig_filename_pattern.format(
            est_res_number, "choice", latent_to_plot, "png"))
        fig.write_html(orthonormalized_latents_fig_filename_pattern.format(
            est_res_number, "choice", latent_to_plot, "html"))

        fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
            trials_times=trials_times.numpy(), latentsMeans=test_mu_k_np,
            latentToPlot=latent_to_plot, C=estimatedC_np,
            align_event=align_times, marked_events=marked_events,
            marked_events_colors=marked_events_colors,
            trials_labels=trials_labels, trials_annotations=trials_annotations,
            trials_colors=trials_colors_lm, xlabel="Time (msec)")
        fig.write_image(orthonormalized_latents_fig_filename_pattern.format(
            est_res_number, "lm", latent_to_plot, "png"))
        fig.write_html(orthonormalized_latents_fig_filename_pattern.format(
            est_res_number, "lm", latent_to_plot, "html"))


if __name__ == "__main__":
    main(sys.argv)
