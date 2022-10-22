
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
    parser.add_argument("--orthonormalized_latents_fig_filename_pattern",
                        help="figure filename for an orthonormalized latent",
                        type=str,
                        default="../../figures/{:08d}_orthonormalized_estimatedLatent_latent{:03d}.{:s}")
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
    orthonormalized_latents_fig_filename_pattern = args.orthonormalized_latents_fig_filename_pattern
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

    test_mu_k, _ = model.predictLatents(times=torch.from_numpy(trial_times))
    n_latents = test_mu_k[0].shape[1]
    test_mu_k_np = [test_mu_k[r].detach().numpy()
                    for r in range(len(test_mu_k))]
    estimated_C, estimated_d = model.getSVEmbeddingParams()
    estimatedC_np = estimated_C.detach().numpy()
    trials_colors = [color_trials_surebet \
                     if choice_bino[r]==0 else color_trials_lottery \
                     for r in range(n_trials)]
    trials_indices = np.arange(n_trials)
    trials_labels = np.array([str(i) for i in trials_indices])
    choice_annotations = ["surebet" if choice_bino[r]==0 else "lottery"
                          for r in range(n_trials)]
    trials_annotations = {"choice": choice_annotations}
    for latent_to_plot in range(n_latents):
        print(f"Processing latent {latent_to_plot}")

        fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
            times=trial_times, latentsMeans=test_mu_k_np,
            latentToPlot=latent_to_plot,
            C=estimatedC_np,
            # align_event=align_times, marked_events=marked_events,
            trials_labels=trials_labels,
            trials_annotations=trials_annotations,
            trials_colors=trials_colors,
            xlabel="Time (sec)")
        fig.write_image(orthonormalized_latents_fig_filename_pattern.format(
            est_res_number, latent_to_plot, "png"))
        fig.write_html(orthonormalized_latents_fig_filename_pattern.format(
            est_res_number, latent_to_plot, "html"))

if __name__ == "__main__":
    main(sys.argv)
