
import sys
import argparse
import pickle
import numpy as np
import torch
import plotly.graph_objs as go
import svGPFA.plot.plotUtilsPlotly

def approximateGaussianFilterSTDFromModelParams(model, neuron_to_plot_index):
    C, _ = model.getSVEmbeddingParams()
    kernels = model.getKernels()
    abs_lengthscales = []
    for kernel in kernels:
        assert(type(kernel).__name__ == "ExponentialQuadraticKernel")
        abs_lengthscales.append(abs(kernel.getParams()[0].item()))
    numerator = torch.abs(C[neuron_to_plot_index,:])
    denominator = numerator.sum()
    weights = numerator/denominator
    gf_std = torch.dot(weights, torch.tensor(abs_lengthscales, dtype=torch.double))
    return gf_std

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--neuron_to_plot", help="neuron to plot", type=int,
                        default=23493)
    parser.add_argument("--trials_to_plot", help="trials to plot", type=str,
                        default="[]")
                        # default="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]")
    parser.add_argument("--gf_std_secs", help="gaussian filter std (sec)",
                        type=float, default=-1.0)
    parser.add_argument("--est_res_number", help="estimation result number",
                        type=int, default=-1)
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern", type=str, 
                        default= "../../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--bin_size_secs", help="bin size (secs)",
                        type=float, default=0.01)
    parser.add_argument("--do_not_plot_spikes",
                        help="use this option to skip plotting spikes",
                        action="store_true")
    parser.add_argument("--surebet_color",
                        help="colors for surebet",
                        type=str, default="red")
    parser.add_argument("--lottery_color",
                        help="colors for lottery",
                        type=str, default="blue")
    parser.add_argument("--epoched_spikes_filename_pattern",
                        help="filename containing the epoched spikes",
                        type=str,
                        default="../../results/risky_{:d}_epoched_spikes.pickle")
    parser.add_argument("--fig_filename_pattern",
                        help="figure filename pattern",
                        type=str,
                        default="../../figures/{:d}_smoothedSpikes_binned{:.02f}sec_gfSTD{:f}_neuron{:02d}.{:s}")
    args = parser.parse_args()

    session_id = args.session_id
    neuron_to_plot = args.neuron_to_plot
    if len(args.trials_to_plot)>2:
        trials_to_plot = [int(str) for str in args.trials_to_plot[1:-1].split(",")]
    else:
        trials_to_plot = []
    gf_std_secs = args.gf_std_secs
    est_res_number = args.est_res_number
    model_save_filename_pattern = args.model_save_filename_pattern
    bin_size_secs = args.bin_size_secs
    do_not_plot_spikes = args.do_not_plot_spikes
    surebet_color = args.surebet_color
    lottery_color = args.lottery_color
    epoched_spikes_filename_pattern = \
        args.epoched_spikes_filename_pattern
    fig_filename_pattern = args.fig_filename_pattern

    epoched_spikes_filename = epoched_spikes_filename_pattern.format(session_id)

    with open(epoched_spikes_filename, "rb") as f:
        loadRes = pickle.load(f)
    spikes_times = loadRes["spikes_times"]
    epoch_start_offset = loadRes["epoch_start_offset"]
    epoch_end_offset = loadRes["epoch_end_offset"]
    choice_bino = loadRes["choice_bino"]
    cell_ids = loadRes["cell_ids"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    search_res = np.where(cell_ids == neuron_to_plot)[0]
    if len(search_res) > 0:
        neuron_to_plot_index = search_res[0]
    else:
        raise ValueError(f"neuron {neuron_to_plot} could not be found")

    if gf_std_secs < 0:
        if est_res_number < 0:
            raise ValueError("Either gf_std_secs or est_res_number must be "
                             "given")
        else:
            model_save_filename = model_save_filename_pattern.format(
                est_res_number)
            with open(model_save_filename, "rb") as f:
                loadRes = pickle.load(f)
            model = loadRes["model"]
            gf_std_secs = approximateGaussianFilterSTDFromModelParams(
                model=model, neuron_to_plot_index=neuron_to_plot_index)

    if len(trials_to_plot) == 0:
        trials_to_plot = np.arange(n_trials)
    trials_colors = [surebet_color \
                     if choice_bino[r]==0 else lottery_color \
                     for r in range(n_trials)]
    title = "Session {:d}, Neuron {:d}".format(session_id, neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotSmoothedSpikes(
        spikes_times=spikes_times,
        gf_std_secs=gf_std_secs,
        epoch_start_offset=epoch_start_offset,
        epoch_end_offset=epoch_end_offset,
        bin_size_secs=bin_size_secs,
        neuron_to_plot_index=neuron_to_plot_index,
        trials_to_plot=trials_to_plot,
        trials_colors=trials_colors,
        title=title
    )
    png_fig_filename = fig_filename_pattern.format(session_id,
                                                   bin_size_secs,
                                                   gf_std_secs,
                                                   neuron_to_plot,
                                                   "png")
    html_fig_filename = fig_filename_pattern.format(session_id,
                                                    bin_size_secs,
                                                    gf_std_secs,
                                                    neuron_to_plot,
                                                    "html")
    fig.write_image(png_fig_filename)
    fig.write_html(html_fig_filename)
    fig.show()

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
