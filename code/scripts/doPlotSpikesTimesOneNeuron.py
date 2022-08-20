
import sys
import argparse
import os.path
import pickle
import numpy as np
import torch

import svGPFA.plot.plotUtilsPlotly

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--neuron_to_plot", help="neurontrial to plot", type=int,
                        default=23453)
    parser.add_argument("--xlim", help="x-axis plot limits", type=str,
                        default="(-1.0,3.0)")
    parser.add_argument("--epoched_spikes_filename_pattern",
                        help="filename containing the epoched spikes",
                        type=str,
                        default="../../results/risky_{:d}_epoched_spikes.pickle")
    parser.add_argument("--spikes_times_fig_filename_pattern",
                        help="spikes times figure filename pattern",
                        type=str,
                        default="../../figures/session{:d}_neuron{:02d}_spikes_times.{:s}")
    args = parser.parse_args()

    session_id = args.session_id
    neuron_to_plot = args.neuron_to_plot
    xlim = [float(str) for str in args.xlim[1:-1].split(",")]
    epoched_spikes_filename_pattern = args.epoched_spikes_filename_pattern
    spikes_times_fig_filename_pattern = args.spikes_times_fig_filename_pattern

    epoched_spikes_filename = epoched_spikes_filename_pattern.format(session_id)
    with open(epoched_spikes_filename, "rb") as f:
        loadRes = pickle.load(f)
    spikes_times = loadRes["spikes_times"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    spikes_times_torch = [[torch.from_numpy(spikes_times[r][n]) for n in range(n_neurons)] for r in range(n_trials)]
    trials_indices = torch.arange(n_trials)
    cell_ids = loadRes["cell_ids"]
    search_res = np.where(cell_ids == neuron_to_plot)[0]
    if len(search_res) > 0:
        neuron_to_plot_index = search_res[0]
    else:
        raise ValueError(f"neuron {neuron_to_plot} could not be found")

    title = "Session {:d}, Neuron {:d}".format(session_id, neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getSpikesTimesPlotOneNeuron(
        spikes_times=spikes_times_torch, neuron_index=neuron_to_plot_index,
        trials_indices=trials_indices, title=title)
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="black")
    fig.update_xaxes(range=xlim)
    spikes_times_png_filename = spikes_times_fig_filename_pattern.format(session_id, neuron_to_plot, "png")
    spikes_times_html_filename = spikes_times_fig_filename_pattern.format(session_id, neuron_to_plot, "html")
    fig.write_image(spikes_times_png_filename)
    fig.write_html(spikes_times_html_filename)
    fig.show()
    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
