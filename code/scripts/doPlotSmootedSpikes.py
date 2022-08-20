
import sys
import argparse
import pickle
import numpy as np
import scipy.ndimage
import plotly.graph_objs as go
import gcnu_common.utils.neuralDataAnalysis

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--neuron_to_plot", help="neuron to plot", type=int,
                        default=23453)
    parser.add_argument("--trials_to_plot", help="trials to plot", type=str,
                        default="[0,1,2,3,4,5,6,7,8,9,"
                                 "10,11,12,13,14,15,16,17,18,19,"
                                 "20,21,22,23,24,25,26,27,28,29]")
    parser.add_argument("--gf_std_secs", help="gaussian filter std (sec)", type=float,
                        default=0.05)
    parser.add_argument("--bin_size_secs", help="bin size (secs)",
                        type=float, default=0.01)
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
                        default="../../figures/{:d}_smoothedSpikes_trials{:s}_binned{:.02f}sec_neuron{:02d}.{:s}")
    args = parser.parse_args()

    session_id = args.session_id
    neuron_to_plot = args.neuron_to_plot
    trials_to_plot = [int(str) for str in args.trials_to_plot[1:-1].split(",")]
    gf_std_secs = args.gf_std_secs
    bin_size_secs = args.bin_size_secs
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

    bins_edges = np.arange(epoch_start_offset, epoch_end_offset, bin_size_secs)
    bins_centers = (bins_edges[:-1] + bins_edges[1:])/2
    binned_spikes_times = \
            gcnu_common.utils.neuralDataAnalysis.binNeuronsAndTrialsSpikesTimes(
                spikes_times=spikes_times, bins_edges=bins_edges,
                time_unit="sec")

    gf_std_samples = gf_std_secs / bin_size_secs
    gf_binned_spikes_times = \
        [[scipy.ndimage.gaussian_filter1d(binned_spikes_times[r][n],
                                          gf_std_samples)
          for n in range(n_neurons)]
         for r in range(n_trials)]

    title = "Session {:d}, Neuron {:d}".format(session_id, neuron_to_plot)
    fig = go.Figure()
    for r in trials_to_plot:
        if choice_bino[r] == 0:
            trace_color = surebet_color
        elif choice_bino[r] == 1:
            trace_color = lottery_color
        else:
            raise ValueError("choice_bino item should be either 0 or 1. "
                             "Found choice={:d}".format(choice_bino[r]))
        trace_bar = go.Bar(x=bins_centers,
                           y=binned_spikes_times[r][neuron_to_plot_index],
                           marker_color=trace_color,
                           name="trial {:d}".format(r),
                           legendgroup="trial{:02d}".format(r),
                           showlegend=True)
        trace_line = go.Scatter(x=bins_centers,
                                y=gf_binned_spikes_times[r][neuron_to_plot_index],
                                line=dict(color=trace_color),
                                name="trial {:d}".format(r),
                                legendgroup="trial{:02d}".format(r),
                                showlegend=False)
        fig.add_trace(trace_bar)
        fig.add_trace(trace_line)
    fig.update_layout(title=title)
    png_fig_filename = fig_filename_pattern.format(session_id,
                                                   args.trials_to_plot,
                                                   bin_size_secs, neuron_to_plot,
                                                   "png")
    html_fig_filename = fig_filename_pattern.format(session_id,
                                                    args.trials_to_plot,
                                                    bin_size_secs, neuron_to_plot,
                                                    "html")
    fig.write_image(png_fig_filename)
    fig.write_html(html_fig_filename)
    fig.show()

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
