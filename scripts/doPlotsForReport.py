
import sys
import os.path
import numpy as np
import torch
import pickle
import argparse
import configparser
import pandas as pd
import sklearn.metrics

import gcnu_common.utils.neuralDataAnalysis
import gcnu_common.stats.pointProcesses.tests
import svGPFA.utils.configUtils
import svGPFA.utils.initUtils
import svGPFA.utils.miscUtils
import svGPFA.plot.plotUtilsPlotly

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number", type=int)
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--latent_to_plot", help="trial to plot", type=int, default=0)
    parser.add_argument("--neuron_to_plot", help="neuron to plot", type=int,
                        default=23453)
    parser.add_argument("--trial_to_plot", help="trial to plot", type=int, default=0)
    parser.add_argument("--dt_CIF", help="neuron to plot", type=float,
                        default=0.01)
    parser.add_argument("--ksTestGamma", help="gamma value for KS test", type=int, default=10)
    parser.add_argument("--nTestPoints", help="number of test points where to plot latents", type=int, default=2000)
    parser.add_argument("--trials_indices", help="trials indices to plot",
                        default="[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]")
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

#     parser.add_argument("--trial_choice_column_name", help="trial choice column name",
#                         type=str,
#                         default="choice")
#     parser.add_argument("--trial_rewarded_column_name", help="trial rewarded column name",
#                         type=str,
#                         default="rewarded")
#     parser.add_argument("--location", help="location to analyze", type=int, default=0)
#     parser.add_argument("--min_nSpikes_perNeuron_perTrial",
#                         help="min number of spikes per neuron per trial",
#                         type=int, default=1)
#     parser.add_argument("--events_times_filename",
#                         help="events times filename",
#                         type=str,
#                         default="../../data/022822/s008_tab_m1113182_LR__20210516_173815__probabilistic_switching.df.csv")
#     parser.add_argument("--region_spikes_times_filename_pattern",
#                         help="region spikes times filename pattern",
#                         type=str,
#                         default="../../results/00000000_region{:s}_spikes_times_epochedaligned__last_center_out.{:s}")

    args = parser.parse_args()

    estResNumber = args.estResNumber
    session_id = args.session_id
    latent_to_plot = args.latent_to_plot
    neuron_to_plot = args.neuron_to_plot
    trial_to_plot = args.trial_to_plot
    dt_CIF = args.dt_CIF
    ksTestGamma = args.ksTestGamma
    trials_indices = [int(str) for str in args.trials_indices[1:-1].split(",")]
    color_trials_surebet = args.color_trials_surebet
    color_trials_lottery = args.color_trials_lottery
    epoched_spikes_filename_pattern = args.epoched_spikes_filename_pattern 

#     region = args.region
#     block_types_indices = [int(str) for str in args.block_types_indices[1:-1].split(",")]
#     align_times_column_name = args.align_times_column_name
#     if len(args.sort_times_column_name)>0:
#         sort_times_column_name = args.sort_times_column_name
#     else:
#         sort_times_column_name = ""
#     centerIn_times_column_name = args.centerIn_times_column_name
#     centerOut_times_column_name = args.centerOut_times_column_name
#     sideIn_times_column_name = args.sideIn_times_column_name
#     trial_choice_column_name = args.trial_choice_column_name
#     trial_rewarded_column_name = args.trial_rewarded_column_name
#     min_nSpikes_perNeuron_perTrial = args.min_nSpikes_perNeuron_perTrial
#     events_times_filename = args.events_times_filename
#     region_spikes_times_filename_pattern = args.region_spikes_times_filename_pattern

    epoched_spikes_filename = args.epoched_spikes_filename_pattern.format(session_id)
    estimResMetaDataFilename = "../../results/{:08d}_estimation_metaData.ini".format(estResNumber)
    modelSaveFilename = "../../results/{:08d}_estimatedModel.pickle".format(estResNumber)
    lowerBoundHistVsIterNoFigFilenamePattern = "../../figures/{:08d}_lowerBoundHistVsIterNo.{{:s}}".format(estResNumber)
    lowerBoundHistVsElapsedTimeFigFilenamePattern = "../../figures/{:08d}_lowerBoundHistVsElapsedTime.{{:s}}".format(estResNumber)
    latentsFigFilenamePattern = "../../figures/{:08d}_estimatedLatent_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    orthonormalizedLatentsFigFilenamePattern = "../../figures/{:08d}_orthonormalized_estimatedLatent_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    orthonormalizedLatents3DFigFilenamePattern = "../../figures/{:08d}_3Dorthonormalized_estimatedLatent.{{:s}}".format(estResNumber)
    orthonormalizedLatentsImageFigFilenamePattern = "../../figures/{:08d}_orthonormalized_latent{:03d}_image.{{:s}}".format(estResNumber, latent_to_plot)
    embeddingsFigFilenamePattern = "../../figures/{:08d}_estimatedEmbedding_neuron{:d}.{{:s}}".format(estResNumber, neuron_to_plot)
    embeddingParamsFigFilenamePattern = "../../figures/{:08d}_estimatedEmbeddingParams.{{:s}}".format(estResNumber)
    CIFFigFilenamePattern = "../../figures/{:08d}_CIF_trial{:03d}_neuron{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
    CIFsOneNeuronAllTrialsFigFilenamePattern = "../../figures/{:08d}_CIFsAllTrials_neuron{:d}.{{:s}}".format(estResNumber, neuron_to_plot)
    ksTestTimeRescalingNumericalCorrectionFigFilenamePattern = "../../figures/{:08d}_ksTestTimeRescaling_numericalCorrection_trial{:03d}_neuron{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
    rocFigFilenamePattern = "../../figures/{:08d}_rocAnalysis_trial{:03d}_neuron{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
    kernelsParamsFigFilenamePattern = "../../figures/{:08d}_estimatedKernelsParams.{{:s}}".format(estResNumber)

    with open(epoched_spikes_filename, "rb") as f:
        loadRes = pickle.load(f)
    from_time = loadRes["epoch_start_offset"]
    to_time = loadRes["epoch_end_offset"]
    spikes_times = loadRes["spikes_times"]
    spikes_times = [[torch.tensor(spikes_times[r][n])
                     for n in range(len(spikes_times[r]))]
                    for r in range(len(spikes_times))]
    cell_ids = loadRes["cell_ids"]
    choice_bino = loadRes["choice_bino"]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    trials_indices = np.arange(n_trials)

#     region_spikes_times_filename = region_spikes_times_filename_pattern.format(region, "pickle")
#     with open(region_spikes_times_filename, "rb") as f:
#         loadRes = pickle.load(f)
#     spikes_times = loadRes["spikes_times"]
#     events_times = pd.read_csv(events_times_filename)
#     trials_indices = [r for r in range(len(events_times)) \
#                       if events_times.iloc[r]["block_type_index"] in \
#                       block_types_indices]
#     if len(sort_times_column_name)>0:
#         sort_times = events_times[sort_times_column_name]
#     else:
#         sort_times = None
#     spikes_times = [spikes_times[r] for r in trials_indices]
#     spikes_times, neurons_indices = utils.neuralDataAnalysis.removeUnitsWithLessSpikesThanThrInAnyTrials(
#         spikes_times=spikes_times,
#         min_nSpikes_perNeuron_perTrial=min_nSpikes_perNeuron_perTrial)
#     spikes_times = [[torch.tensor(spikes_times[r][n])
#                      for n in range(len(spikes_times[r]))]
#                     for r in range(len(spikes_times))]

    trial_times = np.arange(from_time, to_time, dt_CIF)

    with open(modelSaveFilename, "rb") as f: estResults = pickle.load(f)
    lower_bound_hist = estResults["lowerBoundHist"]
    elapsed_time_hist = estResults["elapsedTimeHist"]
    model = estResults["model"]
    search_res = np.nonzero(cell_ids == neuron_to_plot)[0]
    cell_ids_str = "".join(str(i)+" " for i in cell_ids)
    if len(search_res)==0:
        raise ValueError("Neuron {:d} is not valid. Valid neurons are ".format(neuron_to_plot) + cell_ids_str)
    else:
        neuron_to_plot_index = search_res[0].item()
#     align_times = events_times.iloc[trials_indices][align_times_column_name].to_numpy()
#     if sort_times is not None:
#         sort_times = sort_times[trials_indices]
#     centerIn_times = events_times.iloc[trials_indices][centerIn_times_column_name].to_numpy()
#     centerOut_times = events_times.iloc[trials_indices][centerOut_times_column_name].to_numpy()
#     sideIn_times = events_times.iloc[trials_indices][sideIn_times_column_name].to_numpy()
#     trialEnd_times = np.append(centerIn_times[1:], np.NAN)
#     marked_events = np.column_stack((centerIn_times, centerOut_times, sideIn_times, trialEnd_times))
#     trials_choices = events_times.iloc[trials_indices][trial_choice_column_name].to_numpy()
#     trials_rewarded = events_times.iloc[trials_indices][trial_rewarded_column_name].to_numpy()
#     trials_annotations = {"choice": trials_choices,
#                           "rewarded": trials_rewarded,
#                           "choice_prev": np.insert(trials_choices[:-1], 0,
#                                                    np.NAN),
#                           "rewarded_prev": np.insert(trials_rewarded[:-1], 0,
#                                                      np.NAN)}
    trials_labels = np.array([str(i) for i in trials_indices])
    choice_annotations = ["surebet" if choice_bino[r]==0 else "lottery"
                          for r in range(n_trials)]
    trials_annotations = {"choice": choice_annotations}
    trials_colors = [color_trials_surebet \
                     if choice_bino[r]==0 else color_trials_lottery \
                     for r in range(n_trials)]
    # plot lower bound history
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=lower_bound_hist)
    fig.write_image(lowerBoundHistVsIterNoFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsIterNoFigFilenamePattern.format("html"))

    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(elapsedTimeHist=elapsed_time_hist, lowerBoundHist=lower_bound_hist)
    fig.write_image(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsElapsedTimeFigFilenamePattern.format("html"))

    # plot estimated latent across trials
    test_mu_k, test_var_k = model.predictLatents(times=torch.from_numpy(trial_times))
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(times=trial_times, latentsMeans=test_mu_k, latentsSTDs=torch.sqrt(test_var_k), latentToPlot=latent_to_plot, trials_colors=trials_colors, xlabel="Time (msec)")
    fig.write_image(latentsFigFilenamePattern.format("png"))
    fig.write_html(latentsFigFilenamePattern.format("html"))

    # plot orthonormalized estimated latent across trials
    # test_mu_k, test_var_k = model.predictLatents(times=torch.from_numpy(trial_times))
    test_mu_k_np = [test_mu_k[r].detach().numpy() for r in range(len(test_mu_k))]
    estimated_C, estimated_d = model.getSVEmbeddingParams()
    estimatedC_np = estimated_C.detach().numpy()
    fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
        times=trial_times, latentsMeans=test_mu_k_np, latentToPlot=latent_to_plot,
        C=estimatedC_np,
        # align_event=align_times, marked_events=marked_events,
        trials_labels=trials_labels,
        trials_annotations=trials_annotations,
        trials_colors=trials_colors,
        xlabel="Time (msec)")
    fig.write_image(orthonormalizedLatentsFigFilenamePattern.format("png"))
    fig.write_html(orthonormalizedLatentsFigFilenamePattern.format("html"))

    fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
        times=trial_times, latentsMeans=test_mu_k_np, 
        C=estimatedC_np, latentsToPlot=[0, 1, 2],
        # align_event=align_times, marked_events=marked_events,
        trials_labels=trials_labels,
        trials_annotations=trials_annotations,
        trials_colors=trials_colors,
    )
    fig.write_image(orthonormalizedLatents3DFigFilenamePattern.format("png"))
    fig.write_html(orthonormalizedLatents3DFigFilenamePattern.format("html"))

    # title = "Latent {:d}, sorted by {:s}".format(latent_to_plot, sort_times_column_name)
    title = "Latent {:d}".format(latent_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentImageOneNeuronAllTrials(
        times=trial_times, latentsMeans=test_mu_k_np, latentToPlot=latent_to_plot,
        C=estimatedC_np,
        # sort_event=sort_times, align_event=align_times, marked_events=marked_events,
        trials_labels=trials_labels,
        # trials_annotations=trials_annotations,
        title=title, 
    )
    fig.write_image(orthonormalizedLatentsImageFigFilenamePattern.format("png"))
    fig.write_html(orthonormalizedLatentsImageFigFilenamePattern.format("html"))

    # plot embedding
    embedding_means, embedding_vars = model.predictEmbedding(times=torch.from_numpy(trial_times))
    embedding_means = embedding_means.detach().numpy()
    embedding_vars = embedding_vars.detach().numpy()
    title = "Neuron {:d}".format(neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(
        times=trial_times,
        embeddingsMeans=embedding_means[:,:,neuron_to_plot_index],
        embeddingsSTDs=np.sqrt(embedding_vars[:,:,neuron_to_plot_index]),
        title=title, trials_colors=trials_colors)
    fig.write_image(embeddingsFigFilenamePattern.format("png"))
    fig.write_html(embeddingsFigFilenamePattern.format("html"))

    # calculate expected CIF values (for KS test and CIF plots)
    with torch.no_grad():
        e_pos_CIF_values = model.computeExpectedPosteriorCIFs(times=torch.from_numpy(trial_times))
    spikes_times_ks = spikes_times[trial_to_plot][neuron_to_plot_index].numpy()
    cif_values_ks = e_pos_CIF_values[trial_to_plot][neuron_to_plot_index]
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trial_to_plot,
                                                           neuron_to_plot,
                                                           len(spikes_times_ks))

    # CIF
    fig = svGPFA.plot.plotUtilsPlotly.getPlotCIF(times=trial_times,
                                                 values=e_pos_CIF_values[trial_to_plot][neuron_to_plot_index], title=title)
    fig.write_image(CIFFigFilenamePattern.format("png"))
    fig.write_html(CIFFigFilenamePattern.format("html"))

    # CIF image
    # title = "Neuron {:d}, sorted by {:s}".format(neuron_to_plot, sort_times_column_name)
    title = "Neuron {:d}".format(neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
        times=trial_times, cif_values=e_pos_CIF_values,
        neuron_index=neuron_to_plot_index,
        # sort_event=sideIn_times, align_event=centerOut_times, marked_events=marked_events,
        trials_colors=trials_colors,
        title=title)
    fig.write_image(CIFsOneNeuronAllTrialsFigFilenamePattern.format("png"))
    fig.write_html(CIFsOneNeuronAllTrialsFigFilenamePattern.format("html"))

    # plot KS test time rescaling (numerical correction)
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = \
            gcnu_common.stats.pointProcesses.tests.KSTestTimeRescalingNumericalCorrection(
                spikesTimes=spikes_times_ks,
                cifTimes=torch.from_numpy(trial_times),
                cifValues=cif_values_ks, gamma=ksTestGamma)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(
        diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx,
        estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb,
        title=title)
    fig.write_image(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("png"))
    fig.write_html(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("html"))

    # ROC predictive analysis
    pk = cif_values_ks*dt_CIF
    bins = pd.interval_range(start=min(trial_times), end=max(trial_times), periods=len(pk))
    cutRes, _ = pd.cut(spikes_times_ks, bins=bins, retbins=True)
    Y = torch.from_numpy(cutRes.value_counts().values)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, pk, pos_label=1)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotResROCAnalysis(fpr=fpr, tpr=tpr, auc=roc_auc, title=title)
    fig.write_image(rocFigFilenamePattern.format("png"))
    fig.write_html(rocFigFilenamePattern.format("html"))

    # plot embedding parameters
    # estimated_C, estimated_d = model.getSVEmbeddingParams()
    fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingParams(C=estimated_C.numpy(), d=estimated_d.numpy())
    fig.write_image(embeddingParamsFigFilenamePattern.format("png"))
    fig.write_html(embeddingParamsFigFilenamePattern.format("html"))

    # plot kernel parameters
    kernelsParams = model.getKernelsParams()
    kernelsTypes = [type(kernel).__name__ for kernel in model.getKernels()]
    fig = svGPFA.plot.plotUtilsPlotly.getPlotKernelsParams(
        kernelsTypes=kernelsTypes, kernelsParams=kernelsParams)
    fig.write_image(kernelsParamsFigFilenamePattern.format("png"))
    fig.write_html(kernelsParamsFigFilenamePattern.format("html"))

    import pdb; pdb.set_trace()

if __name__=="__main__":
    main(sys.argv)
