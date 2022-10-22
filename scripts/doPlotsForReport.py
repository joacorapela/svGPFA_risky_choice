
import sys
import numpy as np
import torch
import pickle
import argparse
import pandas as pd
import sklearn.metrics
import plotly.express as px

import gcnu_common.utils.neuralDataAnalysis
import gcnu_common.stats.pointProcesses.tests
import svGPFA.utils.configUtils
import svGPFA.utils.initUtils
import svGPFA.utils.miscUtils
import svGPFA.plot.plotUtilsPlotly


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("estResNumber", help="estimation result number",
                        type=int)
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--latent_to_plot", help="trial to plot", type=int,
                        default=0)
    parser.add_argument("--neuron_to_plot", help="neuron to plot", type=int,
                        default=23453)
    parser.add_argument("--trial_to_plot", help="trial to plot", type=int,
                        default=0)
    parser.add_argument("--n_time_steps_CIF", help="number of stime steps in "
                        "CIF plots", type=int, default=100)
    parser.add_argument("--ksTestGamma", help="gamma value for KS test",
                        type=int, default=10)
    parser.add_argument("--nTestPoints", help="number of test points where to "
                        "plot latents", type=int, default=2000)
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
#                         default="../data/022822/s008_tab_m1113182_LR__20210516_173815__probabilistic_switching.df.csv")
#     parser.add_argument("--region_spikes_times_filename_pattern",
#                         help="region spikes times filename pattern",
#                         type=str,
#                         default="../results/00000000_region{:s}_spikes_times_epochedaligned__last_center_out.{:s}")

    args = parser.parse_args()

    estResNumber = args.estResNumber
    session_id = args.session_id
    latent_to_plot = args.latent_to_plot
    neuron_to_plot = args.neuron_to_plot
    trial_to_plot = args.trial_to_plot
    n_time_steps_CIF = args.n_time_steps_CIF
    ksTestGamma = args.ksTestGamma
    color_trials_surebet = args.color_trials_surebet
    color_trials_lottery = args.color_trials_lottery
    color_center_on = args.color_center_on
    color_center_poke = args.color_center_poke
    color_choice_on = args.color_choice_on
    color_choice_poke = args.color_choice_poke
    lm_colorscale = args.lm_colorscale
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
    estimResMetaDataFilename = "../results/{:08d}_estimation_metaData.ini".format(estResNumber)
    modelSaveFilename = "../results/{:08d}_estimatedModel.pickle".format(estResNumber)
    lowerBoundHistVsIterNoFigFilenamePattern = "../figures/{:08d}_lowerBoundHistVsIterNo.{{:s}}".format(estResNumber)
    lowerBoundHistVsElapsedTimeFigFilenamePattern = "../figures/{:08d}_lowerBoundHistVsElapsedTime.{{:s}}".format(estResNumber)
    latentsFigFilenamePattern = "../figures/{:08d}_estimatedLatent_{{:s}}_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    orthonormalizedLatentsFigFilenamePattern = "../figures/{:08d}_orthonormalized_estimatedLatent_{{:s}}_latent{:03d}.{{:s}}".format(estResNumber, latent_to_plot)
    orthonormalizedLatents3DFigFilenamePattern = "../figures/{:08d}_3Dorthonormalized_estimatedLatent.{{:s}}".format(estResNumber)
    orthonormalizedLatentsImageFigFilenamePattern = "../figures/{:08d}_orthonormalized_latent{:03d}_image.{{:s}}".format(estResNumber, latent_to_plot)
    embeddingsFigFilenamePattern = "../figures/{:08d}_estimatedEmbedding_{{:s}}_neuron{:d}.{{:s}}".format(estResNumber, neuron_to_plot)
    embeddingParamsFigFilenamePattern = "../figures/{:08d}_estimatedEmbeddingParams.{{:s}}".format(estResNumber)
    CIFFigFilenamePattern = "../figures/{:08d}_CIF_trial{:03d}_neuron{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
    CIFsOneNeuronAllTrialsFigFilenamePattern = "../figures/{:08d}_CIFsAllTrials_{{:s}}_neuron{:d}.{{:s}}".format(estResNumber, neuron_to_plot)
    ksTestTimeRescalingNumericalCorrectionFigFilenamePattern = "../figures/{:08d}_ksTestTimeRescaling_numericalCorrection_trial{:03d}_neuron{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
    rocFigFilenamePattern = "../figures/{:08d}_rocAnalysis_trial{:03d}_neuron{:d}.{{:s}}".format(estResNumber, trial_to_plot, neuron_to_plot)
    kernelsParamsFigFilenamePattern = "../figures/{:08d}_estimatedKernelsParams.{{:s}}".format(estResNumber)

    # load data
    with open(epoched_spikes_filename, "rb") as f:
        loadRes = pickle.load(f)
    spikes_times = loadRes["spikes_times"]
    cell_ids = loadRes["cell_ids"]
    trials_start_times = loadRes["trials_start_times"]
    trials_end_times = loadRes["trials_end_times"]
    rt_fixation = loadRes["rt_fixation"]
    trials_start_times = loadRes["trials_start_times"]
    trials_end_times = loadRes["trials_end_times"]
    spikes_times = loadRes["spikes_times"]
    subject_choices = loadRes["subject_choices"]
    center_on_times = loadRes["center_on_times"]
    choice_on_times = loadRes["choice_on_times"]
    choice_poke_times = loadRes["choice_poke_times"]
    lottery_magnitude = loadRes["lottery_magnitude"].squeeze()

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

    with open(modelSaveFilename, "rb") as f:
        estResults = pickle.load(f)
    valid_trials_indices = estResults["valid_trials_indices"]

    trials_start_times = trials_start_times[valid_trials_indices]
    trials_end_times = trials_end_times[valid_trials_indices]
    spikes_times = [spikes_times[i] for i in valid_trials_indices]
    subject_choices = [subject_choices[i] for i in valid_trials_indices]
    center_on_times = [center_on_times[i] for i in valid_trials_indices]
    center_poke_times = [0.0 for i in valid_trials_indices]
    choice_on_times = [choice_on_times[i] for i in valid_trials_indices]
    choice_poke_times = [choice_poke_times[i] for i in valid_trials_indices]
    lottery_magnitude = [lottery_magnitude[i] for i in valid_trials_indices]
    unique_lottery_magnitudes = np.unique(lottery_magnitude).tolist()

    spikes_times = [[torch.tensor(spikes_times[r][n])
                     for n in range(len(spikes_times[r]))]
                    for r in range(len(spikes_times))]
    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])
    trials_indices = np.arange(n_trials)

    trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
        start_times=np.squeeze(trials_start_times),
        end_times=np.squeeze(trials_end_times),
        n_steps=n_time_steps_CIF)

    # trials_times_np = [[trials_times[r][n].numpy() for n in range(n_neurons)] for r in range(n_trials)]

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
#     trials_choices = events_times.iloc[trials_indices][trial_choice_column_name].to_numpy()
#     trials_rewarded = events_times.iloc[trials_indices][trial_rewarded_column_name].to_numpy()
#     trials_annotations = {"choice": trials_choices,
#                           "rewarded": trials_rewarded,
#                           "choice_prev": np.insert(trials_choices[:-1], 0,
#                                                    np.NAN),
#                           "rewarded_prev": np.insert(trials_rewarded[:-1], 0,
#                                                      np.NAN)}
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

    # plot lower bound history
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
        lowerBoundHist=lower_bound_hist)
    fig.write_image(lowerBoundHistVsIterNoFigFilenamePattern.format("png"))
    fig.write_html(lowerBoundHistVsIterNoFigFilenamePattern.format("html"))

    fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(
        elapsedTimeHist=elapsed_time_hist, lowerBoundHist=lower_bound_hist)
    fig.write_image(
        lowerBoundHistVsElapsedTimeFigFilenamePattern.format("png"))
    fig.write_html(
        lowerBoundHistVsElapsedTimeFigFilenamePattern.format("html"))

    # plot estimated latent across trials with trials colored by choice
    test_mu_k, test_var_k = model.predictLatents(times=trials_times)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
        times=trials_times.numpy(), latentsMeans=test_mu_k,
        latentsSTDs=torch.sqrt(test_var_k), latentToPlot=latent_to_plot,
        trials_colors=trials_colors_c, xlabel="Time (msec)")
    fig.write_image(latentsFigFilenamePattern.format("choice", "png"))
    fig.write_html(latentsFigFilenamePattern.format("choice", "html"))

    # plot estimated latent across trials with trials colored by lottery
    # magnitude
    test_mu_k, test_var_k = model.predictLatents(times=trials_times)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(
        times=trials_times.numpy(), latentsMeans=test_mu_k,
        latentsSTDs=torch.sqrt(test_var_k), latentToPlot=latent_to_plot,
        trials_colors=trials_colors_lm, xlabel="Time (msec)")
    fig.write_image(latentsFigFilenamePattern.format("lm", "png"))
    fig.write_html(latentsFigFilenamePattern.format("lm", "html"))

    # plot orthonormalized estimated latent across trials
    test_mu_k_np = [test_mu_k[r].detach().numpy()
                    for r in range(len(test_mu_k))]
    estimated_C, estimated_d = model.getSVEmbeddingParams()
    estimatedC_np = estimated_C.detach().numpy()
    fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
        trials_times=trials_times.numpy(), latentsMeans=test_mu_k_np,
        latentToPlot=latent_to_plot, C=estimatedC_np, align_event=align_times,
        marked_events=marked_events, marked_events_colors=marked_events_colors,
        trials_labels=trials_labels, trials_annotations=trials_annotations,
        trials_colors=trials_colors_c, xlabel="Time (msec)")
    fig.write_image(orthonormalizedLatentsFigFilenamePattern.format("choice",
                                                                    "png"))
    fig.write_html(orthonormalizedLatentsFigFilenamePattern.format("choice",
                                                                   "html"))

    # plot orthonormalized estimated latent across trials
    test_mu_k_np = [test_mu_k[r].detach().numpy()
                    for r in range(len(test_mu_k))]
    estimated_C, estimated_d = model.getSVEmbeddingParams()
    estimatedC_np = estimated_C.detach().numpy()
    fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentAcrossTrials(
        trials_times=trials_times.numpy(), latentsMeans=test_mu_k_np,
        latentToPlot=latent_to_plot, C=estimatedC_np, align_event=align_times,
        marked_events=marked_events, marked_events_colors=marked_events_colors,
        trials_labels=trials_labels, trials_annotations=trials_annotations,
        trials_colors=trials_colors_lm, xlabel="Time (msec)")
    fig.write_image(orthonormalizedLatentsFigFilenamePattern.format("lm",
                                                                    "png"))
    fig.write_html(orthonormalizedLatentsFigFilenamePattern.format("lm",
                                                                   "html"))

# with trials with different length the following plot is tricky
#
#     fig = svGPFA.plot.plotUtilsPlotly.get3DPlotOrthonormalizedLatentsAcrossTrials(
#         trials_times=trials_times.numpy(), latentsMeans=test_mu_k_np, 
#         C=estimatedC_np, latentsToPlot=[0, 1, 2],
#         # align_event=align_times, marked_events=marked_events,
#         trials_labels=trials_labels,
#         trials_annotations=trials_annotations,
#         trials_colors=trials_colors,
#     )
#     fig.write_image(orthonormalizedLatents3DFigFilenamePattern.format("png"))
#     fig.write_html(orthonormalizedLatents3DFigFilenamePattern.format("html"))

    # title = "Latent {:d}, sorted by {:s}".format(latent_to_plot, sort_times_column_name)
#     title = "Latent {:d}".format(latent_to_plot)
#     fig = svGPFA.plot.plotUtilsPlotly.getPlotOrthonormalizedLatentImageOneNeuronAllTrials(
#         times=trials_times, latentsMeans=test_mu_k_np, latentToPlot=latent_to_plot,
#         C=estimatedC_np,
#         # sort_event=sort_times, align_event=align_times, marked_events=marked_events,
#         trials_labels=trials_labels,
#         # trials_annotations=trials_annotations,
#         title=title, 
#     )
#     fig.write_image(orthonormalizedLatentsImageFigFilenamePattern.format("png"))
#     fig.write_html(orthonormalizedLatentsImageFigFilenamePattern.format("html"))

    # plot embedding colored by choice
    embedding_means, embedding_vars = model.predictEmbedding(
        times=trials_times)
    embedding_means = embedding_means.detach().numpy()
    embedding_vars = embedding_vars.detach().numpy()
    title = "Neuron {:d}".format(neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(
        times=trials_times.numpy(),
        embeddingsMeans=embedding_means[:, :, neuron_to_plot_index],
        embeddingsSTDs=np.sqrt(embedding_vars[:, :, neuron_to_plot_index]),
        title=title, trials_colors=trials_colors_c)
    fig.write_image(embeddingsFigFilenamePattern.format("choice", "png"))
    fig.write_html(embeddingsFigFilenamePattern.format("choice", "html"))

    # plot embedding colored by lottery magnitude
    embedding_means, embedding_vars = model.predictEmbedding(
        times=trials_times)
    embedding_means = embedding_means.detach().numpy()
    embedding_vars = embedding_vars.detach().numpy()
    title = "Neuron {:d}".format(neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(
        times=trials_times.numpy(),
        embeddingsMeans=embedding_means[:, :, neuron_to_plot_index],
        embeddingsSTDs=np.sqrt(embedding_vars[:, :, neuron_to_plot_index]),
        title=title, trials_colors=trials_colors_lm)
    fig.write_image(embeddingsFigFilenamePattern.format("lm", "png"))
    fig.write_html(embeddingsFigFilenamePattern.format("lm", "html"))

    # calculate expected CIF values (for KS test and CIF plots)
    with torch.no_grad():
        e_pos_CIF_values = model.computeExpectedPosteriorCIFs(times=trials_times)
    spikes_times_rn = spikes_times[trial_to_plot][neuron_to_plot_index].numpy()
    cif_values_rn = e_pos_CIF_values[trial_to_plot][neuron_to_plot_index]
    trial_times_r = trials_times[trial_to_plot, :, 0]
    title = "Trial {:d}, Neuron {:d} ({:d} spikes)".format(trial_to_plot,
                                                           neuron_to_plot,
                                                           len(spikes_times_rn))

    # CIF
    # fig = svGPFA.plot.plotUtilsPlotly.getPlotCIF(
    #     times=trials_times.numpy(),
    #     values=e_pos_CIF_values[trial_to_plot][neuron_to_plot_index],
    #     title=title)
    # fig.write_image(CIFFigFilenamePattern.format("png"))
    # fig.write_html(CIFFigFilenamePattern.format("html"))

    # CIF one neuron all trials
    # title = "Neuron {:d}, sorted by {:s}".format(neuron_to_plot,
    #                                              sort_times_column_name)
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
    fig.write_image(CIFsOneNeuronAllTrialsFigFilenamePattern.format("choice",
                                                                    "png"))
    fig.write_html(CIFsOneNeuronAllTrialsFigFilenamePattern.format("choice",
                                                                   "html"))

    title = "Neuron {:d}".format(neuron_to_plot)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotCIFsOneNeuronAllTrials(
        trials_times=trials_times.numpy(), cif_values=e_pos_CIF_values,
        neuron_index=neuron_to_plot_index,
        align_event=center_poke_times,
        marked_events=marked_events,
        marked_events_colors=marked_events_colors,
        trials_annotations=trials_annotations,
        trials_colors=trials_colors_lm,
        title=title)
    fig.write_image(CIFsOneNeuronAllTrialsFigFilenamePattern.format("lm",
                                                                    "png"))
    fig.write_html(CIFsOneNeuronAllTrialsFigFilenamePattern.format("lm",
                                                                   "html"))

    # plot KS test time rescaling (numerical correction)
    diffECDFsX, diffECDFsY, estECDFx, estECDFy, simECDFx, simECDFy, cb = \
        gcnu_common.stats.pointProcesses.tests.KSTestTimeRescalingNumericalCorrection(
                spikes_times=spikes_times_rn,
                cif_times=trial_times_r,
                cif_values=cif_values_rn, gamma=ksTestGamma)
    fig = svGPFA.plot.plotUtilsPlotly.getPlotResKSTestTimeRescalingNumericalCorrection(
        diffECDFsX=diffECDFsX, diffECDFsY=diffECDFsY, estECDFx=estECDFx,
        estECDFy=estECDFy, simECDFx=simECDFx, simECDFy=simECDFy, cb=cb,
        title=title)
    fig.write_image(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("png"))
    fig.write_html(ksTestTimeRescalingNumericalCorrectionFigFilenamePattern.format("html"))

    # ROC predictive analysis
    dt_CIF = (trial_times_r[-1] - trial_times_r[0]) / n_time_steps_CIF
    pk = cif_values_rn*dt_CIF
    bins = pd.interval_range(start=trial_times_r[0].item(),
                             end=trial_times_r[-1].item(), periods=len(pk))
    cutRes, _ = pd.cut(spikes_times_rn, bins=bins, retbins=True)
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
