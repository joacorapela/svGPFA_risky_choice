import sys
import os
import random
import pickle
import argparse
import configparser
import numpy as np
import torch

import gcnu_common.utils.config_dict
import gcnu_common.utils.argparse
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.configUtils
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument("--session_id", help="session id", type=int,
                        default=256511)
    parser.add_argument("--n_latents", help="number of latent variables",
                        type=int, default=3)
    parser.add_argument("--max_rt_fixation", help="maximum fixation reaction "
                        "time (sec)", type=float, default=5.0)
    parser.add_argument("--trials_start_time", help="trials_start_time",
                        type=float, default=-1.0)
    parser.add_argument("--trials_end_time", help="trials_end time",
                        type=float, default=3.0)
    parser.add_argument("--em_max_iter", help="maximum number of EM iterations",
                        type=int, default=50)
    parser.add_argument("--epoched_spikes_filename_pattern",
                        help="filename containing the epoched spikes",
                        type=str,
                        default="../results/risky_{:d}_epoched_spikes.pickle")
    parser.add_argument("--model_save_filename_pattern",
                        help="model save filename pattern",
                        type=str,
                        default="../results/{:08d}_estimatedModel.pickle")
    parser.add_argument("--estimRes_metadata_filename_pattern",
                        help="estimation result metadata filename pattern",
                        type=str,
                        default="../results/{:08d}_estimation_metaData.ini")

    args, remaining = parser.parse_known_args()
    gcnu_common.utils.argparse.add_remaining_to_populated_args(
        populated=args, remaining=remaining)
    session_id = args.session_id
    n_latents = args.n_latents
    max_rt_fixation = args.max_rt_fixation
    trials_start_time = args.trials_start_time
    trials_end_time = args.trials_end_time
    em_max_iter = args.em_max_iter
    epoched_spikes_filename_pattern = \
        args.epoched_spikes_filename_pattern
    model_save_filename_pattern = args.model_save_filename_pattern
    estimRes_metadata_filename_pattern = args.estimRes_metadata_filename_pattern

    # load data
    epoched_spikes_filename = epoched_spikes_filename_pattern.format(session_id)
    with open(epoched_spikes_filename, "rb") as f:
        loadRes = pickle.load(f)
    spikes_times = loadRes["spikes_times"]
    trials_start_times = loadRes["trials_start_times"]
    trials_end_times = loadRes["trials_end_times"]
    rt_fixation = loadRes["rt_fixation"]

    # begin exclude trials with rt_fixation lonter than max_rt_fixation
    valid_trials_indices = np.where(rt_fixation < max_rt_fixation)[0]
    print("Removed {:d} trials out of {:d} trials".format(
        len(rt_fixation)-len(valid_trials_indices), len(rt_fixation)))
    spikes_times = [spikes_times[i] for i in valid_trials_indices]
    trials_start_times = trials_start_times[valid_trials_indices]
    trials_end_times = trials_end_times[valid_trials_indices]
    # end exclude trials with rt_fixation lonter than max_rt_fixation

    n_trials = len(spikes_times)
    n_neurons = len(spikes_times[0])

    # convert spikes_times to torch tensors
    spikes_times = [[torch.from_numpy(spikes_times[r][n])
                     for n in range(n_neurons)] for r in range(n_trials)]

    # build initial parameters
    #    build dynamic parameters
    args = vars(args)
    args["trials_start_times"] = np.array2string(trials_start_times[:, 0],
                                                 separator=",")
    args["trials_end_times"] = np.array2string(trials_end_times[:, 0],
                                               separator=",")
    args_info = svGPFA.utils.initUtils.getArgsInfo()
    dynamic_params = svGPFA.utils.initUtils.getParamsDictFromArgs(
        n_latents=n_latents, n_trials=n_trials, args=args,
        args_info=args_info)
    #    build configuration default parameters
    default_params = svGPFA.utils.initUtils.getDefaultParamsDict(
        n_neurons=n_neurons, n_trials=n_trials, n_latents=n_latents,
        trials_start_time=trials_start_time, trials_end_time=trials_end_time,
        em_max_iter=em_max_iter)
    #    finally, extract initial parameters from the dynamic
    #    and default parameters
    params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(
        n_trials=n_trials, n_neurons=n_neurons, n_latents=n_latents,
        dynamic_params=dynamic_params,
        default_params=default_params)
    kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]
    optim_method = params["optim_params"]["optim_method"]
    prior_cov_reg_param = params["optim_params"]["prior_cov_reg_param"]

    # build model_save_filename
    estPrefixUsed = True
    while estPrefixUsed:
        estResNumber = random.randint(0, 10**8)
        estimResMetaDataFilename = \
            estimRes_metadata_filename_pattern.format(estResNumber)
        if not os.path.exists(estimResMetaDataFilename):
            estPrefixUsed = False
    model_save_filename = model_save_filename_pattern.format(estResNumber)

    # build kernels
    kernels = svGPFA.utils.miscUtils.buildKernels(
        kernels_types=kernels_types, kernels_params=kernels_params0)

    # create model
    kernelMatrixInvMethod = svGPFA.stats.svGPFAModelFactory.kernelMatrixInvChol
    indPointsCovRep = svGPFA.stats.svGPFAModelFactory.indPointsCovChol
    model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.buildModelPyTorch(
        conditionalDist=svGPFA.stats.svGPFAModelFactory.PointProcess,
        linkFunction=svGPFA.stats.svGPFAModelFactory.ExponentialLink,
        embeddingType=svGPFA.stats.svGPFAModelFactory.LinearEmbedding,
        kernels=kernels, kernelMatrixInvMethod=kernelMatrixInvMethod,
        indPointsCovRep=indPointsCovRep)

    model.setInitialParamsAndData(
        measurements=spikes_times,
        initialParams=params["initial_params"],
        eLLCalculationParams=params["quad_params"],
        priorCovRegParam=prior_cov_reg_param)

    # maximize lower bound
    svEM = svGPFA.stats.svEM.SVEM_PyTorch()
    lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
        svEM.maximize(model=model, optim_params=params["optim_params"],
                      method=optim_method)

    # save estimated values
    estimResConfig = configparser.ConfigParser()
    estimResConfig["optim_params"] = params["optim_params"]
    with open(estimResMetaDataFilename, "w") as f:
        estimResConfig.write(f)

    resultsToSave = {"lowerBoundHist": lowerBoundHist,
                     "elapsedTimeHist": elapsedTimeHist,
                     "terminationInfo": terminationInfo,
                     "iterationModelParams": iterationsModelParams,
                     "model": model,
                     "valid_trials_indices": valid_trials_indices}
    with open(model_save_filename, "wb") as f:
        pickle.dump(resultsToSave, f)
    print("Saved results to {:s}".format(model_save_filename))


if __name__ == "__main__":
    main(sys.argv)
