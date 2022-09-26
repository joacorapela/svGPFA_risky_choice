clear all;

session_id = 256511;
packed_data_filename_pattern = "../../data/risky_%d_add.mat";
unpacked_data_filename_pattern = "../../data/risky_%d_add_unpacked_for_python.mat";

packed_data_filename = sprintf(packed_data_filename_pattern, session_id);
unpacked_data_filename = sprintf(unpacked_data_filename_pattern, session_id);

load(packed_data_filename);

cue_t = PD.('cue_t');
target_t = PD.('target_t');
rt_choice = PD.('RT_choice');
rt_fixation = PD.('RT_fixation');
lottery_mag = PD.('lottery_mag');
surebet_mag = PD.('sb_mag');
subject_choice = PD.('subj_choice');
current_reward = PD.('reward_received');
save(unpacked_data_filename)
