clear all;

session_id = 256511;
packed_data_filename_pattern = "../../data/risky_choice_data/risky_%d.mat";
unpacked_data_filename_pattern = "../../data/risky_choice_data/risky_%d_unpacked_for_python.mat";

packed_data_filename = sprintf(packed_data_filename_pattern, session_id);
unpacked_data_filename = sprintf(unpacked_data_filename_pattern, session_id);

load(packed_data_filename);

clusters_labels = CD.info.('label');
cell_id = CD.info.('cellid');

save(unpacked_data_filename)
