# original paper: aninerf

# evaluating on training poses for aninerf
python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True

# evaluating on novel poses for aninerf
python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume True aninerf_animation True init_aninerf aninerf_s9p test_novel_pose True

# visualizing novel view of 0th frame for aninerf
python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True vis_novel_view True begin_ith_frame 0

# visualizing animation of 3rd camera for aninerf
python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True vis_pose_sequence True test_view "3,"

# generating posed mesh for aninerf
python run.py --type visualize --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p vis_posed_mesh True

# training base model for aninerf
python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume False

# training the blend weight fields of unseen human poses
python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume False aninerf_animation True init_aninerf aninerf_s9p




# extension: aligned_aninerf_lbw

# evaluating on training poses for aligned_aninerf_lbw
python run.py --type evaluate --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p resume True

# evaluating on novel poses for aligned_aninerf_lbw
python run.py --type evaluate --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p_full resume True aninerf_animation True init_aninerf aligned_aninerf_lbw_s9p test_novel_pose True

# visualizing novel view of 0th frame for aligned_aninerf_lbw
python run.py --type visualize --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p resume True vis_novel_view True begin_ith_frame 0

# visualizing animation of 3rd camera for aligned_aninerf_lbw
python run.py --type visualize --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p resume True vis_pose_sequence True test_view "3,"

# generating posed mesh for aligned_aninerf_lbw
python run.py --type visualize --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p vis_posed_mesh True

# training base model for aligned_aninerf_lbw
python train_net.py --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p resume False




# extension: aligned_aninerf_pdf

# evaluating on training poses for aligned_aninerf_pdf
python run.py --type evaluate --cfg_file configs/aligned_nerf_pdf/aligned_aninerf_pdf_s9p.yaml exp_name aligned_aninerf_pdf_s9p resume True

# evaluating on novel poses for aligned_aninerf_pdf
python run.py --type evaluate --cfg_file configs/aligned_nerf_pdf/aligned_aninerf_pdf_s9p.yaml exp_name aligned_aninerf_pdf_s9p resume True test_novel_pose True

# visualizing novel view of 0th frame for aligned_aninerf_pdf
python run.py --type visualize --cfg_file configs/aligned_nerf_pdf/aligned_aninerf_pdf_s9p.yaml exp_name aligned_aninerf_pdf_s9p resume True vis_novel_view True begin_ith_frame 0

# visualizing animation of 3rd camera for aligned_aninerf_pdf
python run.py --type visualize --cfg_file configs/aligned_nerf_pdf/aligned_aninerf_pdf_s9p.yaml exp_name aligned_aninerf_pdf_s9p resume True vis_pose_sequence True test_view "3,"

# generating posed mesh for aligned_aninerf_pdf
python run.py --type visualize --cfg_file configs/aligned_nerf_pdf/aligned_aninerf_pdf_s9p.yaml exp_name aligned_aninerf_pdf_s9p vis_posed_mesh True

# training base model for aligned_aninerf_pdf
python train_net.py --cfg_file configs/aligned_nerf_pdf/aligned_aninerf_pdf_s9p.yaml exp_name aligned_aninerf_pdf_s9p resume False




# extension: anisdf_pdf

# evaluating on training poses for anisdf_pdf
python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True

# evaluating on novel poses for anisdf_pdf
python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True test_novel_pose True

# visualizing novel view of 0th frame for anisdf_pdf
python run.py --type visualize --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True vis_novel_view True begin_ith_frame 0

# visualizing animation of 3rd camera for anisdf_pdf
python run.py --type visualize --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True vis_pose_sequence True test_view "3,"

# generating posed mesh for anisdf_pdf
python run.py --type visualize --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p vis_posed_mesh True

# training base model for anisdf_pdf
python train_net.py --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume False