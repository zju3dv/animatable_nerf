# Human3.6M
python run.py --type evaluate --cfg_file configs/aninerf_s1p.yaml exp_name aninerf_s1p resume True
python run.py --type evaluate --cfg_file configs/aninerf_s1p.yaml exp_name aninerf_s1p_full resume True aninerf_animation True init_aninerf aninerf_s1p test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_s5p.yaml exp_name aninerf_s5p resume True
python run.py --type evaluate --cfg_file configs/aninerf_s5p.yaml exp_name aninerf_s5p_full resume True aninerf_animation True init_aninerf aninerf_s5p test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_s6p.yaml exp_name aninerf_s6p resume True
python run.py --type evaluate --cfg_file configs/aninerf_s6p.yaml exp_name aninerf_s6p_full resume True aninerf_animation True init_aninerf aninerf_s6p test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_s7p.yaml exp_name aninerf_s7p resume True
python run.py --type evaluate --cfg_file configs/aninerf_s7p.yaml exp_name aninerf_s7p_full resume True aninerf_animation True init_aninerf aninerf_s7p test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_s8p.yaml exp_name aninerf_s8p resume True
python run.py --type evaluate --cfg_file configs/aninerf_s8p.yaml exp_name aninerf_s8p_full resume True aninerf_animation True init_aninerf aninerf_s8p test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume True
python run.py --type evaluate --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume True aninerf_animation True init_aninerf aninerf_s9p test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_s11p.yaml exp_name aninerf_s11p resume True
python run.py --type evaluate --cfg_file configs/aninerf_s11p.yaml exp_name aninerf_s11p_full resume True aninerf_animation True init_aninerf aninerf_s11p test_novel_pose True

# ZJU-MoCap
python run.py --type evaluate --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume True
python run.py --type evaluate --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_full resume True aninerf_animation True init_aninerf aninerf_313 test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_315.yaml exp_name aninerf_315 resume True
python run.py --type evaluate --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_full resume True aninerf_animation True init_aninerf aninerf_315 test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_377.yaml exp_name aninerf_377 resume True
python run.py --type evaluate --cfg_file configs/aninerf_377.yaml exp_name aninerf_377_full resume True aninerf_animation True init_aninerf aninerf_377 test_novel_pose True

python run.py --type evaluate --cfg_file configs/aninerf_386.yaml exp_name aninerf_386 resume True
python run.py --type evaluate --cfg_file configs/aninerf_386.yaml exp_name aninerf_386_full resume True aninerf_animation True init_aninerf aninerf_386 test_novel_pose True


# Extended version: image synthesis evaluation with S9 of Human3.6M as an example

# Vanilla Animatable NeRF
python run.py --type evaluate --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p resume True
python run.py --type evaluate --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p_full resume True aninerf_animation True init_aninerf aligned_aninerf_lbw_s9p test_novel_pose True

# Pose-dependent displacement field + Animatable NeRF
python run.py --type evaluate --cfg_file configs/aligned_nerf_pdf/aligned_aninerf_pdf_s9p.yaml exp_name aligned_aninerf_pdf_s9p resume True
python run.py --type evaluate --cfg_file configs/aligned_nerf_pdf/aligned_aninerf_pdf_s9p.yaml exp_name aligned_aninerf_pdf_s9p test_novel_pose True resume True

# Pose-dependent displacement field + SDF field (full Animatable Neural Fields)
python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume True
python run.py --type evaluate --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p test_novel_pose True resume True 

# Extended version: 3D reconstruction with S9 of Human3.6M as an example (SDF-PDF version). Only for the SyntheticHuman dataset
python run.py --type evaluate --cfg_file configs/sdf_pdf/synhuman/anisdf_pdf_rp01_nathan.yaml exp_name anisdf_pdf_nathan resume True vis_posed_mesh True
