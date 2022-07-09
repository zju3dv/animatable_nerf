# Human3.6M
python train_net.py --cfg_file configs/aninerf_s1p.yaml exp_name aninerf_s1p resume False
python train_net.py --cfg_file configs/aninerf_s1p.yaml exp_name aninerf_s1p_full resume False aninerf_animation True init_aninerf aninerf_s1p

python train_net.py --cfg_file configs/aninerf_s5p.yaml exp_name aninerf_s5p resume False
python train_net.py --cfg_file configs/aninerf_s5p.yaml exp_name aninerf_s5p_full resume False aninerf_animation True init_aninerf aninerf_s5p

python train_net.py --cfg_file configs/aninerf_s6p.yaml exp_name aninerf_s6p resume False
python train_net.py --cfg_file configs/aninerf_s6p.yaml exp_name aninerf_s6p_full resume False aninerf_animation True init_aninerf aninerf_s6p

python train_net.py --cfg_file configs/aninerf_s7p.yaml exp_name aninerf_s7p resume False
python train_net.py --cfg_file configs/aninerf_s7p.yaml exp_name aninerf_s7p_full resume False aninerf_animation True init_aninerf aninerf_s7p

python train_net.py --cfg_file configs/aninerf_s8p.yaml exp_name aninerf_s8p resume False
python train_net.py --cfg_file configs/aninerf_s8p.yaml exp_name aninerf_s8p_full resume False aninerf_animation True init_aninerf aninerf_s8p

python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p resume False
python train_net.py --cfg_file configs/aninerf_s9p.yaml exp_name aninerf_s9p_full resume False aninerf_animation True init_aninerf aninerf_s9p

python train_net.py --cfg_file configs/aninerf_s11p.yaml exp_name aninerf_s11p resume False
python train_net.py --cfg_file configs/aninerf_s11p.yaml exp_name aninerf_s11p_full resume False aninerf_animation True init_aninerf aninerf_s11p

# ZJU-MoCap
python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume False
python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313_full resume False aninerf_animation True init_aninerf aninerf_313

python train_net.py --cfg_file configs/aninerf_315.yaml exp_name aninerf_315 resume False
python train_net.py --cfg_file configs/aninerf_315.yaml exp_name aninerf_315_full resume False aninerf_animation True init_aninerf aninerf_315

python train_net.py --cfg_file configs/aninerf_377.yaml exp_name aninerf_377 resume False
python train_net.py --cfg_file configs/aninerf_377.yaml exp_name aninerf_377_full resume False aninerf_animation True init_aninerf aninerf_377

python train_net.py --cfg_file configs/aninerf_386.yaml exp_name aninerf_386 resume False
python train_net.py --cfg_file configs/aninerf_386.yaml exp_name aninerf_386_full resume False aninerf_animation True init_aninerf aninerf_386

# Extended version: training with S9 of Human3.6M as an example

# Vanilla Animatable NeRF
python train_net.py --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p resume False
python train_net.py --cfg_file configs/aligned_nerf_lbw/aligned_aninerf_lbw_s9p.yaml exp_name aligned_aninerf_lbw_s9p_full resume False aninerf_animation True init_aninerf aligned_aninerf_lbw_s9p

# Pose dependent displacement field + Animatable NeRF
python train_net.py --cfg_file configs/aligned_nerf_pdf/aligned_aninerf_pdf_s9p.yaml exp_name aligned_aninerf_pdf_s9p resume False

# Pose dependent displacement field + SDF field (full Animatable Neural Fields)
python train_net.py --cfg_file configs/sdf_pdf/anisdf_pdf_s9p.yaml exp_name anisdf_pdf_s9p resume False
