CUDA_VISIBLE_DEVICES=0 ns-train neus --pipeline.model.sdf-field.inside-outside False --vis tensorboard rodin-data --data data/ablation_3d_consistency_rodinv1_300imgs_new_gfpgan_refine/restored_imgs --subject Alexis_Shakin_3JL42
CUDA_VISIBLE_DEVICES=1 ns-train neus --pipeline.model.sdf-field.inside-outside False --vis tensorboard rodin-data --data data/Alexis_Shakin_3JL42 --subject Alexis_Shakin_3JL42