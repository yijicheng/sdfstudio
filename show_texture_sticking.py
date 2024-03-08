import imageio
import numpy as np
import os
from PIL import Image
from skimage.transform import resize

# files = [f'comp_stylenerf_1024/000001_{127-i:0>3}.png' for i in range(128)]
# files = [f'STYLESDF_FFHQ1024_comp/90{127-i:0>6}.png' for i in range(128)]
# files = [f'GRAM1024_64_SSRF_BGVI_LITE_Z_L2_PATCH_comp/grid_1154_{i}.png' for i in range(128)]
# files = [f'/root/blob2/utils/Rodin/output_1115/2d_sr_ablation_3d_consistency_rodinv1_new_80.5/restored_imgs/Alexis_Shakin_3JL42/{i:03}.png' for i in range(360)]
# files = [f'/root/blob2/utils/Rodin/output_1115/3d_consistency_ours_80.2_Render/Alexis_Shakin_3JL42/restore/ngp_ep0000_{i:04}_rgb.png' for i in range(360)]
# files = [f'/root/blob2/utils/Rodin/output_1115/3d_consistency_gt_80.2_Render/Alexis_Shakin_3JL42/restore/ngp_ep0000_{i:04}_rgb.png' for i in range(360)]

# files = [f'/root/blob2/utils/Rodin/output_1114/rodinv1_conditional_video_30degrees/gfpgan_refine_Albert_Chait_WTVNL/restored_imgs/{i:03}.png' for i in range(360)]
files = [f'/root/blob2/utils/Rodin/output_0305/3d_consistency_ours_Render/Albert_Chait_WTVNL/restore/ngp_ep0000_{i:04}_rgb.png' for i in range(360)]
# files = [f'/root/blob2/utils/Rodin/output_0305/3d_consistency_gt_Render/Albert_Chait_WTVNL/restore/ngp_ep0000_{i:04}_rgb.png' for i in range(360)]


# files = [f'epigraf_epi/seed-0013-{i:04d}.png' for i in range(128)]
# files = [f'gmpi_epi/0002_{i:04d}.png' for i in range(128)]
pos = [512-64, 96]
# pos = [512-64-50, 256]
pos = [512-64, 680]
pos = [256-32, 16]  # eg3d 5 0
pos = [256-32-48, 160]  # eg3d 5 1
pos = [256-32-32, 32]  # eg3d 2 0
pos = [256-32-48, 145]  # eg3d 2 1


pos = [280, 670] # w, h

size = 128 # 64

result = np.zeros((360, size, 3), dtype=np.uint8)

# for i, filename in enumerate(files):
#     img = imageio.imread(filename)[..., :3]
#     result[:, i] = img[pos[1]:pos[1]+128, pos[0]]
#     if i in (0, 63, 127):
#         img[pos[1]:pos[1]+128, pos[0]-1:pos[0]+1] = [255, 0, 0]
#         imageio.imsave(f'texture_sticking_{i}.png', img)

# for i, filename in enumerate(files):
#     img = imageio.imread(filename)[..., :3]
#     result[:, i] = img[pos[1], pos[0]:pos[0]+128]
#     if i in (0, 63, 127):
#         img[pos[1]-3:pos[1]+3, pos[0]-3:pos[0]+128+3] = [0, 255, 0]
#         imageio.imsave(f'texture_sticking_{i}.png', img)

for i, filename in enumerate(files):
    img = imageio.imread(filename)
    img = Image.fromarray(img).resize((1024, 1024))
    img = np.array(img)
    result[i, :] = img[pos[1], pos[0]:pos[0]+size]
    if i in (0, 129, 259):
        img[pos[1]-1:pos[1]+1, pos[0]-1:pos[0]+size+1] = [0, 255, 0]
        imageio.imsave(f'texture_sticking_{i}.png', img)

result = Image.fromarray(result)
result.resize((1024, 1024 * 3), Image.LANCZOS).save('texture_sticking.png')

# # transpose
# result = result.transpose(1, 0, 2)
# result = Image.fromarray(result)
# result.resize((600, 200), Image.LANCZOS).save('texture_sticking.png')