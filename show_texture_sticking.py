import imageio
import numpy as np
import os
from PIL import Image

files = [f'comp_stylenerf_1024/000001_{127-i:0>3}.png' for i in range(128)]
# files = [f'STYLESDF_FFHQ1024_comp/90{127-i:0>6}.png' for i in range(128)]
# files = [f'GRAM1024_64_SSRF_BGVI_LITE_Z_L2_PATCH_comp/grid_1154_{i}.png' for i in range(128)]
files = [f'eg3d_epi_fix/0002{127-i:04d}.png' for i in range(128)]
# files = [f'epigraf_epi/seed-0013-{i:04d}.png' for i in range(128)]
# files = [f'gmpi_epi/0002_{i:04d}.png' for i in range(128)]
pos = [512-64, 96]
# pos = [512-64-50, 256]
pos = [512-64, 680]
pos = [256-32, 16]  # eg3d 5 0
pos = [256-32-48, 160]  # eg3d 5 1
pos = [256-32-32, 32]  # eg3d 2 0
pos = [256-32-48, 145]  # eg3d 2 1

size = 64 # 64

result = np.zeros((128, size, 3), dtype=np.uint8)

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
    img = imageio.imread(filename)[..., :3]
    result[i, :] = img[pos[1], pos[0]:pos[0]+size]
    if i in (0, 63, 127):
        img[pos[1]-1:pos[1]+1, pos[0]-1:pos[0]+size+1] = [0, 255, 0]
        imageio.imsave(f'texture_sticking_{i}.png', img)

# transpose
result = result.transpose(1, 0, 2)
result = Image.fromarray(result)
result.resize((600, 200), Image.LANCZOS).save('texture_sticking.png')