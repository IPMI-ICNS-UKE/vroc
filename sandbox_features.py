import numpy as np
import SimpleITK as sitk

from vroc.feature_extractor import OrientedHistogramFeatureExtrator

# fig, ax = plt.subplots(7, 10)
ohs = []
forces = []
hover_data = []
downscaling = 4
histo_plots = []
clim_max = 0

# NLST
data = [
    {
        "moving_image": f"/datalake/learn2reg/NLST_fixed/imagesTr/NLST_{i:04d}_0000.nii.gz",
        "fixed_image": f"/datalake/learn2reg/NLST_fixed/imagesTr/NLST_{i:04d}_0001.nii.gz",
        "moving_mask": f"/datalake/learn2reg/NLST_fixed/masksTr/NLST_{i:04d}_0000.nii.gz",
        "fixed_mask": f"/datalake/learn2reg/NLST_fixed/masksTr/NLST_{i:04d}_0001.nii.gz",
        "id": f"NLST_{i:04d}",
        "color": "indigo",
    }
    for i in range(1, 101)
]

# DIRLAB
for i_moving_phase in [0, 1, 2, 3, 4, 6, 7, 8, 9]:
    data.extend(
        {
            "moving_image": f"/datalake/dirlab2022_v2/data/Case{i:02d}Pack/Images/phase_{i_moving_phase}.mha",
            "fixed_image": f"/datalake/dirlab2022_v2/data/Case{i:02d}Pack/Images/phase_5.mha",
            "moving_mask": f"/datalake/dirlab2022_v2/data/Case{i:02d}Pack/segmentation/mask_{i_moving_phase}.mha",
            "fixed_mask": f"/datalake/dirlab2022_v2/data/Case{i:02d}Pack/segmentation/mask_5.mha",
            "id": f"DIRLAB_{i:02d}/{i_moving_phase} -> 5",
            "color": "orangered",
        }
        for i in range(1, 11)
    )
feature_extractor = OrientedHistogramFeatureExtrator(device="cuda:1")
for d in data:
    print(d["id"])

    moving_image = sitk.ReadImage(d["moving_image"])
    fixed_image = sitk.ReadImage(d["fixed_image"])
    moving_mask = sitk.ReadImage(d["moving_mask"])
    fixed_mask = sitk.ReadImage(d["fixed_mask"])

    image_spacing = fixed_image.GetSpacing()[::-1]

    moving_image = sitk.GetArrayFromImage(moving_image)
    fixed_image = sitk.GetArrayFromImage(fixed_image)
    moving_mask = sitk.GetArrayFromImage(moving_mask)
    fixed_mask = sitk.GetArrayFromImage(fixed_mask)

    moving_image = np.swapaxes(moving_image, 0, 2)
    fixed_image = np.swapaxes(fixed_image, 0, 2)
    moving_mask = np.swapaxes(moving_mask, 0, 2)
    fixed_mask = np.swapaxes(fixed_mask, 0, 2)

    print("lets go")
    oh = feature_extractor.extract(
        fixed_image=fixed_image,
        moving_image=moving_image,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        image_spacing=image_spacing,
    )

    ohs.append(oh)
    hover_data.append(d)

ohs = np.array(ohs)
import pandas as pd
import umap.plot

from vroc.plot import plot_embedding

hover_data = pd.DataFrame.from_records(hover_data)

mapper = umap.UMAP(n_neighbors=2, min_dist=0.0, metric="euclidean", densmap=True)
mapper.fit(ohs.reshape(len(ohs), -1))
plot_embedding(
    mapper.embedding_,
    tooltip_images=ohs,
    patients=hover_data["id"],
    colors=hover_data["color"],
)
# for direction in ('x', 'y', 'z'):
#     plot = umap.plot.interactive(mapper, hover_data=hover_data, values=hover_data[f'moving -> fixed / median {direction}'])
#     show(plot)
#     demons = OrientedHistogram._cartesian_to_spherical(demons)
#
#     CLIM = (-5.0, 5.0)
#     i = (i - 1) % 10
#     histo_plot = ax[0, i - 1].imshow(oh, cmap='inferno')
#     ax[1, i - 1].imshow(moving_image[:, 100 // downscaling, :], clim=(-1000, 200))
#     ax[2, i - 1].imshow(fixed_image[:, 100 // downscaling, :], clim=(-1000, 200))
#     ax[3, i - 1].imshow(fixed_image[:, 100 // downscaling, :] - moving_image[:, 100 // downscaling, :], cmap='seismic', clim=(-1000, 1000))
#     ax[4, i - 1].imshow(demons[0, :, 100 // downscaling, :], cmap='seismic', clim=CLIM)
#     ax[5, i - 1].imshow(demons[1, :, 100 // downscaling, :], cmap='seismic', clim=CLIM)
#     ax[6, i - 1].imshow(demons[2, :, 100 // downscaling, :], cmap='seismic', clim=CLIM)
#     histo_plots.append(histo_plot)
#
#     clim_max = max(clim_max, np.percentile(oh, 99))
#
#     if i == 9:
#         break
#
#
# for histo_plot in histo_plots:
#     histo_plot.set_clim(0, clim_max)
