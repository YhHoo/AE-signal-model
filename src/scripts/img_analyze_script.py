import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
import numpy as np
from src.utils.helpers import direct_to_dir


img_bank = []
for dist in range(11):
    filename = direct_to_dir(where='result') + 'xcor_cwt_DistDiff[{}m]_sample[22]'.format(dist) + '.png'
    xcor_img = mpimg.imread(filename)
    img_bank.append(xcor_img[350:624, 53:928, :])

filename = direct_to_dir(where='result') + 'xcor_cwt_DistDiff[0m]_sample[22]' + '.png'
time_img = mpimg.imread(filename)
time_img = time_img[52:323, 40:905, :]


fig = plt.figure(figsize=(15, 7))
fig.subplots_adjust(left=0.02, bottom=0, right=1, top=0.98, wspace=0.1, hspace=0.2)
for i in range(11):
    ax = fig.add_subplot(4, 3, i+1)
    ax.imshow(img_bank[i])
    ax.set_title('Dist_diff[{}m]'.format(i), fontsize=7)

ax = fig.add_subplot(4, 3, 12)
ax.imshow(time_img)
ax.set_title('Time Series of 0m', fontsize=7)

# grid_0 = AxesGrid(fig, 141,
#                   nrows_ncols=(5, 2),
#                   axes_pad=0.1,
#                   share_all=True,
#                   label_mode="L")
#
# for val, ax in zip(img_bank, grid_0):
#     # this configure titles for each heat map
#     # ax.set_title(each_subplot_title[0], position=(-0.15, 0.388), fontsize=7, rotation='vertical')
#     # this configure the dimension of the heat map in the fig object
#     im = ax.imshow(val)  # (left, right, bottom, top)
#
#
plt.show()

