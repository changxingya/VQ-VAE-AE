import torch
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


# with torch.no_grad():
#     for data in train_ped2:
#         out = model(data.cuda())
#         total_tensor.append(out.cpu().detach())
#
# total_tensor = torch.cat(total_tensor, 0)
#
# np.save('p3d_data.npy', total_tensor.numpy())
#
# print(total_tensor.shape)

fig=plt.figure()
ax = Axes3D(fig)
total_np = np.load('p3d_data.npy')
tensor_feature = torch.from_numpy(total_np)
pca3 = PCA(n_components=3)  # 降到3d
fea = pca3.fit_transform(tensor_feature)
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(fea)

ax.scatter(fea[:, 0], fea[:, 1], fea[:, 2], s=10, c=y_pred, alpha=0.5)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
plt.savefig("1.png")










