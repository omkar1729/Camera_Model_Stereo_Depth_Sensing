import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.scatter3D(objp[:,0], objp[:,1],objp[:,2])
#ax.scatter3D(triangulate[0]/triangulate[3],triangulate[1]/triangulate[3],triangulate[2]/triangulate[3])
#ax.scatter3D(origin2[0],origin2[1],origin2[2],'yellow')
ax.scatter3D(0.0,0.0,0.0,c='red')

plt.show()