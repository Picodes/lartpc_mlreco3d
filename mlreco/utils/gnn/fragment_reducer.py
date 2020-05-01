import numpy as np

def build_image_filled(voxels, clust, values=None, size=32):
    # Build an empty image
    image = []
    # Find the max voxel range of voxels, hence the grid step (on a ]0,1] scale)
    mins      = np.min(voxels, axis=0)
    ranges    = np.ptp(voxels, axis=0)
    max_range = np.max(ranges)+1
    grid_step = 1./max_range
    # Rescale and center voxels
    grid_mins = (voxels-mins-ranges/2-0.5)*grid_step + 0.5
    # For each voxel in the input, divide up the energy into the output pixels it overlaps with
    new_step = 1./size
    for e, v in enumerate(grid_mins):
        # Find the new voxels it covers in x, y and z
        imins  = (v/new_step).astype(int)
        imaxs  = ((v-1e-9+grid_step)/new_step).astype(int)+1
        irange = imaxs-imins
        # Find the fractions that goes into each voxel for each axis
        for i in range(imins[0], imaxs[0]):
            for j in range(imins[1], imaxs[1]):
                for k in range(imins[2], imaxs[2]):
                    idx = np.array([i,j,k])
                    fracs = (np.min([v+grid_step, (idx+1)*new_step], axis=0)-np.max([v, idx*new_step], axis=0))/grid_step
                    image.append([i,j,k,len(voxels),np.prod(fracs)*values[e]])
    # Return image, scaling factor and offset
    return np.array(image), float(size)/max_range, mins+ranges/2

def build_image_no_filling(voxels, clust, values=None, size=32):
    # Find the max voxel range of voxels, hence the grid step (on a ]0,1] scale)
    mins      = np.min(voxels, axis=0)
    ranges    = np.ptp(voxels, axis=0)
    max_range = np.max(ranges)+1
    grid_step = 1./max_range
    # Rescale and center voxels
    grid_mins = (voxels-mins-ranges/2-0.5)*grid_step + 0.5
    # For each voxel in the input, divide up the energy into the output pixels it overlaps with
    new_step = 1./size
    image = (grid_mins*32).astype(int)
    image = np.concatenate((np.array(image),np.ones((len(image[:,0]),2))),1)
    return image, float(size)/max_range, mins+ranges/2

def build_image_wth_tre(voxels, clust, values=None, size=32):
    # Find the max voxel range of voxels, hence the grid step (on a ]0,1] scale)
    mins      = np.min(voxels, axis=0)
    ranges    = np.ptp(voxels, axis=0)
    max_range = np.max(ranges)+1
    #if no scaling is needed
    if (max_range < 32):
        image = (voxels-mins-ranges/2-0.5)*1/32 + 0.5
        image = (image*32).astype(int)
        image = np.concatenate((np.array(image),np.ones((len(image[:,0]),2))),1)
        return image, 1, mins+ranges/2
    else:
        grid_step = 1./max_range
        # Rescale and center voxels in (0,1)
        image = (voxels-mins-ranges/2-0.5)*grid_step + 0.5
        # For each voxel in the input, divide up the energy into the output pixels it overlaps with
        new_step = 1./size
        image = (image*32).astype(int)
        image = np.concatenate((np.array(image),np.ones((len(image[:,0]),2))),1)
        return image, float(size)/max_range, mins+ranges/2