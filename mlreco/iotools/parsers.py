from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from larcv import larcv
from mlreco.utils.ppn import get_ppn_info
from mlreco.utils.dbscan import dbscan_types
from mlreco.utils.groups import filter_duplicate_voxels, filter_nonimg_voxels

def parse_sparse2d_meta(data):
    event_tensor2d = data[0]
    projection_id = 0  # default
    if isinstance(event_tensor2d, tuple):
        projection_id = event_tensor2d[1]
        event_tensor2d = event_tensor2d[0]

    tensor2d = event_tensor2d.sparse_tensor_2d(projection_id)
    meta = tensor2d.meta()
    # return np.array([[
    #     meta.min_x(),
    #     meta.min_y(),
    #     meta.max_x(),
    #     meta.max_y(),
    #     meta.pixel_width(),
    #     meta.pixel_height()
    # ]])
    return [
        meta.min_x(),
        meta.min_y(),
        meta.max_x(),
        meta.max_y(),
        meta.pixel_width(),
        meta.pixel_height()
    ]


def parse_sparse2d_scn(data):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object
    Returns the data in format to pass to SCN
    Args:
        array of larcv::EventSparseTensor2D
        optionally, array of (larcv::EventSparseTensor2D, int) for projection id
    Return:
        voxels - numpy array(int32) with shape (N,2) - coordinates
        data   - numpy array(float32) with shape (N,C) - pixel values/channels
    """
    meta = None
    output = []
    np_voxels = None
    for event_tensor2d in data:
        projection_id = 0  # default
        if isinstance(event_tensor2d, tuple):
            projection_id = event_tensor2d[1]
            event_tensor2d = event_tensor2d[0]

        tensor2d = event_tensor2d.sparse_tensor_2d(projection_id)
        num_point = tensor2d.as_vector().size()

        if meta is None:

            meta = tensor2d.meta()
            np_voxels = np.empty(shape=(num_point, 2), dtype=np.int32)
            larcv.fill_2d_voxels(tensor2d, np_voxels)

        # else:
        #     assert meta == tensor2d.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_2d_pcloud(tensor2d, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_sparse3d_scn(data):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object
    Returns the data in format to pass to SCN
    Args:
        array of larcv::EventSparseTensor3D
    Return:
        voxels - numpy array(int32) with shape (N,3) - coordinates
        data   - numpy array(float32) with shape (N,C) - pixel values/channels
    """
    meta = None
    output = []
    np_voxels = None
    for event_tensor3d in data:
        num_point = event_tensor3d.as_vector().size()
        if meta is None:
            meta = event_tensor3d.meta()
            np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
            larcv.fill_3d_voxels(event_tensor3d, np_voxels)
        else:
            assert meta == event_tensor3d.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_3d_pcloud(event_tensor3d, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_semantics(data):
    from larcv import larcv
    event_cluster3d = data[0]
    event_particle  = data[1]
    event_tensor3d = larcv.generate_semantics(event_cluster3d,event_particle)
    data = [event_tensor3d]
    res = parse_sparse3d_scn(data)
    return res


def parse_sparse3d(data):
    """
    A function to retrieve sparse tensor from larcv::EventSparseTensor3D object
    Args:
        array of larcv::EventSparseTensor3D (one per channel)
    Return:
        a numpy array with the shape (N,3+C) where 3+C represents
        (x,y,z) coordinate and C stored pixel values (channels).
    """
    meta = None
    output = []
    for event_tensor3d in data:
        num_point = event_tensor3d.as_vector().size()
        if meta is None:
            meta = event_tensor3d.meta()
            np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
            larcv.fill_3d_voxels(event_tensor3d, np_voxels)
            output.append(np_voxels)
        else:
            assert meta == event_tensor3d.meta()
        np_values = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_3d_pcloud(event_tensor3d, np_values)
        output.append(np_values)
    return np.concatenate(output, axis=-1)


def parse_tensor3d(data):
    """
    A function to retrieve larcv::EventSparseTensor3D as a dense numpy array
    Args:
        array of larcv::EventSparseTensor3D
    Return:
        a numpy array of a dense 3d tensor object, last dimension = channels
    """
    np_data = []
    meta = None
    for event_tensor3d in data:
        if meta is None:
            meta = event_tensor3d.meta()
        else:
            assert meta == event_tensor3d.meta()
        np_data.append(np.array(larcv.as_ndarray(event_tensor3d)))
    return np.stack(np_data, axis=-1)


def parse_particle_asis(data):
    """
    A function to copy construct & return an array of larcv::Particle
    Args:
        length 1 array of larcv::EventParticle
    Return:
        a python list of larcv::Particle object
    """
    particles = data[0]
    clusters  = data[1]
    assert particles.as_vector().size() in [clusters.as_vector().size(),clusters.as_vector().size()-1]

    meta = clusters.meta()

    particles = [larcv.Particle(p) for p in data[0].as_vector()]
    funcs = ["first_step","last_step","position","end_position"]
    for p in particles:
        for f in funcs:
            pos = getattr(p,f)()
            x = (pos.x() - meta.min_x()) / meta.size_voxel_x()
            y = (pos.y() - meta.min_y()) / meta.size_voxel_y()
            z = (pos.z() - meta.min_z()) / meta.size_voxel_z()
            getattr(p,f)(x,y,z,pos.t())
    return particles


def parse_particle_points(data):
    """
    A function to retrieve particles ground truth points tensor, includes
    spatial coordinates and point type.
    Args:
        length 2 array of larcv::EventSparseTensor3D and larcv::EventParticle
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N, 2) where 2 represents the particle data
        index and the class of the ground truth point respectively.
    """
    particles_v = data[1].as_vector()
    part_info = get_ppn_info(particles_v, data[0].meta())
    if part_info.shape[0] > 0:
        #return part_info[:, :3], part_info[:, 3][:, None]
        return part_info[:, :3], np.column_stack([part_info[:, -6],part_info[:, -1]])
    else:
        #return np.empty(shape=(0, 3), dtype=np.int32), np.empty(shape=(0, 1), dtype=np.float32)
        return np.empty(shape=(0, 3), dtype=np.int32), np.empty(shape=(0, 2), dtype=np.float32)


def parse_particle_graph(data):
    """
    A function to parse larcv::EventParticle to construct edges between particles (i.e. clusters)
    Args:
        length 1 array of larcv::EventParticle
    Return:
        a numpy array of directed edges where each edge is (parent,child) cluster index ID.
    """
    particles = data[0]

    # For convention, construct particle id => cluster id mapping
    particle_to_cluster = np.zeros(shape=[particles.as_vector().size()],dtype=np.int32)

    # Fill group mapping
    for cluster_id in range(particles.as_vector().size()):
        p = particles.as_vector()[cluster_id]
        particle_id = p.id()
        particle_to_cluster[particle_id] = cluster_id

    # Fill edges (directed, [parent,child] pair)
    edges = np.empty((0,2), dtype = np.int32)
    for cluster_id in range(particles.as_vector().size()):
        p = particles.as_vector()[cluster_id]
        for child in p.children_id():
            if cluster_id != particle_to_cluster[child]:
                edges = np.vstack((edges, [cluster_id,particle_to_cluster[child]]))

    return edges


def parse_dbscan(data):
    """
    A function to create dbscan tensor
    Args:
        length 1 array of larcv::EventSparseTensor3D
    Return:
        voxels - numpy array(int32) with shape (N,3) - coordinates
        data   - numpy array(float32) with shape (N,1) - dbscan cluster. -1 if not assigned
    """
    np_voxels, np_types = parse_sparse3d_scn(data)
    # now run dbscan on data
    clusts = dbscan_types(np_voxels, np_types)
    # start with no clusters assigned.
    np_types.fill(-1)
    for i, c in enumerate(clusts):
        np_types[c] = i
    return np_voxels, np_types


def parse_cluster2d(data):
    """
    A function to retrieve clusters tensor
    Args:
        length 1 array of larcv::EventClusterVoxel3D
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,2) where 2 is pixel value and cluster id, respectively
    """
    cluster_event = data[0].as_vector().front()
    meta = cluster_event.meta()
    num_clusters = cluster_event.size()
    clusters_voxels, clusters_features = [], []
    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.float32)
            larcv.as_flat_arrays(cluster,meta,x, y, value)
            cluster_id = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=i, dtype=np.float32)
            clusters_voxels.append(np.stack([x, y], axis=1))
            clusters_features.append(np.column_stack([value, cluster_id]))
    np_voxels   = np.concatenate(clusters_voxels, axis=0)
    np_features = np.concatenate(clusters_features, axis=0)

    return np_voxels, np_features


def parse_cluster3d(data):
    """
    a function to retrieve clusters tensor
    args:
        length 1 array of larcv::EventClusterVoxel3D
    return:
        a numpy array with the shape (n,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (n,2) where 2 is voxel value and cluster id, respectively
    """
    cluster_event = data[0]
    meta = cluster_event.meta()
    num_clusters = cluster_event.as_vector().size()
    clusters_voxels, clusters_features = [], []
    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            z = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.float32)
            larcv.as_flat_arrays(cluster,meta,x, y, z, value)
            cluster_id = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=i, dtype=np.float32)
            clusters_voxels.append(np.stack([x, y, z], axis=1))
            clusters_features.append(np.column_stack([value,cluster_id]))
    np_voxels   = np.concatenate(clusters_voxels, axis=0)
    np_features = np.concatenate(clusters_features, axis=0)
    return np_voxels, np_features


def parse_cluster3d_full(data):
    """
    a function to retrieve clusters tensor
    args:
        length 2 array of larcv::EventClusterVoxel3D and larcv::EventParticle
    return:
        a numpy array with the shape (n,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (n,4) where 4 is voxel value,
        cluster id, group id and semantic type, respectively
    """
    
    parts = parse_particle_asis([data[1],data[0]])
    
    cluster_event = data[0]
    particles_v = data[1].as_vector()
    meta = cluster_event.meta()
    num_clusters = cluster_event.as_vector().size()
    clusters_voxels, clusters_features = [], []
    ### SHOULD BE FIXED AT SUPERA LEVEL LATER ###
    group_ids = np.array([particles_v[i].group_id() for i in range(particles_v.size())])
    nuclear_group = num_clusters + 1
    for i, gid in enumerate(np.unique(group_ids)):
        # If the group's parent is not EM or LE, proceed
        if particles_v[int(gid)].shape() != 0 and particles_v[int(gid)].shape() != 4:
            continue
        # If the group's parent is nuclear activity, Delta or Michel, make it non primary
        process = particles_v[int(gid)].creation_process()
        parent_pdg_code = abs(particles_v[int(gid)].parent_pdg_code())
        if 'Inelastic' in process or 'Capture' in process or parent_pdg_code == 13:
            idxs = np.where(group_ids == gid)[0]
            group_ids[idxs] = nuclear_group
            nuclear_group += 1
            continue
        # Make the first (in time) non-empty cluster the true group ID
        idxs = np.where(group_ids == gid)[0]
        clust_times = np.array([particles_v[int(j)].first_step().t() for j in idxs])
        nonempty = np.where([cluster_event.as_vector()[int(j)].as_vector().size() for j in idxs])[0]
        if len(nonempty):
            first_id = np.argmin(clust_times[nonempty])
            group_ids[idxs] = idxs[nonempty][first_id]
        # If the primary cluster is empty, make its direct children primaries
        #clust_size = cluster_event.as_vector()[int(gid)].as_vector().size()
        #if not clust_size:
        #    idxs = np.where(group_ids == gid)[0]
        #    for j in idxs:
        #        if j != gid and cluster_event.as_vector()[int(j)].size():
        #            particle = particles_v[int(j)]
        #            group_id = particle.id()
        #            while particle.parent_id() != gid:
        #                parent_id = particle.parent_id()
        #                particle = particles_v[parent_id]
        #                if not cluster_event.as_vector()[parent_id].size():
        #                    continue
        #                group_id = particle.id()
        #            group_ids[j] = group_id
    #############################################
    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            z = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.float32)
            larcv.as_flat_arrays(cluster,meta,x, y, z, value)
            assert i == particles_v[i].id()
            cluster_id = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=particles_v[i].id(), dtype=np.float32)
            group_id = np.full(shape=(cluster.as_vector().size()),
                               #fill_value=particles_v[i].group_id(), dtype=np.float32)
                               fill_value=group_ids[i], dtype=np.float32)
            sem_type = np.full(shape=(cluster.as_vector().size()),
                               fill_value=particles_v[i].shape(), dtype=np.float32)
            
            #ADDED PART
            x_init,y_init,z_init = parts[i].first_step().x(),parts[i].first_step().y(),parts[i].first_step().z()
            speed = (parts[i].px()**2+parts[i].py()**2+parts[i].pz()**2)**0.5
            speed_x,speed_y,speed_z = parts[i].px()/speed,parts[i].py()/speed,parts[i].pz()/speed
  
            x_feat = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=x_init, dtype=np.float32)
            y_feat = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=y_init, dtype=np.float32)
            z_feat = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=z_init, dtype=np.float32)
            xs_feat = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=speed_x, dtype=np.float32)
            ys_feat = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=speed_y, dtype=np.float32)
            zs_feat = np.full(shape=(cluster.as_vector().size()),
                                 fill_value=speed_z, dtype=np.float32)
            #END OF ADDED PART
            
            clusters_voxels.append(np.stack([x, y, z], axis=1))
            #clusters_features.append(np.column_stack([value,cluster_id,group_id,sem_type]))
            clusters_features.append(np.column_stack([x_feat,y_feat,z_feat,xs_feat,ys_feat,zs_feat,value,cluster_id,group_id,sem_type]))
         
    np_voxels   = np.concatenate(clusters_voxels, axis=0)
    np_features = np.concatenate(clusters_features, axis=0)
    return np_voxels, np_features


def parse_cluster3d_clean(data):
    """
    A function to retrieve clusters tensor.  Do the following cleaning:
    1) lexicographically sort group data (images are lexicographically sorted)
    2) remove voxels from group data that are not in image
    3) choose only one group per voxel (by lexicographic order)
    Args:
        length 2 array of larcv::EventClusterVoxel3D and larcv::EventSparseTensor3D
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,1) where 2 represents (value, cluster_id)
    """
    grp_voxels, grp_data = parse_cluster3d([data[0]])
    img_voxels, img_data = parse_sparse3d_scn([data[1]])

    # step 1: lexicographically sort group data
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm,:]
    grp_data = grp_data[perm]

    perm = np.lexsort(img_voxels.T)
    img_voxels = img_voxels[perm,:]
    img_data = img_data[perm]

    # step 2: remove duplicates
    sel1 = filter_duplicate_voxels(grp_voxels, usebatch=False)
    inds1 = np.where(sel1)[0]
    grp_voxels = grp_voxels[inds1,:]
    grp_data = grp_data[inds1]

    # step 3: remove voxels not in image
    sel2 = filter_nonimg_voxels(grp_voxels, img_voxels, usebatch=False)
    inds2 = np.where(sel2)[0]
    grp_voxels = grp_voxels[inds2,:]
    grp_data = grp_data[inds2]

    return grp_voxels, grp_data

def parse_cluster3d_clean_full(data):
    """
    A function to retrieve clusters tensor.  Do the following cleaning:
    1) lexicographically sort group data (images are lexicographically sorted)
    2) remove voxels from group data that are not in image
    3) choose only one group per voxel (by lexicographic order)
    Args:
        length 3 array of larcv::EventClusterVoxel3D, larcv::EventParticle and larcv::EventSparseTensor3D
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,4) where 4 represens (value, cluster_id, group_id, sem_type)
    """
    grp_voxels, grp_data = parse_cluster3d_full(data)
    img_voxels, img_data = parse_sparse3d_scn([data[2]])

    # step 1: lexicographically sort group data
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm,:]
    grp_data = grp_data[perm]

    perm = np.lexsort(img_voxels.T)
    img_voxels = img_voxels[perm,:]
    img_data = img_data[perm]

    # step 2: remove duplicates
    sel1 = filter_duplicate_voxels(grp_voxels, usebatch=False)
    inds1 = np.where(sel1)[0]
    grp_voxels = grp_voxels[inds1,:]
    grp_data = grp_data[inds1]

    # step 3: remove voxels not in image
    sel2 = filter_nonimg_voxels(grp_voxels, img_voxels, usebatch=False)
    inds2 = np.where(sel2)[0]
    grp_voxels = grp_voxels[inds2,:]
    grp_data = grp_data[inds2]

    # step 4: override semantic labels with those from sparse3d (TODO)
    grp_data[:,-1] = img_data[:,-1]

    return grp_voxels, grp_data


def parse_sparse3d_clean(data):
    """
    A function to retrieve clusters tensor.  Do the following cleaning:
    1) lexicographically sort coordinates
    2) choose only one group per voxel (by lexicographic order)
    3) get labels from the image labels for each voxel in addition to groups
    Args:
        length 3 array of larcv::EventSparseTensor3D
        Typically [sparse3d_mcst_reco, sparse3d_mcst_reco_group, sparse3d_fivetypes_reco]
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,3) where 3 is energy + cluster id + label
    """
    img_voxels, img_data = parse_sparse3d_scn([data[0]])
    perm = np.lexsort(img_voxels.T)
    img_voxels = img_voxels[perm]
    #img_data = img_data[perm]
    img_voxels, unique_indices = np.unique(img_voxels, axis=0, return_index=True)
    #img_data = img_data[unique_indices]

    grp_voxels, grp_data = parse_sparse3d_scn([data[1]])
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm]
    grp_data = grp_data[perm]
    grp_voxels, unique_indices = np.unique(grp_voxels, axis=0, return_index=True)
    grp_data = grp_data[unique_indices]

    label_voxels, label_data = parse_sparse3d_scn([data[2]])
    perm = np.lexsort(label_voxels.T)
    label_voxels = label_voxels[perm]
    label_data = label_data[perm]
    label_voxels, unique_indices = np.unique(label_voxels, axis=0, return_index=True)
    label_data = label_data[unique_indices]

    sel2 = filter_nonimg_voxels(grp_voxels, label_voxels[(label_data<5).reshape((-1,)),:], usebatch=False)
    inds2 = np.where(sel2)[0]
    grp_voxels = grp_voxels[inds2]
    grp_data = grp_data[inds2]

    sel2 = filter_nonimg_voxels(img_voxels, label_voxels[(label_data<5).reshape((-1,)),:], usebatch=False)
    inds2 = np.where(sel2)[0]
    img_voxels = img_voxels[inds2]
    img_data = img_data[inds2]
    return grp_voxels, np.concatenate([img_data, grp_data, label_data[label_data<5][:, None]], axis=1)


def parse_cluster3d_scales(data):
    """
    Retrieves clusters tensors at different spatial sizes.
    Parameters
    ----------
    data: list
        length 2 array of larcv::EventClusterVoxel3D and larcv::EventSparseTensor3D
    Returns
    -------
    list of tuples
    """
    grp_voxels, grp_data = parse_cluster3d_clean(data)
    spatial_size = data[0].meta().num_voxel_x()
    max_depth = int(np.floor(np.log2(spatial_size))-1)
    scales = []
    for d in range(max_depth):
        scale_voxels = np.floor(grp_voxels/2**d)#.astype(int)
        scale_voxels, unique_indices = np.unique(scale_voxels, axis=0, return_index=True)
        scale_data = grp_data[unique_indices]
        scales.append((scale_voxels, scale_data))
    return scales


def parse_sparse3d_scn_scales(data):
    """
    Retrieves sparse tensors at different spatial sizes.
    Parameters
    ----------
    data: list
        length 1 array of larcv::EventSparseTensor3D
    Returns
    -------
    list of tuples
    """
    grp_voxels, grp_data = parse_sparse3d_scn(data)
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm]
    grp_data = grp_data[perm]

    spatial_size = data[0].meta().num_voxel_x()
    max_depth = int(np.floor(np.log2(spatial_size))-1)
    scales = []
    for d in range(max_depth):
        scale_voxels = np.floor(grp_voxels/2**d)#.astype(int)
        scale_voxels, unique_indices = np.unique(scale_voxels, axis=0, return_index=True)
        scale_data = grp_data[unique_indices]
        # perm = np.lexsort(scale_voxels.T)
        # scale_voxels = scale_voxels[perm]
        # scale_data = scale_data[perm]
        scales.append((scale_voxels, scale_data))
    return scales
