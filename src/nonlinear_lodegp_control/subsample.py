import torch

def subsample_kmeans(X, Y, m, num_iters=10):
    """
    Subsample (X,Y) down to m points using K-means clustering + medoids.
    X: Tensor of shape (N, 2). Y: Tensor of shape (N,) or (N,1). 
    Returns: (X_sub, Y_sub, idxs) where X_sub has shape (m,2).
    """
    N, d = X.shape
    if m >= N:
        idxs = torch.arange(N)
        return X.clone(), Y.clone(), idxs

    # Randomly initialize cluster centers by picking m distinct points
    perm = torch.randperm(N)
    centers = X[perm[:m]].clone()

    for _ in range(num_iters):
        # Assign points to nearest center (squared Euclidean distance)
        # dist2 has shape (N, m)
        diff = X.unsqueeze(1) - centers.unsqueeze(0)  # (N, m, 2)
        dist2 = torch.sum(diff*diff, dim=2)          # (N, m)
        labels = torch.argmin(dist2, dim=1)          # which cluster each point belongs to

        # Recompute centers as the mean of assigned points
        new_centers = torch.zeros_like(centers)
        for k in range(m):
            members = (labels == k).nonzero(as_tuple=False).squeeze(1)
            if len(members) > 0:
                new_centers[k] = X[members].mean(dim=0)
            else:
                # If a cluster gets no points, keep its center unchanged
                new_centers[k] = centers[k]
        # Check for convergence (optional)
        if torch.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    # Now select a medoid (nearest actual point) from each cluster
    selected_idxs = []
    for k in range(m):
        members = (labels == k).nonzero(as_tuple=False).squeeze(1)
        if len(members) > 0:
            # Compute distances of cluster members to the cluster center
            dists = torch.sum((X[members] - centers[k])**2, dim=1)
            medoid = members[torch.argmin(dists)]
            selected_idxs.append(medoid.item())
    # Remove duplicates (if any) while preserving order
    # (Some clusters might have merged or been empty, so final count may be < m.)
    selected_idxs = list(dict.fromkeys(selected_idxs))
    # If we still have fewer than m points (due to empty clusters), add random points
    if len(selected_idxs) < m:
        remaining = set(range(N)) - set(selected_idxs)
        add_count = m - len(selected_idxs)
        if add_count > 0 and remaining:
            extra = torch.tensor(list(remaining))[torch.randperm(len(remaining))[:add_count]]
            selected_idxs += extra.tolist()
    selected_idxs = selected_idxs[:m]
    idxs_tensor = torch.tensor(selected_idxs, dtype=torch.long)

    X_sub = X[idxs_tensor]
    Y_sub = Y[idxs_tensor]
    return X_sub, Y_sub, idxs_tensor


def subsample_farthest_point(X, Y, m):
    """
    Subsample (X,Y) down to m points using greedy farthest-point sampling (FPS).
    X: Tensor of shape (N, 2). Y: Tensor of shape (N,) or (N,1).
    Returns: (X_sub, Y_sub, idxs) where X_sub has shape (m,2).
    """
    N, d = X.shape
    if m >= N:
        idxs = torch.arange(N)
        return X.clone(), Y.clone(), idxs

    # Start with a random point
    first = torch.randint(0, N, (1,), dtype=torch.long).item()
    selected = [first]

    # Compute initial distances (squared) from the first point
    dist = torch.sum((X - X[first])**2, dim=1)
    dist[first] = 0.0  # distance to itself

    # Greedily add farthest point
    for _ in range(m - 1):
        idx_next = torch.argmax(dist).item()
        selected.append(idx_next)
        # Update distances: for each point, keep the min distance to any selected so far
        new_dist = torch.sum((X - X[idx_next])**2, dim=1)
        dist = torch.minimum(dist, new_dist)
        dist[idx_next] = 0.0  # avoid picking it again

    idxs = torch.tensor(selected, dtype=torch.long)
    X_sub = X[idxs]
    Y_sub = Y[idxs]
    return X_sub, Y_sub, idxs
