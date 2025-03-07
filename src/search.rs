use std::cmp::Ordering;
use std::collections::BinaryHeap;
use crate::vectors::*;

#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
pub(crate) struct SearchPair {
    distance_sq: f32,
    index: usize,
}
impl From<(usize, f32)> for SearchPair {
    fn from(value: (usize, f32)) -> Self {
        Self {index: value.0, distance_sq: value.1}
    }
}
impl Eq for SearchPair {}
impl Ord for SearchPair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap()
    }
}
#[derive(Debug, PartialEq, Copy, Clone)]
pub(crate) struct Cluster {
    start_index: usize,
    length: usize,
}

pub(crate) fn apply_clusters(vectors: &mut VectorsList, clusters: &[Vec<usize>]) -> (Box<[usize]>, Box<[Cluster]>) {
    let clusters_flat = clusters
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .into_boxed_slice();
    debug_assert_eq!(clusters_flat.len(), vectors.count);

    // todo: there might be a better algorithm for applying a permutation to an array 
    let mut actual_indexes = (0..vectors.count).collect::<Vec<_>>().into_boxed_slice();
    for (index, &current_place_should_be) in clusters_flat.iter().enumerate() {
        let current_place_was_originally_at = actual_indexes[index];
        if current_place_was_originally_at == current_place_should_be {
            continue;
        }
        let mut should_be_is_actually_at = actual_indexes[current_place_should_be];
        while should_be_is_actually_at < index {
            should_be_is_actually_at = actual_indexes[should_be_is_actually_at]
        }
        actual_indexes.swap(index, should_be_is_actually_at);
        vectors.swap(index, should_be_is_actually_at);
    }

    let mut cluster_refs = Vec::with_capacity(clusters.len());
    let mut start = 0;
    for cluster in clusters {
        let cluster_len = cluster.len();
        let end = start + cluster_len;
        cluster_refs.push(Cluster{ start_index: start, length: cluster_len });
        start = end;
    }
    (actual_indexes, cluster_refs.into_boxed_slice())
}

pub(crate) fn flat_search(vectors: &VectorsList, search: &[f32], topk: usize) -> BinaryHeap<SearchPair> {
    debug_assert_eq!(vectors.dim, search.len());
    let topk = topk.min(vectors.count);
    let mut heap = BinaryHeap::<SearchPair>::with_capacity(topk);
    if topk == 0 {
        return heap;
    }
    heap.push((0, f32::MAX).into());
    for (index, vector) in vectors.iter().enumerate() {
        let distance = calc_distance_sq(search, vector);
        let worst_found= heap.peek().unwrap();
        if distance < worst_found.distance_sq {
            if heap.len() >= topk {
                heap.pop();
            }
            heap.push((index, distance).into())
        }
    }
    heap
}

pub(crate) fn search_clustered_by_indices(vectors: &VectorsList, centroids: &VectorsList, clusters: &[Vec<usize>], search: &[f32], nprobe: usize, topk: usize) -> BinaryHeap<SearchPair> {
    debug_assert_eq!(vectors.dim, centroids.dim);
    debug_assert!(vectors.count >= centroids.count);
    debug_assert_eq!(centroids.count, clusters.len());
    debug_assert_eq!(vectors.dim, search.len());
    let nprobe = nprobe.min(centroids.count);
    let topk = topk.min(vectors.count);
    if (topk == 0) || (nprobe == 0) {
        return BinaryHeap::new();
    }
    let mut result = BinaryHeap::with_capacity(topk);
    let mut heaps = Vec::with_capacity(nprobe);
    let best_centroids = flat_search(centroids, search, nprobe);
    for SearchPair{index: centroid_index, ..} in best_centroids {
        let cluster = clusters[centroid_index].as_slice();
        let found = search_by_indices(vectors, cluster, search, topk);
        heaps.push(found);
    }
    merge_heaps(heaps, &mut result);
    result
}

fn search_by_indices(vectors: &VectorsList, indices: &[usize], search: &[f32], topk: usize) -> BinaryHeap<SearchPair> {
    debug_assert_eq!(vectors.dim, search.len());
    let topk = topk.min(indices.len());
    let mut heap = BinaryHeap::<SearchPair>::with_capacity(topk);
    if topk == 0 {
        return heap;
    }
    heap.push((0, f32::MAX).into());
    for &index in indices {
        let vector = vectors.get(index);
        let distance = calc_distance_sq(search, vector);
        let worst_found= heap.peek().unwrap();
        if distance < worst_found.distance_sq {
            if heap.len() >= topk {
                heap.pop();
            }
            heap.push((index, distance).into())
        }
    }
    heap
}

fn merge_heaps(from: Vec<BinaryHeap<SearchPair>>, to: &mut BinaryHeap<SearchPair>) {
    let size = to.capacity();
    for heap in from {
        for value in heap {
            let worst = to.peek();
            let should_insert = match worst {
                None => true,
                Some(worst) => worst.distance_sq > value.distance_sq,
            };
            if should_insert {
                if to.len() >= size {
                    to.pop();
                }
                to.push(value)
            }
        }
    }
}

pub(crate) fn restore_indices(search_results: &mut [SearchPair], orig_indices: &[usize]) {
    for pair in search_results.iter_mut() {
        pair.index = orig_indices[pair.index];
    }
}

pub(crate) fn search_clustered(vectors: &VectorsList, centroids: &VectorsList, clusters: &[Cluster], search: &[f32], nprobe: usize, topk: usize) -> BinaryHeap<SearchPair> {
    debug_assert_eq!(vectors.dim, centroids.dim);
    debug_assert!(vectors.count >= centroids.count);
    debug_assert_eq!(centroids.count, clusters.len());
    debug_assert_eq!(vectors.dim, search.len());
    let nprobe = nprobe.min(centroids.count);
    let topk = topk.min(vectors.count);
    if (topk == 0) || (nprobe == 0) {
        return BinaryHeap::new();
    }
    let mut result = BinaryHeap::with_capacity(topk);
    let mut heaps = Vec::with_capacity(nprobe);
    let best_centroids = flat_search(centroids, search, nprobe);
    for SearchPair{index: centroid_index, ..} in best_centroids {
        let found = search_cluster(vectors, clusters[centroid_index], search, topk);
        heaps.push(found);
    }
    merge_heaps(heaps, &mut result);
    result
}

fn search_cluster(vectors: &VectorsList, cluster: Cluster, search: &[f32], topk: usize) -> BinaryHeap<SearchPair> {
    debug_assert_eq!(vectors.dim, search.len());
    debug_assert!((cluster.start_index + cluster.length) <= vectors.count);
    let topk = topk.min(cluster.length);
    let mut heap = BinaryHeap::<SearchPair>::with_capacity(topk);
    if topk == 0 {
        return heap;
    }
    heap.push((0, f32::MAX).into());
    for (index_in_cluster, vector) in vectors.get_cluster(cluster.start_index, cluster.length).enumerate() {
        let distance = calc_distance_sq(search, vector);
        let worst_found= heap.peek().unwrap();
        if distance < worst_found.distance_sq {
            if heap.len() >= topk {
                heap.pop();
            }
            let overall_index = cluster.start_index + index_in_cluster;
            heap.push((overall_index, distance).into())
        }
    }
    heap
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_flat_search() {
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let search = [0.5, 0.5];
        let result = flat_search(&vectors, &search, 3).into_sorted_vec();
        let expected_result = [
            SearchPair::from((1, 2.5)),
            SearchPair::from((3, 4.5)),
            SearchPair::from((0, 6.5)),
        ];
        assert_eq!(expected_result, result.as_slice());
        let result = flat_search(&vectors, &search, 0).into_sorted_vec();
        assert_eq!(Vec::<SearchPair>::new(), result.as_slice());

        let result = flat_search(&vectors, &search, 4).into_sorted_vec();
        let expected_result = [
            SearchPair::from((1, 2.5)),
            SearchPair::from((3, 4.5)),
            SearchPair::from((0, 6.5)),
            SearchPair::from((2, 8.5)),
        ];
        assert_eq!(expected_result, result.as_slice());

        let result = flat_search(&vectors, &search, 5).into_sorted_vec();
        assert_eq!(expected_result, result.as_slice());
    }

    #[test]
    fn test_search_by_indices() {
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [ 0.0,  0.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
        ];
        let indices = [3, 0, 1, 4];
        let vectors = VectorsList::from_arrays(&vectors);
        let search = [0.5, 0.5];
        let result = search_by_indices(&vectors, &indices, &search, 3).into_sorted_vec();
        let expected_result = [
            SearchPair::from((1, 2.5)),
            SearchPair::from((4, 4.5)),
            SearchPair::from((0, 6.5)),
        ];
        assert_eq!(expected_result, result.as_slice());
        let result = search_by_indices(&vectors, &indices, &search, 0).into_sorted_vec();
        assert_eq!(Vec::<SearchPair>::new(), result.as_slice());
        let result = search_by_indices(&vectors, &[], &search, 3).into_sorted_vec();
        assert_eq!(Vec::<SearchPair>::new(), result.as_slice());

        let result = search_by_indices(&vectors, &indices, &search, 4).into_sorted_vec();
        let expected_result = [
            SearchPair::from((1, 2.5)),
            SearchPair::from((4, 4.5)),
            SearchPair::from((0, 6.5)),
            SearchPair::from((3, 8.5)),
        ];
        assert_eq!(expected_result, result.as_slice());

        let result = search_by_indices(&vectors, &indices, &search, 5).into_sorted_vec();
        assert_eq!(expected_result, result.as_slice());
    }

    #[test]
    fn test_search_cluster() {
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
            [ 0.0,  0.0],
            [ 1.0,  1.0],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let clusters = [
            Cluster{ start_index: 0, length: 4 },
            Cluster{ start_index: 4, length: 0 },
            Cluster{ start_index: 4, length: 2 },
        ];
        let search = [0.5, 0.5];
        let result = search_cluster(&vectors, clusters[0], &search, 3).into_sorted_vec();
        let expected_result = [
            SearchPair::from((1, 2.5)),
            SearchPair::from((3, 4.5)),
            SearchPair::from((0, 6.5)),
        ];
        assert_eq!(expected_result, result.as_slice());
        let result = search_cluster(&vectors, clusters[0], &search, 0).into_sorted_vec();
        assert_eq!(Vec::<SearchPair>::new(), result.as_slice());
        let result = search_cluster(&vectors, clusters[1], &search, 0).into_sorted_vec();
        assert_eq!(Vec::<SearchPair>::new(), result.as_slice());

        let result = search_cluster(&vectors, clusters[0], &search, 4).into_sorted_vec();
        let expected_result = [
            SearchPair::from((1, 2.5)),
            SearchPair::from((3, 4.5)),
            SearchPair::from((0, 6.5)),
            SearchPair::from((2, 8.5)),
        ];
        assert_eq!(expected_result, result.as_slice());

        let result = search_cluster(&vectors, clusters[0], &search, 5).into_sorted_vec();
        assert_eq!(expected_result, result.as_slice());

        let expected_result = [
            SearchPair::from((4, 0.5)),
            SearchPair::from((5, 0.5)),
        ];
        let result = search_cluster(&vectors, clusters[2], &search, 5).into_sorted_vec();
        assert_eq!(expected_result, result.as_slice());
    }

    #[test]
    fn test_merge_heaps() {
        let heaps = vec![
            BinaryHeap::from([
                SearchPair::from((8, 1.3)),
                SearchPair::from((3, 1.1)),
                SearchPair::from((6, 1.5)),
            ]),
            BinaryHeap::from([
                SearchPair::from((1, 1.6)),
                SearchPair::from((5, 0.5)),
                SearchPair::from((0, 0.6)),
            ]),
            BinaryHeap::from([
                SearchPair::from((2, 2.7)),
                SearchPair::from((7, 1.7)),
                SearchPair::from((4, 1.9)),
            ]),
        ];
        let mut to = BinaryHeap::with_capacity(3);
        merge_heaps(heaps, &mut to);
        let expected = [
            SearchPair::from((5, 0.5)),
            SearchPair::from((0, 0.6)),
            SearchPair::from((3, 1.1)),
        ];
        assert_eq!(expected, to.into_sorted_vec().as_slice());
    }

    #[test]
    fn test_apply_clusters() {
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [ 0.0,  0.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
        ];
        let mut vectors = VectorsList::from_arrays(&vectors);
        let clusters = vec![
            vec![1, 4, 3],
            vec![0, 2],
        ];
        let (order, clusters) = apply_clusters(&mut vectors, clusters.as_slice());
        let expected_ids = [1, 4, 3, 0, 2];
        assert_eq!(expected_ids, *order);
        let expected_vectors = [
            [ 2.0,  1.0],
            [ 2.0, -1.0],
            [-2.0, -1.0],
            [-2.0,  1.0],
            [ 0.0,  0.0],
        ];
        assert_eq!(expected_vectors, vectors.as_vec_of_slices().as_slice());
        let expected_clusters = [
            Cluster{ start_index: 0, length: 3 },
            Cluster{ start_index: 3, length: 2 },
        ];
        assert_eq!(expected_clusters, *clusters);
    }

    #[test]
    fn test_search() {
        let vectors = [
            [0.011371851, -0.16351199, -0.4190402],
            [-0.620872, -0.10241318, 0.16831112],
            [-0.8375113, -0.022119522, 0.5023525],
            [-0.9260869, 0.4402535, 0.43311024],
            [0.38382888, 0.40681696, -0.5775645],
            [-0.1622498, -0.64915395, -0.3980608],
            [-0.82016706, 0.6719475, 0.7351916],
            [0.6963484, -0.41792607, -0.76312375],
            [0.8004396, 0.9044554, -0.57431555],
            [-0.40366173, 0.11015701, -0.13648367],
            [0.21032763, 0.8652916, -0.57494235],
            [0.81958795, 0.87540984, -0.68126965],
            [0.73155046, 0.8409946, 0.5153327],
            [0.5080087, 0.047470808, 0.16673613],
            [0.29809928, -0.028960466, 0.3172562],
            [0.7131562, 0.9706938, 0.9241655],
            [-0.08819795, 0.41905618, -0.589731],
            [-0.49635363, 0.4587593, -0.9722729],
            [0.22035623, -0.68881845, 0.7012863],
            [0.083728075, 0.60909224, -0.79403067],
            [0.31492043, -0.5709784, 0.38491774],
            [0.032473803, 0.8270147, -0.61215043],
            [-0.9772687, 0.721421, 0.47607756],
            [-0.3524022, 0.8305044, -0.452116],
            [-0.93919516, 0.8466265, 0.41512346],
            [-0.30589652, 0.11914897, -0.8679035],
            [-0.7820468, -0.9360666, 0.9605527],
            [0.71809673, 0.23928714, 0.95128345],
            [-0.124883175, 0.06561279, -0.35512543],
            [0.0894928, 0.13267231, -0.89454174],
        ];
        let mut vectors = VectorsList::from_arrays(&vectors);
        let centroids = [
            [-0.8535168, 0.4259526, 0.45502776],
            [0.2825691, 0.7597978, -0.6094842],
            [-0.08711442, 0.008312721, -0.599587],
            [-0.082256705, -0.7319545, 0.6822522],
            [0.5937823, 0.4138972, 0.57495475],
        ];
        let centroids = VectorsList::from_arrays(&centroids);
        let clusters = [
            vec![1, 2, 3, 6, 22, 24],
            vec![4, 8, 10, 11, 19, 21, 23],
            vec![0, 5, 7, 9, 16, 17, 25, 28, 29],
            vec![18, 20, 26],
            vec![12, 13, 14, 15, 27],
        ];
        let search = [
            0.16593146,
            -0.7438214,
            0.014241219,
        ];
        let nprobe = 3;
        let topk = 6;
        let expected_perfect = [
            SearchPair::from((20, 0.18947348)),
            SearchPair::from((5, 0.2866578)),
            SearchPair::from((18, 0.47801834)),
            SearchPair::from((0, 0.5483805)),
            SearchPair::from((14, 0.6203126)),
            SearchPair::from((13, 0.7664149)),
        ];

        let result = flat_search(&vectors, &search, topk);
        assert_eq!(expected_perfect, result.into_sorted_vec().as_slice());
        let result = search_clustered_by_indices(&vectors, &centroids, &clusters, &search, nprobe, topk);
        assert_eq!(expected_perfect, result.into_sorted_vec().as_slice());

        let result = search_clustered_by_indices(&vectors, &centroids, &clusters, &search, nprobe - 1, topk);
        let expected_some_missed = [
            SearchPair::from((20, 0.18947348)),
            SearchPair::from((5, 0.2866578)),
            SearchPair::from((18, 0.47801834)),
            SearchPair::from((0, 0.5483805)),
            SearchPair::from((28, 0.8761885)),
            SearchPair::from((7, 0.9918462)),
        ];
        assert_eq!(expected_some_missed, result.into_sorted_vec().as_slice());

        let (order, clusters) = apply_clusters(&mut vectors, clusters.as_slice());
        let expected_order = [
            1, 2, 3, 6, 22, 24,
            4, 8, 10, 11, 19, 21, 23,
            0, 5, 7, 9, 16, 17, 25, 28, 29,
            18, 20, 26,
            12, 13, 14, 15, 27,
        ];
        assert_eq!(expected_order, *order);

        let mut result = flat_search(&vectors, &search, topk).into_sorted_vec();
        restore_indices(result.as_mut_slice(), &order);
        assert_eq!(expected_perfect, result.as_slice());
        let mut result = search_clustered(&vectors, &centroids, &clusters, &search, nprobe, topk).into_sorted_vec();
        restore_indices(result.as_mut_slice(), &order);
        assert_eq!(expected_perfect, result.as_slice());

        let mut result = search_clustered(&vectors, &centroids, &clusters, &search, nprobe - 1, topk).into_sorted_vec();
        restore_indices(result.as_mut_slice(), &order);
        assert_eq!(expected_some_missed, result.as_slice());
    }
}
