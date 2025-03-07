use std::mem;
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::SystemTime;
use rand::Rng;
use crate::vectors::*;

pub(crate) trait InitCentroids {
    fn run(&mut self, vectors: &VectorsList) -> VectorsList;
}

pub(crate) struct InitKmeansPP<R: Rng> {
    pub nlist: usize,
    pub rng: R,
}
impl<R: Rng> InitCentroids for InitKmeansPP<R> {
    fn run(&mut self, vectors: &VectorsList) -> VectorsList {
        let indices = kmeans_pp_pick_indices(vectors, self.nlist, &mut self.rng);
        VectorsList::from_vectors_by_indices(vectors, indices.as_slice())
    }
}

pub(crate) struct InitRandom<R: Rng> {
    pub nlist: usize,
    pub rng: R,
}
impl<R: Rng> InitCentroids for InitRandom<R> {
    fn run(&mut self, vectors: &VectorsList) -> VectorsList {
        let indices = pick_random_indices(vectors, self.nlist, &mut self.rng);
        VectorsList::from_vectors_by_indices(vectors, indices.as_slice())
    }
}

pub(crate) fn kmeans_multi_try_multi_thread(
    vectors: &VectorsList,
    thread_initializers: &mut [impl InitCentroids + Send],
    tries_per_thread: usize,
    max_try_iterations: usize,
) -> (VectorsList, Vec<Vec<usize>>, f32, usize, usize) {
    debug_assert!(tries_per_thread > 0);
    debug_assert!(thread_initializers.len() > 0);
    let mut best_wcss = f32::MAX;
    let mut best_centroids = VectorsList::new(1, 1);
    let mut best_cluster = vec![];
    let mut best_thread = 0;
    let mut best_iter = 0;
    std::thread::scope(|s| {
        let mut handles = Vec::with_capacity(thread_initializers.len());
        for init in thread_initializers {
            let handle = s.spawn(|| {
                kmeans_multi_try(vectors, init, tries_per_thread, max_try_iterations)
            });
            handles.push(handle);
        }
        for (thread_no, handle) in handles.into_iter().enumerate() {
            let (centroids, cluster, wcss, iter) = handle.join().unwrap();
            if wcss < best_wcss {
                best_centroids = centroids;
                best_cluster = cluster;
                best_wcss = wcss;
                best_thread = thread_no;
                best_iter = iter;
            }
        }
    });
    (best_centroids, best_cluster, best_wcss, best_thread, best_iter)
}

pub(crate) fn kmeans_multi_try(
    vectors: &VectorsList,
    init_centroids_fn: &mut impl InitCentroids,
    tries: usize,
    max_try_iterations: usize,
) -> (VectorsList, Vec<Vec<usize>>, f32, usize) {
    debug_assert!(tries > 0);
    let mut best_wcss = f32::MAX;
    let mut best_centroids = VectorsList::new(1, 1);
    let mut best_cluster = vec![];
    let mut best_iteration = 0;
    for iteration in 0..tries {
        let start = SystemTime::now();
        let init_centroids = init_centroids_fn.run(vectors);
        let duration = SystemTime::now().duration_since(start);
        //println!("init centroids in {duration:?}");
        let (centroids, cluster, wcss, _) = kmeans(&vectors, init_centroids, max_try_iterations);
        if wcss < best_wcss {
            best_centroids = centroids;
            best_cluster = cluster;
            best_wcss = wcss;
            best_iteration = iteration;
        }
    }
    (best_centroids, best_cluster, best_wcss, best_iteration)
}

fn kmeans(vectors: &VectorsList, init_centroids: VectorsList, max_iterations: usize) -> (VectorsList, Vec<Vec<usize>>, f32, usize) {
    let nlist = init_centroids.count;
    let dimensions = vectors.dim;
    debug_assert!(nlist > 1);
    debug_assert!(nlist <= vectors.count);
    debug_assert_eq!(init_centroids.dim, dimensions);
    debug_assert!(max_iterations > 0);

    let mut centroids = init_centroids;
    let mut centroids_new = VectorsList::new(nlist, dimensions);
    let mut clusters = Vec::with_capacity(nlist); // todo: potentially optimise
    let estimate_cluster_size = (vectors.count + nlist + 1) / nlist; // intdiv with ceil
    for _ in 0..nlist {
        clusters.push(Vec::with_capacity(estimate_cluster_size)); // todo: potentially optimise
    }

    let mut iteration = 0;
    let mut is_any_changed = true;
    while is_any_changed {
        let start = SystemTime::now();
        iteration += 1;
        for cluster in clusters.iter_mut() {
            cluster.truncate(0);
        }
        for (vector_index, vector) in vectors.iter().enumerate() {
            // todo: potentially optimise
            let mut min_distance = f32::MAX;
            let mut min_distance_index = 0;
            for (centroid_index, centroid_vector) in centroids.iter().enumerate() {
                let distance = calc_distance_sq(vector, centroid_vector);
                if distance < min_distance {
                    min_distance = distance;
                    min_distance_index = centroid_index;
                }
            }
            clusters[min_distance_index].push(vector_index);
        }
        if iteration > max_iterations {
            // can't break at the start of the loop, need to calculate clusters since we've updated centroids
            break;
        }
        let mid = SystemTime::now();
        is_any_changed = false;
        for (centroid_index, new_centroid) in centroids_new.iter_mut().enumerate() {
            // todo: potentially optimise
            let cluster = &clusters[centroid_index];
            calc_centroid(cluster, vectors, new_centroid);
            let is_changed = !is_same_vector(centroids.get(centroid_index), new_centroid);
            is_any_changed = is_any_changed || is_changed;
        }
        mem::swap(&mut centroids, &mut centroids_new);
        let end = SystemTime::now();
        let duration = end.duration_since(start);
        let clusters_duration = mid.duration_since(start);
        let move_centroids_duration = end.duration_since(mid);
        println!("kmeans iteration {iteration} in {duration:?}. clustering {clusters_duration:?} move centroids {move_centroids_duration:?}")
    }
    let start = SystemTime::now();
    let wcss = wcss(&centroids, &clusters, vectors);
    let duration = SystemTime::now().duration_since(start);
    println!("final wcss in {duration:?}");

    (centroids, clusters, wcss, iteration)
}

fn _kmeans2(vectors: &VectorsList, init_centroids: VectorsList, max_iterations: usize) -> (VectorsList, Vec<Vec<usize>>, f32, usize) {
    /*
    todo:
        an attempt at multi-threaded implementation
        using 8 threads achieves around x3.75 speedup (time is measured only for the affected section, not the whole function)
        using separate threads for each call of kmeans function achieves x8 speedup for 8 threads
            x8 when using random init and 200k vectors
            x7 when using kmeans++ and 200k vectors
            x7 when using random init and 1M vectors
            and this approach has no penalty when using 1 thread, while the inner thread approach is slower than not threaded implementation when using only 1 thread
        so i'm going to be using that approach instead
        however, the approach with outer threads also uses more memory, so maybe it would degrade at some point?
            the largest usage of memory is by the index, which takes at least num_vectors*sizeof(usize)
            so for dim=512, index is taking 1/256 of the memory of the original dataset
            yeah, probably should not degrade because of memory 
     */
    let nlist = init_centroids.count;
    let dimensions = vectors.dim;
    debug_assert!(nlist > 1);
    debug_assert!(nlist <= vectors.count);
    debug_assert_eq!(init_centroids.dim, dimensions);
    debug_assert!(max_iterations > 0);

    let mut centroids = init_centroids;
    let mut centroids_new = VectorsList::new(nlist, dimensions);
    let mut clusters = Vec::with_capacity(nlist); // todo: potentially optimise
    let estimate_cluster_size = (vectors.count + nlist + 1) / nlist; // intdiv with ceil
    for _ in 0..nlist {
        clusters.push(Vec::with_capacity(estimate_cluster_size)); // todo: potentially optimise
    }

    let mut iteration = 0;
    let mut is_any_changed = true;
    // todo: do better chunking
    let threads_count = 8.min(vectors.count);
    let thread_chunk_size = int_div_ceil(vectors.count, threads_count);
    while is_any_changed {
        let start = SystemTime::now();
        iteration += 1;
        for cluster in clusters.iter_mut() {
            cluster.truncate(0);
        }

        std::thread::scope(|s| {
            let centroids = &centroids;
            let (tx, rx) = mpsc::channel();
            let tx = Arc::new(tx);
            for (chunk_no, chunk) in vectors.data.chunks(thread_chunk_size * vectors.dim).enumerate() {
                let tx = tx.clone();
                s.spawn(move || {
                    for (index_in_chunk, vector) in chunk.chunks(vectors.dim).enumerate() {
                        // todo: potentially optimise
                        let mut min_distance = f32::MAX;
                        let mut min_distance_index = 0;
                        for (centroid_index, centroid_vector) in centroids.iter().enumerate() {
                            let distance = calc_distance_sq(vector, centroid_vector);
                            if distance < min_distance {
                                min_distance = distance;
                                min_distance_index = centroid_index;
                            }
                        }
                        let vector_index = (chunk_no * thread_chunk_size) + index_in_chunk;
                        tx.send((min_distance_index, vector_index)).unwrap();
                    }
                });
            }
            drop(tx); // make sure that receiver can stop
            for (min_distance_index, vector_index) in &rx {
                clusters[min_distance_index].push(vector_index);
            }
        });

        if iteration > max_iterations {
            // can't break at the start of the loop, need to calculate clusters since we've updated centroids
            break;
        }
        let mid = SystemTime::now();
        is_any_changed = false;
        for (centroid_index, new_centroid) in centroids_new.iter_mut().enumerate() {
            // todo: potentially optimise
            let cluster = &clusters[centroid_index];
            calc_centroid(cluster, vectors, new_centroid);
            let is_changed = !is_same_vector(centroids.get(centroid_index), new_centroid);
            is_any_changed = is_any_changed || is_changed;
        }
        mem::swap(&mut centroids, &mut centroids_new);
        let end = SystemTime::now();
        let duration = end.duration_since(start);
        let clusters_duration = mid.duration_since(start);
        let move_centroids_duration = end.duration_since(mid);
        println!("kmeans iteration {iteration} in {duration:?}. clustering {clusters_duration:?} move centroids {move_centroids_duration:?}")
    }
    let start = SystemTime::now();
    let wcss = wcss(&centroids, &clusters, vectors);
    let duration = SystemTime::now().duration_since(start);
    //println!("final wcss in {duration:?}");

    (centroids, clusters, wcss, iteration)
}

fn calc_centroid(cluster: &[usize], vectors: &VectorsList, new_centroid: &mut [f32]) {
    new_centroid.fill(0.0);
    let len = cluster.len() as f32;
    for &vector_index in cluster {
        let vector = vectors.get(vector_index);
        for (index, value) in vector.iter().enumerate() {
            new_centroid[index] += value / len;
        }
    }
}

fn wcss(centroids: &VectorsList, clusters: &[Vec<usize>], vectors: &VectorsList) -> f32 {
    let mut wcss = 0.0;
    for (i, cluster) in clusters.iter().enumerate() {
        let centroid = centroids.get(i);
        for &vector_index in cluster {
            let distance_sq = calc_distance_sq(centroid, vectors.get(vector_index));
            wcss += distance_sq;
        }
    }
    wcss
}

fn pick_random_indices(vectors: &VectorsList, nlist: usize, rng: &mut impl Rng) -> Vec<usize> {
    let nlist = nlist.min(vectors.count);
    let mut indices = Vec::with_capacity(nlist);
    while indices.len() < nlist {
        let rand = rng.random_range(0..vectors.count);
        if !indices.contains(&rand) {
            indices.push(rand);
        }
    }
    indices
}

// made public for benchmarks
pub(crate) fn kmeans_pp_pick_indices(vectors: &VectorsList, select_count: usize, rng: &mut impl Rng) -> Vec<usize> {
    // todo: potentially optimise
    let count = vectors.count;
    let select_count = select_count.min(count);
    if select_count == 0 {
        return Vec::new();
    }
    let mut indices = Vec::with_capacity(select_count);
    let mut min_distances = vec![f32::MAX; count].into_boxed_slice();
    let mut cumulative = vec![0.0; count].into_boxed_slice();
    let first_index = rng.random_range(0..count);
    indices.push(first_index);
    while indices.len() < select_count {
        update_distances(vectors, vectors.get(*indices.last().unwrap()), &mut min_distances, &mut cumulative);
        let prob_range_end = cumulative.last().copied().unwrap();
        if prob_range_end == 0.0 { // todo: do i need to compare with epsilon instead?
            break;
        }
        let rand = rng.random::<f32>() * prob_range_end; // todo: looks like random::<f32> generates a float between 0 and 1, but i'm not sure if that's reliable
        let index = pick_from_distribution(&cumulative, rand);
        indices.push(index);
    }
    indices
}

fn update_distances(vectors: &VectorsList, last_picked_vector: &[f32], min_distances: &mut [f32], cumulative: &mut [f32]){
    // using cumulative probability distribution
    debug_assert_eq!(vectors.count, cumulative.len());
    debug_assert_eq!(vectors.count, min_distances.len());
    
    let mut distance_accumulated = 0.0;
    for (index, vector) in vectors.iter().enumerate() {
        let distance_from_newest_centroid = calc_distance_sq(vector, last_picked_vector);
        let min_distance = min_distances[index].min(distance_from_newest_centroid);
        min_distances[index] = min_distance;
        distance_accumulated += min_distance;
        cumulative[index] = distance_accumulated;
    }
}

fn _update_distances2(vectors: &VectorsList, last_picked_vector: &[f32], min_distances: &mut [f32], cumulative: &mut [f32], thread_chunk_size: usize, handles: &mut Vec<JoinHandle<()>>) {
    // using cumulative probability distribution
    debug_assert_eq!(vectors.count, cumulative.len());
    debug_assert_eq!(vectors.count, min_distances.len());
    debug_assert_eq!(handles.len(), 0);
    /*
    todo:
        regardless of the size, i can only get to around x2 speedup by having around 4 threads
        and the timing of each run is very different from each other (every call to update_distances when we pick a centroid)
        splitting into 100 threads makes each thread finish at around 1/20 of original time, measured inside the thread closure
        and there is a huge variation in execution time, from 1 to 8 ms per job (when splitting 1M set into 100 threads)
        however reducing the size of the original dataset does reliably reduce the execution time of jobs
        200K set in 1 thread is processed more than 2 times faster than each of 125K jobs when splitting 1M set into 8 threads
        i guess it's cache or memory contention? or thread scheduling?
        rewriting update_distances_chunk to store multiple values at the same time instead of one by one did not change anything
     */
    let min_distance_parts = min_distances.chunks_mut(thread_chunk_size);
    debug_assert_eq!(min_distance_parts.len(), handles.capacity());
    for (chunk_no, chunk) in min_distance_parts.enumerate() {
        let job = move || {
            update_distances_chunk(vectors, last_picked_vector, chunk, chunk_no, thread_chunk_size);
        };
        // todo: test if using sending results to a channel and collecting them in the main thread would be faster?
        // using this hack instead of scoped threads to be able to easily change this to a thread pool
        let job = unsafe { make_static(Box::new(job)) };
        // todo: check if using threads here makes sense when we are using threads for kmeans retries
        let handle = std::thread::spawn(job);
        handles.push(handle);
    }
    while let Some(handle) = handles.pop() {
        handle.join().unwrap();
    }

    /*
    todo: separate accumulation takes around 5% of time when the main part is done in multi threads
        there is a way to move this into the threads too, but i don't want to bother for just 5%
        have several separately accumulated parts, and make a 2 step choice - first weighted choice of a part, then within a part
     */
    let mut distance_accumulated = 0.0;
    for (index, distance) in min_distances.iter().enumerate() {
        distance_accumulated += distance;
        cumulative[index] = distance_accumulated;
    }
}

pub(crate) unsafe fn make_static<'a, T>(f: Box<dyn FnOnce() -> T + Send + 'a>) -> Box<dyn FnOnce() -> T + Send + 'static> {
    mem::transmute(f)
}

fn int_div_ceil(n: usize, div_by: usize) -> usize {
    (n + div_by - 1) / div_by
}

// extracted to be able to swap implementations
fn update_distances_chunk(vectors: &VectorsList, last_picked_vector: &[f32], chunk: &mut [f32], chunk_no: usize, thread_chunk_size: usize) {
    let vector_start = chunk_no * thread_chunk_size;
    let vector_count = chunk.len();
    for (index, vector) in vectors.get_cluster(vector_start, vector_count).enumerate() {
        let distance_from_newest_centroid = calc_distance_sq(vector, last_picked_vector);
        let min_distance = chunk[index].min(distance_from_newest_centroid);
        chunk[index] = min_distance;
    }
}

fn pick_from_distribution(dist: &[f32], seek: f32) -> usize {
    //let found = dist.iter().position(|&v| v > seek);
    let find_result = dist.binary_search_by(
        |&probe| probe.partial_cmp(&seek).unwrap()
    );
    let index = match find_result {
        Ok(x) => x + 1,
        Err(x) => x,
    };
    if index < dist.len() {
        return index;
    }

    /*
    todo: can this be optimized?
        ideally we should guarantee that seek is smaller than last value in the distribution
        and a special case where whole distribution is just zeroes
        looks like this code is not being run in practice, however i'm not sure how to guarantee it
     */
    let last = dist[dist.len() - 1];
    let before_last = dist[dist.len() - 2];
    if last > before_last {
        return dist.len() - 1;
    }
    if seek == 0.0 {
        return 0;
    }
    dist.iter().rposition(|&v| v < last).unwrap_or(0)
}

#[cfg(test)]
mod test {
    use rand::random;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use crate::vectors::VectorsList;
    use super::*;

    #[test]
    fn test_calc_centroid() {
        let vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-6.0, -6.0, -6.0],
            [7.0, 8.0, 9.0],
            [-2.0, -2.0, -2.0],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let cluster = [0, 1, 3, 4];
        let mut new_centroid = vec![0.0; vectors.dim];
        calc_centroid(&cluster, &vectors, new_centroid.as_mut_slice());
        let expected_vector = [2.5, 3.25, 4.0];
        assert_eq!(expected_vector, new_centroid.as_slice());
    }

    #[test]
    fn test_kmeans() {
        // rectangle, long sides
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
        ];
        let init_centroids_orig = [
            [0.5,  0.5],
            [0.5, -0.5],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let init_centroids = VectorsList::from_arrays(&init_centroids_orig);
        let max_iterations = 100;
        let (centroids, clusters, wcss, _) = kmeans(&vectors, init_centroids, max_iterations);
        let expected_centroids = [
            [0.0,  1.0],
            [0.0, -1.0],
        ];
        let expected_clusters = [
            [0, 1].as_slice(),
            [2, 3].as_slice(),
        ];
        assert_eq!(expected_centroids, centroids.as_vec_of_slices().as_slice());
        assert_eq!(expected_clusters, clusters.as_slice());
        assert_eq!(16.0, wcss);

        // 1 iteration
        let init_centroids = VectorsList::from_arrays(&init_centroids_orig);
        let (centroids, clusters, wcss, _) = kmeans(&vectors, init_centroids, 1);
        assert_eq!(expected_centroids, centroids.as_vec_of_slices().as_slice());
        assert_eq!(expected_clusters, clusters.as_slice());
        assert_eq!(16.0, wcss);

        // rectangle, short sides
        let init_centroids = [
            [ 0.5, 0.5],
            [-0.5, 0.5],
        ];
        let init_centroids = VectorsList::from_arrays(&init_centroids);
        let (centroids, clusters, wcss, _) = kmeans(&vectors, init_centroids, max_iterations);
        let expected_centroids = [
            [ 2.0, 0.0],
            [-2.0, 0.0],
        ];
        let expected_clusters = [
            [1, 3].as_slice(),
            [0, 2].as_slice(),
        ];
        assert_eq!(expected_centroids, centroids.as_vec_of_slices().as_slice());
        assert_eq!(expected_clusters, clusters.as_slice());
        assert_eq!(4.0, wcss);

        // nlist == vectors count
        let init_centroids = [
            [-0.1,  0.1],
            [ 0.2,  0.2],
            [-0.3, -0.3],
            [ 0.4, -0.4],
        ];
        let init_centroids = VectorsList::from_arrays(&init_centroids);
        let (centroids, clusters, wcss, _) = kmeans(&vectors, init_centroids, max_iterations);
        let expected_centroids = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
        ];
        let expected_clusters = [
            [0].as_slice(),
            [1].as_slice(),
            [2].as_slice(),
            [3].as_slice(),
        ];
        assert_eq!(expected_centroids, centroids.as_vec_of_slices().as_slice());
        assert_eq!(expected_clusters, clusters.as_slice());
        assert_eq!(0.0, wcss);
    }

    #[test]
    fn test_kmeans_large() {
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
        let init_centroids_orig = [
            [-0.7820468, -0.9360666, 0.9605527],
            [0.81958795, 0.87540984, -0.68126965],
            [-0.49635363, 0.4587593, -0.9722729],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let init_centroids = VectorsList::from_arrays(&init_centroids_orig);
        let max_iterations = 100;
        let (centroids, clusters, wcss, _) = kmeans(&vectors, init_centroids, max_iterations);
        let expected_centroids = [
            [-0.84330684, 0.23137842, 0.5272456],
            [0.50059825, 0.11566986, 0.565854],
            [0.074622124, 0.3370875, -0.60391706],
        ];
        let expected_clusters = [
            [1, 2, 3, 6, 22, 24, 26].as_slice(),
            [12, 13, 14, 15, 18, 20, 27].as_slice(),
            [0, 4, 5, 7, 8, 9, 10, 11, 16, 17, 19, 21, 23, 25, 28, 29].as_slice(),
        ];
        assert_eq!(expected_centroids, centroids.as_vec_of_slices().as_slice());
        assert_eq!(expected_clusters, clusters.as_slice());
        assert_eq!(12.950671, wcss);
    }

    #[test]
    #[should_panic]
    fn test_kmeans_dimensions_mismatch() {
        let vectors = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ];
        let init_centroids = [
            [0.5,  0.5],
            [0.5, -0.5],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let init_centroids = VectorsList::from_arrays(&init_centroids);
        kmeans(&vectors, init_centroids, 100);
    }

    #[test]
    #[should_panic]
    fn test_kmeans_zero_iterations() {
        let vectors = [
            [1.0, 2.0],
            [4.0, 5.0],
        ];
        let init_centroids = [
            [0.5,  0.5],
            [0.5, -0.5],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let init_centroids = VectorsList::from_arrays(&init_centroids);
        kmeans(&vectors, init_centroids, 0);
    }

    #[test]
    fn test_update_probability_distributions() {
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let mut min_distances = vec![f32::MAX; vectors.count];
        let mut cumulative = vec![0.0; vectors.count];

        update_distances(&vectors, vectors.get(0), min_distances.as_mut_slice(), cumulative.as_mut_slice());
        let expected_weights = [0.0, 16.0, 20.0, 40.0];
        assert_eq!(expected_weights, cumulative.as_slice());

        min_distances.fill(f32::MAX);
        update_distances(&vectors, vectors.get(1), min_distances.as_mut_slice(), cumulative.as_mut_slice());
        let expected_weights = [16.0, 16.0, 36.0, 40.0];
        assert_eq!(expected_weights, cumulative.as_slice());

        update_distances(&vectors, vectors.get(0), min_distances.as_mut_slice(), cumulative.as_mut_slice());
        let expected_weights = [0.0, 0.0, 4.0, 8.0];
        assert_eq!(expected_weights, cumulative.as_slice());
    }

    #[test]
    #[should_panic]
    fn test_update_probability_distributions_invalid_buffer() {
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let mut min_distances = vec![f32::MAX; 2];
        let mut cumulative = vec![0.0; vectors.count];
        update_distances(&vectors, vectors.get(0), min_distances.as_mut_slice(), cumulative.as_mut_slice());
    }

    #[test]
    #[should_panic]
    fn test_update_probability_distributions_invalid_buffer2() {
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
        ];
        let vectors = VectorsList::from_arrays(&vectors);
        let mut min_distances = vec![f32::MAX; vectors.count];
        let mut cumulative = vec![0.0; 2];
        update_distances(&vectors, vectors.get(0), min_distances.as_mut_slice(), cumulative.as_mut_slice());
    }

    #[test]
    fn test_pick_from_distribution() {
        let weights = [16.0, 16.0, 36.0, 40.0];
        assert_eq!(0, pick_from_distribution(&weights, 0.0));
        assert_eq!(0, pick_from_distribution(&weights, 15.999));
        assert_eq!(2, pick_from_distribution(&weights, 16.0));
        assert_eq!(2, pick_from_distribution(&weights, 35.999));
        assert_eq!(3, pick_from_distribution(&weights, 36.0));
        assert_eq!(3, pick_from_distribution(&weights, 39.999));
        assert_eq!(3, pick_from_distribution(&weights, 40.0));
        assert_eq!(3, pick_from_distribution(&weights, 500.0));
        let weights = [0.0, 16.0, 20.0, 40.0];
        assert_eq!(1, pick_from_distribution(&weights, 0.0));
        assert_eq!(1, pick_from_distribution(&weights, 15.0));
        assert_eq!(2, pick_from_distribution(&weights, 16.0));
        let weights = [10.0; 100];
        assert_eq!(0, pick_from_distribution(&weights, 9.0));
        assert_eq!(0, pick_from_distribution(&weights, 10.0));
        assert_eq!(0, pick_from_distribution(&weights, 11.0));
        let weights = [0.0; 100];
        assert_eq!(0, pick_from_distribution(&weights, 0.0));
        assert_eq!(0, pick_from_distribution(&weights, 1.0));
    }

    #[test]
    fn test_kmeans_pp_pick_indices_all() {
        let seed = random();
        println!("seed {seed}");
        let mut rng = SmallRng::seed_from_u64(seed);

        // pick all vectors
        let vectors_orig = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-6.0, -6.0, -6.0],
            [7.0, 8.0, 9.0],
            [-2.0, -2.0, -2.0],
            [1.000001, 2.0, 3.0],
        ];
        let vectors = VectorsList::from_arrays(&vectors_orig);
        let mut picked = kmeans_pp_pick_indices(&vectors, vectors.count, &mut rng);
        picked.sort();
        assert_eq!([0, 1, 2, 3, 4, 5], picked.as_slice());
        let mut picked = kmeans_pp_pick_indices(&vectors, vectors.count + 10, &mut rng);
        picked.sort();
        assert_eq!([0, 1, 2, 3, 4, 5], picked.as_slice());
        let picked = kmeans_pp_pick_indices(&vectors, 0, &mut rng);
        assert_eq!(Vec::<usize>::new(), picked);

        // pick all non-duplicated vectors
        let vectors_with_duplicates_orig = [
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0],
            [-6.0, -6.0, -6.0],
            [-6.0, -6.0, -6.0],
            [7.0, 8.0, 9.0],
            [7.0, 8.0, 9.0],
            [-2.0, -2.0, -2.0],
            [1.000001, 2.0, 3.0],
        ];
        let vectors = VectorsList::from_arrays(&vectors_with_duplicates_orig);
        let mut picked = kmeans_pp_pick_indices(&vectors, 6, &mut rng);
        picked.sort();
        let selected = VectorsList::from_vectors_by_indices(&vectors, &picked);
        assert_eq!(vectors_orig, selected.as_vec_of_slices().as_slice());

        // try to pick all vectors with duplicates - don't actually return duplicates
        let mut picked = kmeans_pp_pick_indices(&vectors, vectors.count, &mut rng);
        picked.sort();
        let selected = VectorsList::from_vectors_by_indices(&vectors, &picked);
        assert_eq!(vectors_orig, selected.as_vec_of_slices().as_slice());
    }

    #[test]
    fn test_pick_random_indices_all() {
        let seed = random();
        println!("seed {seed}");
        let mut rng = SmallRng::seed_from_u64(seed);

        // pick all vectors
        let vectors_orig = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-6.0, -6.0, -6.0],
            [7.0, 8.0, 9.0],
            [-2.0, -2.0, -2.0],
            [1.000001, 2.0, 3.0],
        ];
        let vectors = VectorsList::from_arrays(&vectors_orig);
        let mut picked = pick_random_indices(&vectors, vectors.count, &mut rng);
        picked.sort();
        assert_eq!([0, 1, 2, 3, 4, 5], picked.as_slice());
        let mut picked = pick_random_indices(&vectors, vectors.count + 10, &mut rng);
        picked.sort();
        assert_eq!([0, 1, 2, 3, 4, 5], picked.as_slice());
        let picked = pick_random_indices(&vectors, 0, &mut rng);
        assert_eq!(Vec::<usize>::new(), picked);
    }

    #[test]
    fn test_pick_kmeans_multi_try() {
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
            [-2.0, -1.0],
            [ 2.0, -1.0],
        ];
        let init_centroids_bad = [
            [0.5,  0.5],
            [0.5, -0.5],
        ];
        let init_centroids_good = [
            [ 0.5, 0.5],
            [-0.5, 0.5],
        ];
        let mut init = InitCentroidsMock{
            num: 0,
            options: [init_centroids_bad, init_centroids_good],
        };
        let vectors = VectorsList::from_arrays(&vectors);
        let (centroids, clusters, wcss, _) = kmeans_multi_try(&vectors, &mut init, 5, 100);

        let expected_centroids = [
            [ 2.0, 0.0],
            [-2.0, 0.0],
        ];
        let expected_clusters = [
            [1, 3].as_slice(),
            [0, 2].as_slice(),
        ];
        assert_eq!(expected_centroids, centroids.as_vec_of_slices().as_slice());
        assert_eq!(expected_clusters, clusters.as_slice());
        assert_eq!(4.0, wcss);
    }

    #[test]
    #[should_panic]
    fn test_pick_kmeans_multi_try_zero_tries() {
        let vectors = [
            [-2.0,  1.0],
            [ 2.0,  1.0],
        ];
        let init_centroids_bad = [
            [0.5,  0.5],
            [0.5, -0.5],
        ];
        let init_centroids_good = [
            [ 0.5, 0.5],
            [-0.5, 0.5],
        ];
        let mut init = InitCentroidsMock{
            num: 0,
            options: [init_centroids_bad, init_centroids_good],
        };
        let vectors = VectorsList::from_arrays(&vectors);
        kmeans_multi_try(&vectors, &mut init, 0, 100);
    }

    struct InitCentroidsMock<const OPT: usize, const NLIST: usize, const DIM: usize> {
        options: [[[f32; DIM]; NLIST]; OPT],
        num: usize,
    }
    impl<const OPT: usize, const NLIST: usize, const DIM: usize> InitCentroids for InitCentroidsMock<OPT, NLIST, DIM> {
        fn run(&mut self, _vectors: &VectorsList) -> VectorsList {
            let picked = self.num % self.options.len();
            self.num += 1;
            VectorsList::from_arrays(&self.options[picked])
        }
    }
}