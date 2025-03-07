#![feature(test)]
#![feature(portable_simd)]

extern crate test;

use crate::vectors::*;

#[path = "../src/vectors.rs"]
mod vectors;
#[path = "../src/kmeans.rs"]
mod kmeans;

/*
there is barely any difference between calc_distances_only and update_distances_and_accum, so optimizing everything else is not needed;
however moving accumulation into a separate loop slows things down by 2-3%
depending on how much faster we can get by parallelizing the distance calculation, this % can grow
accumulation can be done inside the parallel loop too, but it requires some changes (have 2 layers of cumulative probabilities and do selection in 2 steps)
 */

fn update_distances_and_accum(vectors: &VectorsList, last_picked_vector: &[f32], min_distances: &mut [f32], cumulative: &mut [f32]) {
    let mut distance_accumulated = 0.0;
    for (index, vector) in vectors.iter().enumerate() {
        let distance_from_newest_centroid = calc_distance_sq(vector, last_picked_vector);
        let min_distance = min_distances[index].min(distance_from_newest_centroid);
        min_distances[index] = min_distance;
        distance_accumulated += min_distance;
        cumulative[index] = distance_accumulated;
    }
}
fn update_distances_and_accum2(vectors: &VectorsList, last_picked_vector: &[f32], min_distances: &mut [f32], cumulative: &mut [f32]) {
    for (index, vector) in vectors.iter().enumerate() {
        let distance_from_newest_centroid = calc_distance_sq(vector, last_picked_vector);
        let min_distance = min_distances[index].min(distance_from_newest_centroid);
        min_distances[index] = min_distance;
    }
    let mut distance_accumulated = 0.0;
    for (index, distance) in min_distances.iter().enumerate() {
        distance_accumulated += distance;
        cumulative[index] = distance_accumulated;
    }
}
fn accumulate_only(min_distances: &[f32], cumulative: &mut [f32]) {
    let mut distance_accumulated = 0.0;
    for (index, distance) in min_distances.iter().enumerate() {
        distance_accumulated += distance;
        cumulative[index] = distance_accumulated;
    }
}

fn calc_distances_only(vectors: &VectorsList, last_picked_vector: &[f32], min_distances: &mut [f32]) {
    for (index, vector) in vectors.iter().enumerate() {
        let distance_from_newest_centroid = calc_distance_sq(vector, last_picked_vector);
        min_distances[index] = distance_from_newest_centroid;
    }
}

// no change in performance
fn calc_distances_only_v2(vectors: &VectorsList, last_picked_vector: &[f32], mut min_distances: &mut [f32]) {
    let mut data = &*vectors.data;
    let dim = vectors.dim;
    while data.len() > 0 {
        let vector = unsafe { data.get_unchecked(0..dim) };
        data = unsafe { data.get_unchecked(dim..) };
        let distance_from_newest_centroid = calc_distance_sq(vector, last_picked_vector);
        min_distances[0] = distance_from_newest_centroid;
        min_distances = unsafe{ min_distances.get_unchecked_mut(1..) };
    }
}

// no change in performance
fn calc_distances_only_multi(vectors: &VectorsList, last_picked_vector: &[f32], mut min_distances: &mut [f32]) {
    let mut data = &*vectors.data;
    let dim = vectors.dim;
    while data.len() > 0 {
        let d1 = calc_distance_sq(last_picked_vector, unsafe { data.get_unchecked(0..dim) });
        data = unsafe { data.get_unchecked(dim..) };
        let d2 = calc_distance_sq(last_picked_vector, unsafe { data.get_unchecked(0..dim) });
        data = unsafe { data.get_unchecked(dim..) };
        let d3 = calc_distance_sq(last_picked_vector, unsafe { data.get_unchecked(0..dim) });
        data = unsafe { data.get_unchecked(dim..) };
        let d4 = calc_distance_sq(last_picked_vector, unsafe { data.get_unchecked(0..dim) });
        data = unsafe { data.get_unchecked(dim..) };
        let distances = [d1, d2, d3, d4];
        min_distances[0..4].copy_from_slice(&distances);
        min_distances = unsafe{ min_distances.get_unchecked_mut(4..) };
    }
}

#[cfg(test)]
mod bench {
    use std::hint::black_box;
    use rand::rngs::SmallRng;
    use rand::{rng, SeedableRng};
    use crate::vectors::*;
    use crate::kmeans::*;
    use test::Bencher;
    use super::*;

    #[bench]
    fn bench_kmeans_pp_init(b: &mut Bencher) {
        let vectors_count = black_box(2000);
        let dimensions = black_box(512);
        let nlist = black_box(16);

        let seed = 2465599376081375548;
        let mut rng = SmallRng::seed_from_u64(seed);
        //let rng = rng();

        let vectors = VectorsList::new_random_norm(vectors_count, dimensions, &mut rng);

        b.iter(|| black_box(kmeans_pp_pick_indices(&vectors, nlist, &mut rng)));
    }

    fn init_vectors(count: usize, dimensions: usize) -> VectorsList {
        let vectors_count = black_box(count);
        let dimensions = black_box(dimensions);
        let seed = 2465599376081375548;
        let mut rng = SmallRng::seed_from_u64(seed);
        //let rng = rng();
        let vectors = VectorsList::new_random_norm(vectors_count, dimensions, &mut rng);
        vectors
    }

    #[bench]
    fn bench_calc_distances(b: &mut Bencher) {
        let vectors = init_vectors(2000, 512);
        let last_picked_vector = vectors.get(5);
        let mut min_distances = black_box(vec![0.0; vectors.count].into_boxed_slice());
        b.iter(
            || {
                black_box(calc_distances_only(&vectors, last_picked_vector, &mut min_distances))
            }
        );
    }

    #[bench]
    fn bench_calc_distances_v2(b: &mut Bencher) {
        let vectors = init_vectors(2000, 512);
        let last_picked_vector = vectors.get(5);
        let mut min_distances = black_box(vec![0.0; vectors.count].into_boxed_slice());
        b.iter(
            || {
                black_box(calc_distances_only_v2(&vectors, last_picked_vector, &mut min_distances))
            }
        );
    }

    #[bench]
    fn bench_calc_distances_multi(b: &mut Bencher) {
        let vectors = init_vectors(2000, 512);
        let last_picked_vector = vectors.get(5);
        let mut min_distances = black_box(vec![0.0; vectors.count].into_boxed_slice());
        b.iter(
            || {
                black_box(calc_distances_only_multi(&vectors, last_picked_vector, &mut min_distances))
            }
        );
    }

    #[bench]
    fn bench_update_distances_and_accum(b: &mut Bencher) {
        let vectors = init_vectors(2000, 512);
        let last_picked_vector = vectors.get(5);
        let mut min_distances = black_box(vec![0.0; vectors.count].into_boxed_slice());
        let mut cumulative = black_box(vec![0.0; vectors.count].into_boxed_slice());
        b.iter(
            || {
                black_box(update_distances_and_accum(&vectors, last_picked_vector, &mut min_distances, &mut cumulative))
            }
        );
    }

    #[bench]
    fn bench_update_distances_and_accum2(b: &mut Bencher) {
        let vectors = init_vectors(2000, 512);
        let last_picked_vector = vectors.get(5);
        let mut min_distances = black_box(vec![0.0; vectors.count].into_boxed_slice());
        let mut cumulative = black_box(vec![0.0; vectors.count].into_boxed_slice());
        b.iter(
            || {
                black_box(update_distances_and_accum2(&vectors, last_picked_vector, &mut min_distances, &mut cumulative))
            }
        );
    }

    #[bench]
    fn bench_accum(b: &mut Bencher) {
        let vectors = init_vectors(2000, 512);
        let last_picked_vector = vectors.get(5);
        let mut min_distances = black_box(vec![f32::MAX; vectors.count].into_boxed_slice());
        calc_distances_only(&vectors, last_picked_vector, &mut min_distances);
            
        let mut cumulative = black_box(vec![0.0; vectors.count].into_boxed_slice());
        b.iter(
            || {
                black_box(accumulate_only(&min_distances, &mut cumulative))
            }
        );
    }
}