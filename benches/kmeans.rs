#![feature(test)]
#![feature(portable_simd)]

extern crate test;
#[path = "../src/vectors.rs"]
mod vectors;
#[path = "../src/kmeans.rs"]
mod kmeans;

#[cfg(test)]
mod bench {
    use std::hint::black_box;
    use rand::rngs::SmallRng;
    use rand::{rng, SeedableRng};
    use crate::vectors::*;
    use crate::kmeans::*;
    use test::Bencher;

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
}