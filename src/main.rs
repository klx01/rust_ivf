#![feature(portable_simd)]
mod vectors;
mod kmeans;
mod search;

use std::time::{Duration, SystemTime};
use rand::{random, rng, Rng};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use crate::kmeans::*;
use crate::vectors::*;
use crate::search::*;

fn main() {
    let vectors_count = 200000;
    let dimensions = 512;
    let nlist = 64;
    let threads = 8;
    let kmeans_tries_per_thread = 1;
    let max_iterations = 100;
    let nprobe = 16;
    let topk = 5;

    //let seed = random();
    let seed = 2465599376081375548;
    println!("seed {seed}");
    let mut rng = SmallRng::seed_from_u64(seed);

    let start = SystemTime::now();
    let mut vectors = VectorsList::new_random_norm(vectors_count, dimensions, &mut rng);
    let duration = SystemTime::now().duration_since(start);
    println!("generated in {duration:?}");

    let mut inits = Vec::with_capacity(threads);
    for _ in 0..threads {
        // todo: decide which is the best way to init, random or kmeans++
        // todo: actually in my fixed setup random seems to have better recall than ++
        inits.push(InitRandom{
            nlist,
            rng: SmallRng::seed_from_u64(rng.random()),
        });
        /*inits.push(InitKmeansPP{
            nlist,
            rng: SmallRng::seed_from_u64(rng.random()),
        });*/
    }
    println!("start clustering in {threads} threads {kmeans_tries_per_thread} tries each");
    let start = SystemTime::now();
    let (centroids, clusters, wcss, best_thread, best_iter) = kmeans_multi_try_multi_thread(&vectors, inits.as_mut_slice(), kmeans_tries_per_thread, max_iterations);
    let duration = SystemTime::now().duration_since(start);
    println!("clustered in {duration:?} wcss {wcss} best thread {best_thread} iter {best_iter}");

    // todo: after optimizing clustering, optimize search too!

    let search = vectors.get(5).to_vec();

    let start = SystemTime::now();
    let found = flat_search(&vectors, search.as_slice(), topk);
    let duration = SystemTime::now().duration_since(start);
    println!("flat search in {duration:?}");
    dbg!(found.into_sorted_vec());

    let start = SystemTime::now();
    let found = search_clustered_by_indices(&vectors, &centroids, clusters.as_slice(), search.as_slice(), nprobe, topk);
    let duration = SystemTime::now().duration_since(start);
    println!("clustered search by indices in {duration:?}");
    dbg!(found.into_sorted_vec());

    let start = SystemTime::now();
    let (order, clusters) = apply_clusters(&mut vectors, clusters.as_slice());
    let duration = SystemTime::now().duration_since(start);
    println!("apply clusters in {duration:?}");

    let start = SystemTime::now();
    let found = flat_search(&vectors, search.as_slice(), topk);
    let duration = SystemTime::now().duration_since(start);
    println!("flat search after clustering in {duration:?}");
    let mut found = found.into_sorted_vec();
    dbg!(&found);
    restore_indices(&mut found, &order);
    dbg!(found);

    let start = SystemTime::now();
    let found = search_clustered(&vectors, &centroids, &clusters, search.as_slice(), nprobe, topk);
    let duration = SystemTime::now().duration_since(start);
    println!("clustered search in {duration:?}");
    let mut found = found.into_sorted_vec();
    dbg!(&found);
    restore_indices(&mut found, &order);
    dbg!(found);

    /*
reference result:
[src/main.rs:45:5] found.into_sorted_vec() = [
    SearchPair {
        distance_sq: 0.0,
        index: 5,
    },
    SearchPair {
        distance_sq: 1.534167,
        index: 94300,
    },
    SearchPair {
        distance_sq: 1.6029203,
        index: 197694,
    },
    SearchPair {
        distance_sq: 1.6138234,
        index: 35881,
    },
    SearchPair {
        distance_sq: 1.6183726,
        index: 129963,
    },
]
     */
    /*
generated in Ok(186.003ms)
clustered in Ok(3.577731s) (6 iterations of kmeans (+ init kmeans++), no retries):
    init centroids in Ok(544.092ms)
    kmeans iteration 1 in Ok(496.816ms). clustering Ok(441.091ms) move centroids Ok(55.725ms)
    final wcss in Ok(50.293ms)
    497 * 6 + 50 + 544 = 3576, so no other major time losses
flat search in Ok(8.046ms)
clustered search by indices in Ok(13.192ms)
apply clusters in Ok(80.999ms)
flat search after clustering in Ok(8.751ms)
clustered search in Ok(2.156ms)
     */

    // todo: is there any better metric than wcss? (looks like faiss uses wcss for deciding best try?)
    // todo: what should be the amount of kmeans++ retries?
    // todo: test on actual datasets, and not randomly generated
    // todo: bench different cases: more vectors with less dimensions, less vectors with more dimensions
    // todo: looks like my clustering has kinda bad recall?

    // todo: compare speed with faiss
    // todo: try to use gpu?
    // todo: implement quantized ivf
}
