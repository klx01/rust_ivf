#![feature(test)]
#![feature(portable_simd)]

extern crate test;
#[path = "../src/vectors.rs"]
mod vectors;
use std::simd::f32x4;
use std::simd::f32x16;
use std::simd::StdFloat;

// run with
// cargo +nightly bench

/*
in the current implementation, calculating distance is the largest bottleneck          (largest bottleneck? narrowest?)
you need to run it for (vectors_count * clusters_count) times for each iteration of kmeans (clustering specifically)
and for each initialisation of kmeans++

for vectors_count = 200000, dimensions = 512, clusters_count nlist = 64,
using simple implementation, each clustering iteration takes 4.7-4.8 seconds
using simd x4 hand-unrolled x4 implementation, each clustering iteration takes 0.95 - 1 seconds
using simd x16 implementation with optimized rollup, each clustering iteration takes 0.44 seconds
 */

/*
simple implementation
for arm it generates vectorised and unrolled loop, and both arrays are iterated using post-indexed loads.
however the summation is not vectorised, it extracts scalars from vectors and sums them one by one
i've tried to make it generate completely vectorised loop, but was not able to without using simd by hand
for x86 it generates a simple scalar loop
 */
fn calc_distance_sq_simple(v1: &[f32], v2: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (index, &value1) in v1.iter().enumerate() {
        let value2 = unsafe{ v2.get_unchecked(index) };
        let diff = value1 - value2;
        sum += diff * diff;
    }
    sum
}

/*
for arm, no change in codegen from the simple version
for x86, loop was unrolled
 */
fn calc_distance_sq_zip(v1: &[f32], v2: &[f32]) -> f32 {
    v1
        .iter()
        .zip(v2)
        .map(|(v1, v2)| v1 - v2)
        .map(|v| v * v)
        .sum()
}

/*
for arm, this version actually made it vectorise the adds, but it also became slower
not completely sure why yet.
one suspicious part is that it writes into memory and then reads it after loop was ended.
for x86 it vectorized and unrolled the loop. But it also does the same stores and loads
 */
fn calc_distance_sq_with_array(v1: &[f32], v2: &[f32]) -> f32 {
    let mut sum = [0.0, 0.0, 0.0, 0.0];
    for (index, &value1) in v1.iter().enumerate() {
        let value2 = unsafe{ v2.get_unchecked(index) };
        let diff = value1 - value2;
        sum[index % 4] += diff * diff;
    }
    sum.iter().sum()
}

/*
using simd by hand.
for both arm and x86, whole loop is vectorized, but not unrolled.
simd x4 is 4-5 times faster than the simple version.
for arm, besides my simd loop and a simple loop it also generates the half-vectorised loop from the simple version.
i guess it does not see that the last loop only has <4 iterations?
didn't check if it does that for x86
also for arm it now no longer uses post-indexing loads, it uses the basic loads and issues 2 additional adds to increment address
 */
fn calc_distance_sq_simd_x4(v1: &[f32], v2: &[f32]) -> f32 {
    let mut sum = f32x4::splat(0.0);
    let split = v1.len() - (v1.len() % 4);
    let mut start = 0;
    while start < split {
        let end = start + 4;
        let val1 = f32x4::from_slice(unsafe { v1.get_unchecked(start..end) });
        let val2 = f32x4::from_slice(unsafe { v2.get_unchecked(start..end) });
        let diff = val1 - val2;
        sum += diff * diff;
        start = end;
    }
    let mut sum = sum.as_array().iter().sum();
    for index in split..v1.len() {
        let value1 = unsafe{ v1.get_unchecked(index) };
        let value2 = unsafe{ v2.get_unchecked(index) };
        let diff = value1 - value2;
        sum += diff * diff;
    }
    sum
}

/*
using simd x16 instead of x4, results in an unrolled loop of 4 simd x4 operations.
2 times faster than simd x4 for large sizes, a bit slower for small sizes
for sizes < 16 it's slower than simple version, can be fixed by adding a condition
sum.as_array().iter().sum() is done in a scalar way
 */
fn calc_distance_sq_simd_x16(v1: &[f32], v2: &[f32]) -> f32 {
    let mut sum = f32x16::splat(0.0);
    let split = v1.len() - (v1.len() % 16);
    let mut start = 0;
    while start < split {
        let end = start + 16;
        let val1 = f32x16::from_slice(unsafe { v1.get_unchecked(start..end) });
        let val2 = f32x16::from_slice(unsafe { v2.get_unchecked(start..end) });
        let diff = val1 - val2;
        sum += diff * diff;
        start = end;
    }
    let mut sum = sum.as_array().iter().sum();
    for index in split..v1.len() {
        let value1 = unsafe{ v1.get_unchecked(index) };
        let value2 = unsafe{ v2.get_unchecked(index) };
        let diff = value1 - value2;
        sum += diff * diff;
    }
    sum
}

/*
use vectors for the final sum. around 25% times faster than previous version
 */
fn calc_distance_sq_simd_x16_v2(v1: &[f32], v2: &[f32]) -> f32 {
    let mut sum = f32x16::splat(0.0);
    let split = v1.len() - (v1.len() % 16);
    let mut start = 0;
    while start < split {
        let end = start + 16;
        let val1 = f32x16::from_slice(unsafe { v1.get_unchecked(start..end) });
        let val2 = f32x16::from_slice(unsafe { v2.get_unchecked(start..end) });
        let diff = val1 - val2;
        sum += diff * diff;
        start = end;
    }
    let sum = sum.as_array();
    let sum = f32x4::from_slice(unsafe { sum.get_unchecked(0..4) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(4..8) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(8..12) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(12..16) });
    let mut sum = sum.as_array().iter().sum();
    for index in split..v1.len() {
        let value1 = unsafe{ v1.get_unchecked(index) };
        let value2 = unsafe{ v2.get_unchecked(index) };
        let diff = value1 - value2;
        sum += diff * diff;
    }
    sum
}

/*
this stops emitting a dead branch.
also seems to have a slightly better performance for small sizes?
 */
fn calc_distance_sq_simd_x16_v3(mut v1: &[f32], mut v2: &[f32]) -> f32 {
    let mut sum = f32x16::splat(0.0);
    let iterations = v1.len() / 16;
    for _ in 0..iterations {
        let val1 = f32x16::from_slice(unsafe { v1.get_unchecked(..16) });
        v1 = unsafe { v1.get_unchecked(16..) };
        let val2 = f32x16::from_slice(unsafe { v2.get_unchecked(..16) });
        v2 = unsafe { v2.get_unchecked(16..) };
        let diff = val1 - val2;
        sum += diff * diff;
    }
    let sum = sum.as_array();
    let sum = f32x4::from_slice(unsafe { sum.get_unchecked(0..4) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(4..8) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(8..12) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(12..16) });
    let mut sum = sum.as_array().iter().sum();
    for (index, value1) in v1.iter().enumerate() {
        let value2 = unsafe{ v2.get_unchecked(index) };
        let diff = value1 - value2;
        sum += diff * diff;
    }
    sum
}

/*
using fused multiply-add
for aarch64 i see that it emits fmla instead of fmul fadd, but there seems to be no change in performance
there is no change in performance for 512, but 527 and 271 seem to be faster somehow? why? 
for x64 the generated assembly looks insane
both of these were only checked in godbolt, so maybe check if actual generated assembly is the same
 */
fn calc_distance_sq_simd_x16_fma(mut v1: &[f32], mut v2: &[f32]) -> f32 {
    let mut sum = f32x16::splat(0.0);
    let iterations = v1.len() / 16;
    for _ in 0..iterations {
        let val1 = f32x16::from_slice(unsafe { v1.get_unchecked(..16) });
        v1 = unsafe { v1.get_unchecked(16..) };
        let val2 = f32x16::from_slice(unsafe { v2.get_unchecked(..16) });
        v2 = unsafe { v2.get_unchecked(16..) };
        let diff = val1 - val2;
        sum = diff.mul_add(diff, sum);
    }
    let sum = sum.as_array();
    let sum = f32x4::from_slice(unsafe { sum.get_unchecked(0..4) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(4..8) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(8..12) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(12..16) });
    let mut sum = sum.as_array().iter().sum();
    for (index, value1) in v1.iter().enumerate() {
        let value2 = unsafe{ v2.get_unchecked(index) };
        let diff = value1 - value2;
        sum += diff * diff;
    }
    sum
}

/*
try to improve more!
todo: things left to improve:
use post-indexed loads instead of simple load + 2 adds. Not sure if it's actually faster, but i want to try to test it.
skip summing zeroes for the small size path (can be done easily, but for some reason adds unnecessary instructions to the main path)
    (i'm talking about creating zeroed f32x16 and summing it when size is <16)
try to make fused multiply add work properly?
i'm out of ideas how to make the compiler do what i want
    maybe just use intrinsics?
can the small path be optimised too? 527 is 1.5 times slower than 512
 */
fn calc_distance_sq_test(mut v1: &[f32], mut v2: &[f32]) -> f32 {
    let mut sum = f32x16::splat(0.0);
    let iterations = v1.len() / 16;
    for _ in 0..iterations {
        let val1 = f32x16::from_slice(unsafe { v1.get_unchecked(..16) });
        v1 = unsafe { v1.get_unchecked(16..) };
        let val2 = f32x16::from_slice(unsafe { v2.get_unchecked(..16) });
        v2 = unsafe { v2.get_unchecked(16..) };
        let diff = val1 - val2;
        sum += diff * diff;
    }
    let sum = sum.as_array();
    let sum = f32x4::from_slice(unsafe { sum.get_unchecked(0..4) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(4..8) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(8..12) })
        + f32x4::from_slice(unsafe { sum.get_unchecked(12..16) });
    let mut sum = sum.as_array().iter().sum();
    for (index, value1) in v1.iter().enumerate() {
        let value2 = unsafe{ v2.get_unchecked(index) };
        let diff = value1 - value2;
        sum += diff * diff;
    }
    sum
}

#[cfg(test)]
mod bench {
    use std::hint::black_box;
    use rand::rngs::SmallRng;
    use rand::{rng, SeedableRng};
    use test::Bencher;
    use crate::vectors::*;
    use super::*;

    #[bench]
    fn test_distance_implementations(b: &mut Bencher) {
        // not actually a bench, but a test
        let funcs = [
            calc_distance_sq_simple,
            calc_distance_sq_zip,
            calc_distance_sq_with_array,
            calc_distance_sq_simd_x4,
            calc_distance_sq_simd_x16,
            calc_distance_sq_simd_x16_v2,
            calc_distance_sq_simd_x16_v3,
            calc_distance_sq_simd_x16_fma,
            calc_distance_sq_test,
        ];
        let mut numbers = [0.0; 100];
        let mut numbers_expected = 0.0;
        for i in 0..100 {
            let casted = i as f32;
            numbers[i] = casted;
            numbers_expected += casted * casted;
        }
        for (index, func) in funcs.iter().enumerate() {
            assert_eq!(0.0, func(&[1.0, 1.0], &[1.0, 1.0]), "func number {index}");
            assert_eq!(2.0, func(&[1.0, 0.0], &[0.0, 1.0]), "func number {index}");
            assert_eq!(4.0, func(&[1.0, 0.0], &[-1.0, 0.0]), "func number {index}");
            assert_eq!(100.0, func(&[1.0; 100], &[0.0; 100]), "func number {index}");
            assert_eq!(400.0, func(&[2.0; 100], &[0.0; 100]), "func number {index}");
            assert_eq!(numbers_expected, func(&numbers, &[0.0; 100]), "func number {index}");
        }
    }

    fn gen_vectors(size: usize) -> VectorsList {
        let seed = 2465599376081375548;
        let mut rng = SmallRng::seed_from_u64(seed);
        //let mut rng = rng();
        let vectors_count = 2;
        let dimensions = black_box(size);
        let vectors = VectorsList::new_random(vectors_count, dimensions, &mut rng);
        vectors
    }

    fn bench_calc_distance(b: &mut Bencher, func: impl Fn(&[f32], &[f32]) -> f32, size: usize) {
        let vectors = gen_vectors(size);
        let v1 = vectors.get(0);
        let v2 = vectors.get(1);
        b.iter(|| func(v1, v2));
    }

    macro_rules! bench_calc_distance {
        ($func_name:ident) => {
            mod $func_name {
                use super::*;
                #[bench]
                fn bench_527(b: &mut Bencher) { bench_calc_distance(b, $func_name, 527); }
                #[bench]
                fn bench_512(b: &mut Bencher) { bench_calc_distance(b, $func_name, 512); }
                #[bench]
                fn bench_271(b: &mut Bencher) { bench_calc_distance(b, $func_name, 271); }
                #[bench]
                fn bench_047(b: &mut Bencher) { bench_calc_distance(b, $func_name, 47); }
                #[bench]
                fn bench_015(b: &mut Bencher) { bench_calc_distance(b, $func_name, 15); }
            }
        };
    }
    bench_calc_distance!(calc_distance_sq_simple);
    bench_calc_distance!(calc_distance_sq_zip);
    bench_calc_distance!(calc_distance_sq_with_array);
    bench_calc_distance!(calc_distance_sq_simd_x4);
    bench_calc_distance!(calc_distance_sq_simd_x16);
    bench_calc_distance!(calc_distance_sq_simd_x16_v2);
    bench_calc_distance!(calc_distance_sq_simd_x16_v3);
    bench_calc_distance!(calc_distance_sq_simd_x16_fma);
    bench_calc_distance!(calc_distance_sq_test);
}