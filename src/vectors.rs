use rand::Rng;

#[derive(Debug)]
pub(crate) struct VectorsList {
    pub count: usize,
    pub dim: usize,
    data: Box<[f32]>,
}
impl VectorsList {
    pub fn new(count: usize, dim: usize) -> Self {
        let size = count * dim;
        let data = vec![0.0; size].into_boxed_slice();
        Self::from_parts(count, dim, data)
    }
    #[allow(unused)]
    pub fn new_random_norm(count: usize, dim: usize, rng: &mut impl Rng) -> Self {
        // todo: potentially optimise
        let mut vectors = Self::new(count, dim);
        for vector in vectors.iter_mut() {
            fill_random_normalized_vector(vector, rng)
        }
        vectors
    }
    #[allow(unused)]
    pub fn new_random(count: usize, dim: usize, rng: &mut impl Rng) -> Self {
        // todo: potentially optimise
        let mut vectors = Self::new(count, dim);
        for vector in vectors.iter_mut() {
            fill_random_vector(vector, rng)
        }
        vectors
    }
    pub fn from_vectors_by_indices(vectors: &VectorsList, indices: &[usize]) -> Self {
        // todo: potentially optimise
        let count = indices.len();
        let dim = vectors.dim;
        let size = count * dim;
        let mut data = Vec::with_capacity(size);
        for &index in indices {
            let vector = vectors.get(index);
            data.extend_from_slice(vector);
        }
        let data = data.into_boxed_slice();
        Self::from_parts(count, dim, data)
    }
    #[allow(unused)]
    pub fn from_arrays<const N: usize>(from_data: &[[f32; N]]) -> Self {
        let count = from_data.len();
        let dim = N;
        let data = from_data.as_flattened().to_vec().into_boxed_slice();
        Self::from_parts(count, dim, data)
    }
    fn from_parts(count: usize, dim: usize, data: Box<[f32]>) -> Self {
        debug_assert!(count > 0);
        debug_assert!(dim > 0);
        debug_assert!(data.len() == count * dim);
        Self{count, dim, data}
    }
    pub fn get(&self, index: usize) -> &[f32] {
        // todo: potentially optimise
        let start = index * self.dim;
        let end = start + self.dim;
        &self.data[start..end]
    }
    pub fn iter(&self) -> impl Iterator<Item=&[f32]> + use<'_> {
        // todo: potentially optimise
        self.data.chunks(self.dim)
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item=&mut [f32]> + use<'_> {
        // todo: potentially optimise
        self.data.chunks_mut(self.dim)
    }
    pub fn as_vec_of_slices(&self) -> Vec<&[f32]> {
        self.iter().collect()
    }
    pub fn swap(&mut self, index1: usize, index2: usize) {
        debug_assert!(index1 < self.count);
        debug_assert!(index2 < self.count);
        debug_assert!(index1 != index2);
        let (index1, index2) = if index1 > index2 {
            (index2, index1)
        } else {
            (index1, index2)
        };
        let split_at = index2 * self.dim;
        let (part1, part2) = self.data.split_at_mut(split_at);
        let start_at = index1 * self.dim;
        let vec1 = &mut part1[start_at..(start_at + self.dim)];
        let vec2 = &mut part2[..self.dim];
        vec1.swap_with_slice(vec2);
    }
    pub fn get_cluster(&self, start_vector_index: usize, length: usize) -> impl Iterator<Item=&[f32]> + use<'_> {
        // todo: potentially optimise
        let start = start_vector_index * self.dim;
        let end = start + (length * self.dim);
        self.data[start..end].chunks(self.dim)
    }
}

fn fill_random_vector(v: &mut [f32], rng: &mut impl Rng) {
    for value in v.iter_mut() {
        *value = rng.random_range(-1.0..=1.0);
    }
}

fn fill_random_normalized_vector(v: &mut [f32], rng: &mut impl Rng) {
    let mut sum = 0.0;
    for value in v.iter_mut() {
        *value = rng.random_range(-1.0..=1.0);
        sum += *value * *value;
    }
    let length = sum.sqrt();
    for value in v.iter_mut() {
        *value /= length
    }
}

pub(crate) fn is_same_vector(v1: &[f32], v2: &[f32]) -> bool {
    calc_distance_sq(v1, v2) < 0.0001
}

pub(crate) fn calc_distance_sq(mut v1: &[f32], mut v2: &[f32]) -> f32 {
    use std::simd::f32x16;
    use std::simd::f32x4;
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
pub(crate) fn float_equal(v1: f32, v2: f32) -> bool {
    (v1 - v2).abs() < 0.0001
}

#[cfg(test)]
pub(crate) fn vector_length(v: &[f32]) -> f32 {
    let mut sum = 0.0;
    for value in v {
        sum += value * value;
    }
    sum.sqrt()
}

#[cfg(test)]
mod test {
    use rand::rng;
    use super::*;
    #[test]
    fn test_calc_distance_sq() {
        assert_eq!(0.0, calc_distance_sq(&[1.0, 1.0], &[1.0, 1.0]));
        assert_eq!(2.0, calc_distance_sq(&[1.0, 0.0], &[0.0, 1.0]));
        assert_eq!(4.0, calc_distance_sq(&[1.0, 0.0], &[-1.0, 0.0]));
        assert_eq!(100.0, calc_distance_sq(&[1.0; 100], &[0.0; 100]));
        assert_eq!(400.0, calc_distance_sq(&[2.0; 100], &[0.0; 100]));
        let mut numbers = [0.0; 100];
        let mut numbers_expected = 0.0;
        for i in 0..100 {
            let casted = i as f32;
            numbers[i] = casted;
            numbers_expected += casted * casted;
        }
        assert_eq!(numbers_expected, calc_distance_sq(&numbers, &[0.0; 100]));
    }

    #[test]
    fn test_is_same_vector() {
        assert!(is_same_vector(&[1.0, 1.0], &[1.0, 1.0]));
        assert!(!is_same_vector(&[1.0, 0.0], &[0.0, 1.0]));
    }

    #[test]
    fn test_vector_length() {
        assert!(float_equal(0.0, vector_length(&[0.0, 0.0])));
        assert!(float_equal(1.0, vector_length(&[1.0, 0.0])));
        assert!(float_equal(2.0f32.sqrt(), vector_length(&[1.0, 1.0])));
        assert!(float_equal(2.0, vector_length(&[2.0, 0.0])));
    }

    #[test]
    fn test_fill_vector() {
        let mut vec = [0.0; 100];
        fill_random_normalized_vector(vec.as_mut_slice(), &mut rng());
        assert!(float_equal(1.0, vector_length(&vec)));
    }

    #[test]
    fn test_vectors_list() {
        let orig = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-6.0, -6.0, -6.0],
            [7.0, 8.0, 9.0],
            [-2.0, -2.0, -2.0],
        ];
        let mut vectors = VectorsList::from_arrays(&orig);
        assert_eq!(5, vectors.count);
        assert_eq!(3, vectors.dim);
        assert_eq!(15, vectors.data.len());
        assert_eq!(orig[0], vectors.get(0));
        assert_eq!(orig[3], vectors.get(3));
        assert_eq!(orig, vectors.as_vec_of_slices().as_slice());

        vectors.swap(1, 3);
        let expected = [
            [1.0, 2.0, 3.0],
            [7.0, 8.0, 9.0],
            [-6.0, -6.0, -6.0],
            [4.0, 5.0, 6.0],
            [-2.0, -2.0, -2.0],
        ];
        assert_eq!(expected, vectors.as_vec_of_slices().as_slice());
        vectors.swap(1, 3);
        assert_eq!(orig, vectors.as_vec_of_slices().as_slice());
        vectors.swap(1, 2);
        let expected = [
            [1.0, 2.0, 3.0],
            [-6.0, -6.0, -6.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [-2.0, -2.0, -2.0],
        ];
        assert_eq!(expected, vectors.as_vec_of_slices().as_slice());
    }

    #[test]
    fn test_from_vectors_by_indices() {
        let orig = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-6.0, -6.0, -6.0],
            [7.0, 8.0, 9.0],
            [-2.0, -2.0, -2.0],
        ];
        let vectors = VectorsList::from_arrays(&orig);
        let sample = VectorsList::from_vectors_by_indices(&vectors, &[0, 2, 3]);
        assert_eq!(3, sample.count);
        assert_eq!(3, vectors.dim);
        assert_eq!(orig[0], sample.get(0));
        assert_eq!(orig[2], sample.get(1));
        assert_eq!(orig[3], sample.get(2));
    }

    #[test]
    #[should_panic]
    fn test_no_indices() {
        let orig = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [-6.0, -6.0, -6.0],
            [7.0, 8.0, 9.0],
            [-2.0, -2.0, -2.0],
        ];
        let vectors = VectorsList::from_arrays(&orig);
        VectorsList::from_vectors_by_indices(&vectors, &[]);
    }

    #[test]
    fn test_create_rand_vectors() {
        let vectors = VectorsList::new_random_norm(5, 10, &mut rng());
        assert_eq!(vectors.count, 5);
        assert_eq!(vectors.dim, 10);
        assert_eq!(vectors.data.len(), 50);
        assert!(float_equal(1.0, vector_length(vectors.get(0))));
    }

    #[test]
    #[should_panic]
    fn test_create_rand_vectors_zero_count() {
        VectorsList::new_random_norm(0, 10, &mut rng());
    }

    #[test]
    #[should_panic]
    fn test_create_rand_vectors_zero_dimensions() {
        VectorsList::new_random_norm(5, 0, &mut rng());
    }
}