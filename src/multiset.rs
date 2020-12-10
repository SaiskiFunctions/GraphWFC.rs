use nalgebra::allocator::Allocator;
use nalgebra::{ClosedAdd, DefaultAllocator, Dim, DimName, Scalar, SimdPartialOrd, VectorN};
use num_traits::{One, Zero};
use rand::distributions::uniform::SampleUniform;
use rand::prelude::*;
use std::ops::{AddAssign, IndexMut};
use std::slice::Iter;


pub trait Multiset
where
    Self: Clone + PartialEq + IndexMut<usize, Output=<Self as Multiset>::Item>
{
    type Item: Zero + One + Copy + AddAssign + PartialOrd;

    fn from_iter_u<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self::Item>;

    fn from_row_slice_u(slice: &[Self::Item]) -> Self;

    fn empty(size: usize) -> Self;

    fn iter_m(&self) -> Iter<Self::Item>;

    fn num_elems(&self) -> usize;

    fn contains(&self, elem: usize) -> bool;

    fn union(&self, other: &Self) -> Self;

    fn intersection(&self, other: &Self) -> Self;

    fn is_subset(&self, other: &Self) -> bool;

    fn is_subset2(&self, other: &Self) -> bool;

    fn is_subset3(&self, other: &Self) -> bool;

    fn is_singleton(&self) -> bool;

    fn is_empty_m(&self) -> bool;

    fn get_non_zero(&self) -> Option<usize>;

    fn entropy(&self) -> f64;

    fn choose(&mut self, rng: &mut StdRng);

    fn add_assign_m(&mut self, other: &Self);
}

impl<N, D> Multiset for VectorN<N, D>
where
    f64: From<N>,
    N: Scalar + Zero + One + Copy + SimdPartialOrd + PartialOrd + ClosedAdd + SampleUniform,
    D: Dim + DimName,
    DefaultAllocator: Allocator<N, D>,
{
    type Item = N;

    fn from_iter_u<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Self::Item>,
    {
        let mut it = iter.into_iter();
        Self::zeros().map(|n| match it.next() {
            Some(v) => v,
            None => n,
        })
    }

    fn from_row_slice_u(slice: &[Self::Item]) -> Self {
        Self::from_iter_u(slice.iter().copied())
    }

    // size unneeded for statically allocated vectors
    #[allow(unused_variables)]
    fn empty(size: usize) -> Self {
        Self::zeros()
    }

    fn iter_m(&self) -> Iter<Self::Item> {
        self.as_slice().iter()
    }

    fn num_elems(&self) -> usize {
        self.len()
    }

    fn contains(&self, elem: usize) -> bool {
        match self.get(elem as usize) {
            Some(i) => i > &Zero::zero(),
            _ => false,
        }
    }

    fn union(&self, other: &Self) -> Self {
        self.sup(other)
    }

    fn intersection(&self, other: &Self) -> Self {
        self.inf(other)
    }

    fn is_subset(&self, other: &Self) -> bool {
        &(self.inf(other)) == self
    }

    fn is_subset2(&self, other: &Self) -> bool {
        self.zip_fold(other, true, |acc, a, b| acc && a <= b)
    }

    fn is_subset3(&self, other: &Self) -> bool {
        self.iter().zip(other).all(|(a, b)| a <= b)
    }

    fn is_singleton(&self) -> bool {
        self.fold(0, |acc, n| if n != Zero::zero() { acc + 1 } else { acc }) == 1
    }

    fn is_empty_m(&self) -> bool {
        self.sum() == Zero::zero()
    }

    fn get_non_zero(&self) -> Option<usize> {
        match self.argmax() {
            (_, v) if v == Zero::zero() => None,
            (i, _) => Some(i),
        }
    }

    fn entropy(&self) -> f64 {
        let total = f64::from(self.sum());
        -self.fold(0.0, |acc, frequency| {
            if frequency > Zero::zero() {
                let prob = f64::from(frequency) / total;
                acc + prob * prob.log2()
            } else {
                acc
            }
        })
    }

    //noinspection DuplicatedCode
    fn choose(&mut self, rng: &mut StdRng) {
        let total = self.sum();
        let choice = rng.gen_range::<_, N, N>(One::one(), total + One::one());
        let mut acc: N = Zero::zero();
        let mut chosen = false;
        self.iter_mut().for_each(|elem| {
            if chosen {
                *elem = Zero::zero()
            } else {
                acc += *elem;
                if acc < choice {
                    *elem = Zero::zero()
                } else {
                    chosen = true;
                }
            }
        });
    }

    fn add_assign_m(&mut self, other: &Self) {
        self.add_assign(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::U6;
    use Multiset;

    type MultisetVector = VectorN<u16, U6>;
    // type MultisetVector = Vec<u16>;

    #[test]
    fn test_from_iter_u() {
        // From iterators
        MultisetVector::from_iter_u(vec![1, 0, 1].into_iter());
        MultisetVector::from_iter_u(vec![1, 0, 1, 0].into_iter());
        MultisetVector::from_iter_u(vec![1, 0, 1, 0, 1, 1].into_iter());
        MultisetVector::from_iter_u(vec![1, 0, 1, 0, 1, 1, 0].into_iter());

        // From into iters
        MultisetVector::from_iter_u(vec![1, 0, 1]);
        MultisetVector::from_iter_u(vec![1, 0, 1, 0]);
        MultisetVector::from_iter_u(vec![1, 0, 1, 0, 1, 1]);
        MultisetVector::from_iter_u(vec![1, 0, 1, 0, 1, 1, 0]);
    }

    #[test]
    fn test_contains() {
        let mut a: MultisetVector = MultisetVector::from_row_slice_u(&[1, 0, 1, 0]);
        a.add_assign(MultisetVector::from_row_slice_u(&[1, 0, 1, 0]));
        assert!(a.contains(2));
        assert!(!a.contains(1));
        assert!(!a.contains(4))
    }

    #[test]
    fn test_union() {
        let a = MultisetVector::from_row_slice_u(&[1, 0, 1, 0, 0, 0]);
        let b = MultisetVector::from_row_slice_u(&[0, 0, 1, 0, 0, 0]);
        assert_eq!(a, a.union(&b))
    }

    #[test]
    fn test_intersection() {
        let a = MultisetVector::from_row_slice_u(&[1, 0, 1, 0, 0, 0]);
        let b = MultisetVector::from_row_slice_u(&[0, 0, 1, 0, 0, 0]);
        assert_eq!(b, a.intersection(&b))
    }

    #[test]
    fn test_is_subset() {
        let a = MultisetVector::from_row_slice_u(&[1, 0, 1, 1, 0, 0]);
        let b = MultisetVector::from_row_slice_u(&[0, 0, 1, 1, 0, 0]);
        assert!(b.is_subset(&a));
        assert!(!a.is_subset(&b));

        let c = MultisetVector::from_row_slice_u(&[2, 3, 5]);
        let d = MultisetVector::from_row_slice_u(&[2, 3, 1]);
        let e = MultisetVector::from_row_slice_u(&[4, 3, 5]);

        assert!(c.is_subset(&e));
        assert!(!c.is_subset(&d));
    }

    #[test]
    fn test_is_singleton() {
        let a = MultisetVector::from_row_slice_u(&[1, 0, 1, 1, 0, 0]);
        let b = MultisetVector::from_row_slice_u(&[0, 0, 1, 0, 0, 0]);
        assert!(b.is_singleton());
        assert!(!a.is_singleton())
    }

    #[test]
    fn test_is_empty() {
        let a = MultisetVector::from_row_slice_u(&[0, 0, 0, 0, 0, 0]);
        let b = MultisetVector::from_row_slice_u(&[1, 1, 0, 0, 0, 0]);
        assert!(a.is_empty_m());
        assert!(!b.is_empty_m())
    }

    #[test]
    fn test_get_non_zero() {
        let a = MultisetVector::from_row_slice_u(&[0, 0, 3, 0, 0, 6]);
        let b = MultisetVector::from_row_slice_u(&[0, 0, 0]);
        let c = MultisetVector::from_row_slice_u(&[4, 0]);
        assert_eq!(a.get_non_zero(), Some(5));
        assert_eq!(b.get_non_zero(), None);
        assert_eq!(c.get_non_zero(), Some(0))
    }

    #[test]
    fn test_entropy_zero2() {
        let a: &MultisetVector = &MultisetVector::from_row_slice_u(&[200, 0, 0, 0, 0, 0]);
        assert_eq!(a.entropy(), 0.0)
    }

    #[test]
    fn test_entropy_small2() {
        let a: &MultisetVector = &MultisetVector::from_row_slice_u(&[2, 1, 1, 0, 0, 0]);
        assert_eq!(a.entropy(), 1.5)
    }

    #[test]
    fn test_entropy_multiple2() {
        let a: &MultisetVector = &MultisetVector::from_row_slice_u(&[4, 6, 1, 6, 0, 0]);
        let entropy = a.entropy();
        let lt = 1.79219;
        let gt = 1.79220;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_entropy_zero_freq2() {
        let a: &MultisetVector = &MultisetVector::from_row_slice_u(&[4, 6, 0, 6, 0, 0]);
        let entropy = a.entropy();
        let lt = 1.56127;
        let gt = 1.56128;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_choose() {
        let a: &mut MultisetVector = &mut MultisetVector::from_row_slice_u(&[2, 1, 3, 4, 0, 0]);
        let test_rng1 = &mut StdRng::seed_from_u64(1);
        let result1: MultisetVector = MultisetVector::from_row_slice_u(&[0, 0, 3, 0, 0, 0]);
        a.choose(test_rng1);
        assert_eq!(*a, result1);

        let b: &mut MultisetVector = &mut MultisetVector::from_row_slice_u(&[2, 1, 3, 4, 0, 0]);
        let test_rng2 = &mut StdRng::seed_from_u64(10);
        let result2: MultisetVector = MultisetVector::from_row_slice_u(&[2, 0, 0, 0, 0, 0]);
        b.choose(test_rng2);
        assert_eq!(*b, result2)
    }
}
