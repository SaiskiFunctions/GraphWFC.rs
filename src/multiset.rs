use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, DimName, VectorN, Scalar, SimdPartialOrd, ClosedAdd};
use rand::prelude::*;
use num_traits::{Zero, One};
use rand::distributions::uniform::SampleUniform;
use std::iter::{FromIterator, Sum};

pub type MultisetScalar = u32;
pub type Multiset<D> = VectorN<MultisetScalar, D>;

pub trait MultisetTrait2 {
    type Item;

    fn from_iter_u2<I>(iter: I) -> Self
        where I: IntoIterator<Item=Self::Item>;

    fn from_row_slice_u2(slice: &[Self::Item]) -> Self;

    fn contains2(&self, elem: usize) -> bool;

    fn union2(&self, other: &Self) -> Self;

    fn intersection2(&self, other: &Self) -> Self;

    fn is_subset2(&self, other: &Self) -> bool;

    fn is_singleton2(&self) -> bool;

    fn is_empty2(&self) -> bool;

    fn get_non_zero2(&self) -> Option<usize>;

    fn entropy2(&self) -> f32;

    fn choose2(&mut self, rng: &mut StdRng);
}

pub trait MultisetTrait<D: Dim + DimName>
    where
        DefaultAllocator: Allocator<MultisetScalar, D>,
{
    fn from_iter_u<I>(iter: I) -> Multiset<D>
        where
            I: IntoIterator<Item=MultisetScalar>;

    fn from_row_slice_u(slice: &[MultisetScalar]) -> Multiset<D> {
        Multiset::from_iter_u(slice.iter().copied())
    }

    fn contains(&self, elem: usize) -> bool;

    fn union(&self, other: &Multiset<D>) -> Multiset<D>;

    fn intersection(&self, other: &Multiset<D>) -> Multiset<D>;

    fn is_subset(&self, other: &Multiset<D>) -> bool;

    fn is_singleton(&self) -> bool;

    // called empty because is_empty is defined on VectorN already
    fn empty(&self) -> bool;

    fn get_non_zero(&self) -> Option<usize>;

    fn entropy(&self) -> f32;

    fn choose(&mut self, rng: &mut StdRng);
}

impl<N, D> MultisetTrait2 for VectorN<N, D>
    where
        f32: From<N>,
        N: Scalar + Zero + One + Copy + SimdPartialOrd + PartialOrd + ClosedAdd + SampleUniform,
        D: Dim + DimName,
        DefaultAllocator: Allocator<N, D>,
{
    type Item = N;

    fn from_iter_u2<I>(iter: I) -> Self
        where
            I: IntoIterator<Item=Self::Item>,
    {
        let mut it = iter.into_iter();
        Self::zeros().map(|n| match it.next() {
            Some(v) => v,
            None => n,
        })
    }

    fn from_row_slice_u2(slice: &[Self::Item]) -> Self {
        Self::from_iter_u2(slice.iter().copied())
    }

    fn contains2(&self, elem: usize) -> bool {
        match self.get(elem as usize) {
            Some(i) => i > &Zero::zero(),
            _ => false,
        }
    }

    fn union2(&self, other: &Self) -> Self {
        self.sup(other)
    }

    fn intersection2(&self, other: &Self) -> Self {
        self.inf(other)
    }

    fn is_subset2(&self, other: &Self) -> bool {
        &(self.inf(other)) == self
    }

    fn is_singleton2(&self) -> bool {
        self.fold(0, |acc, n| if n != Zero::zero() { acc + 1 } else { acc }) == 1
    }

    fn is_empty2(&self) -> bool {
        self.sum() == Zero::zero()
    }

    fn get_non_zero2(&self) -> Option<usize> {
        match self.argmax() {
            (_, v) if v == Zero::zero() => None,
            (i, _) => Some(i),
        }
    }

    fn entropy2(&self) -> f32 {
        let total = f32::from(self.sum());
        -self.fold(0.0, |acc, frequency| {
            if frequency > Zero::zero() {
                let prob = f32::from(frequency) / total;
                acc + prob * prob.log2()
            } else {
                acc
            }
        })
    }

    fn choose2(&mut self, rng: &mut StdRng) {
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
}

impl<N> MultisetTrait2 for Vec<N>
    where
        f32: From<N>,
        N: Zero + One + Copy + PartialOrd + ClosedAdd + SampleUniform + Sum,
{
    type Item = N;

    fn from_iter_u2<I>(iter: I) -> Self where I: IntoIterator<Item=Self::Item> {
        Vec::from_iter(iter)
    }

    fn from_row_slice_u2(slice: &[Self::Item]) -> Self {
        Self::from_iter_u2(slice.iter().copied())
    }

    fn contains2(&self, elem: usize) -> bool {
        match self.get(elem as usize) {
            Some(i) => i > &Zero::zero(),
            _ => false,
        }
    }

    fn union2(&self, other: &Self) -> Self {
        self.iter().zip(other).map(|(a, b)| if a < b {*b} else {*a}).collect()
    }

    fn intersection2(&self, other: &Self) -> Self {
        self.iter().zip(other).map(|(a, b)| if a > b {*b} else {*a}).collect()
    }

    fn is_subset2(&self, other: &Self) -> bool {
        self.len() <= other.len() && self.iter().zip(other).all(|(a, b)| a <= b)
    }

    fn is_singleton2(&self) -> bool {
        self.iter().fold(0, |acc, &n| if n != Zero::zero() { acc + 1 } else { acc }) == 1
    }

    fn is_empty2(&self) -> bool {
        self.iter().copied().sum::<N>() == Zero::zero()
    }

    fn get_non_zero2(&self) -> Option<usize> {
        let mut max_: N = Zero::zero();
        let mut index: Option<usize> = None;
        for (i, &elem) in self.iter().enumerate() {
            if elem > max_ {
                max_ = elem;
                index = Some(i)
            }
        }
        index
    }

    fn entropy2(&self) -> f32 {
        let total = f32::from(self.iter().copied().sum::<N>());
        -self.iter().fold(0.0, |acc, &frequency| {
            if frequency > Zero::zero() {
                let prob = f32::from(frequency) / total;
                acc + prob * prob.log2()
            } else {
                acc
            }
        })
    }

    fn choose2(&mut self, rng: &mut StdRng) {
        let total: N = self.iter().copied().sum();
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
}

impl<D: Dim + DimName> MultisetTrait<D> for VectorN<MultisetScalar, D>
    where
        DefaultAllocator: Allocator<MultisetScalar, D>,
{
    fn from_iter_u<I>(iter: I) -> Multiset<D>
        where
            I: IntoIterator<Item=MultisetScalar>,
    {
        let mut it = iter.into_iter();
        Multiset::zeros().map(|n| match it.next() {
            Some(v) => v,
            None => n,
        })
    }

    fn contains(&self, elem: usize) -> bool {
        match self.get(elem as usize) {
            Some(i) => i > &0,
            _ => false,
        }
    }

    fn union(&self, other: &Multiset<D>) -> Multiset<D> {
        self.sup(other)
    }

    fn intersection(&self, other: &Multiset<D>) -> Multiset<D> {
        self.inf(other)
    }

    fn is_subset(&self, other: &Multiset<D>) -> bool {
        &(self.intersection(other)) == self
    }

    fn is_singleton(&self) -> bool {
        self.fold(0, |acc, n| if n != 0 { acc + 1 } else { acc }) == 1
    }

    fn empty(&self) -> bool {
        self.sum() == 0
    }

    fn get_non_zero(&self) -> Option<usize> {
        match self.argmax() {
            (_, 0) => None,
            (i, _) => Some(i),
        }
    }

    fn entropy(&self) -> f32 {
        let total = self.sum() as f32;
        -self.fold(0.0, |acc, frequency| {
            if frequency > 0 {
                let prob = frequency as f32 / total;
                acc + prob * prob.log2()
            } else {
                acc
            }
        })
    }

    fn choose(&mut self, rng: &mut StdRng) {
        let total = self.sum();
        let choice = rng.gen_range(1, total + 1);
        let mut acc = 0;
        let mut chosen = false;
        self.iter_mut().for_each(|elem| {
            if chosen {
                *elem = 0
            } else {
                acc += *elem;
                if acc < choice {
                    *elem = 0
                } else {
                    chosen = true;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::U6;
    use MultisetTrait;

    #[test]
    fn test_from_iter_u() {
        // From iterators
        Multiset::<U6>::from_iter_u(vec![1, 0, 1].into_iter());
        Multiset::<U6>::from_iter_u(vec![1, 0, 1, 0].into_iter());
        Multiset::<U6>::from_iter_u(vec![1, 0, 1, 0, 1, 1].into_iter());
        Multiset::<U6>::from_iter_u(vec![1, 0, 1, 0, 1, 1, 0].into_iter());

        // From into iters
        Multiset::<U6>::from_iter_u(vec![1, 0, 1]);
        Multiset::<U6>::from_iter_u(vec![1, 0, 1, 0]);
        Multiset::<U6>::from_iter_u(vec![1, 0, 1, 0, 1, 1]);
        Multiset::<U6>::from_iter_u(vec![1, 0, 1, 0, 1, 1, 0]);
    }

    #[test]
    fn test_contains() {
        let a = Multiset::<U6>::from_row_slice_u(&[1, 0, 1, 0]);
        assert!(a.contains(2));
        assert!(!a.contains(1));
        assert!(!a.contains(4))
    }

    #[test]
    fn test_union() {
        let a = Multiset::<U6>::from_row_slice_u(&[1, 0, 1, 0, 0, 0]);
        let b = Multiset::<U6>::from_row_slice_u(&[0, 0, 1, 0, 0, 0]);
        assert_eq!(a, a.union(&b))
    }

    #[test]
    fn test_intersection() {
        let a = Multiset::<U6>::from_row_slice_u(&[1, 0, 1, 0, 0, 0]);
        let b = Multiset::<U6>::from_row_slice_u(&[0, 0, 1, 0, 0, 0]);
        assert_eq!(b, a.intersection(&b))
    }

    #[test]
    fn test_is_subset() {
        let a = Multiset::<U6>::from_row_slice_u(&[1, 0, 1, 1, 0, 0]);
        let b = Multiset::<U6>::from_row_slice_u(&[0, 0, 1, 1, 0, 0]);
        assert!(b.is_subset(&a));
        assert!(!a.is_subset(&b))
    }

    #[test]
    fn test_is_singleton() {
        let a = Multiset::<U6>::from_row_slice_u(&[1, 0, 1, 1, 0, 0]);
        let b = Multiset::<U6>::from_row_slice_u(&[0, 0, 1, 0, 0, 0]);
        assert!(b.is_singleton());
        assert!(!a.is_singleton())
    }

    #[test]
    fn test_is_empty() {
        let a = Multiset::<U6>::from_row_slice_u(&[0, 0, 0, 0, 0, 0]);
        let b = Multiset::<U6>::from_row_slice_u(&[1, 1, 0, 0, 0, 0]);
        assert!(a.empty());
        assert!(!b.empty())
    }

    #[test]
    fn test_get_non_zero() {
        let a = Multiset::<U6>::from_row_slice_u(&[0, 0, 3, 0, 0, 6]);
        let b = Multiset::<U6>::from_row_slice_u(&[0, 0, 0]);
        let c = Multiset::<U6>::from_row_slice_u(&[4, 0]);
        assert_eq!(a.get_non_zero(), Some(5));
        assert_eq!(b.get_non_zero(), None);
        assert_eq!(c.get_non_zero(), Some(0))
    }

    #[test]
    fn test_entropy_zero() {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 0, 0, 0, 0, 0]);
        assert_eq!(a.entropy(), 0.0)
    }

    #[test]
    fn test_entropy_small() {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[2, 1, 1, 0, 0, 0]);
        assert_eq!(a.entropy(), 1.5)
    }

    #[test]
    fn test_entropy_multiple() {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[4, 6, 1, 6, 0, 0]);
        let entropy = a.entropy();
        let lt = 1.79219;
        let gt = 1.79220;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_entropy_zero_freq() {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[4, 6, 0, 6, 0, 0]);
        let entropy = a.entropy();
        let lt = 1.56127;
        let gt = 1.56128;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_choose() {
        let a: &mut Multiset<U6> = &mut Multiset::from_row_slice_u(&[2, 1, 3, 4, 0, 0]);
        let test_rng1 = &mut StdRng::seed_from_u64(1);
        let result1: Multiset<U6> = Multiset::from_row_slice_u(&[0, 0, 3, 0, 0, 0]);
        a.choose(test_rng1);
        assert_eq!(*a, result1);

        let b: &mut Multiset<U6> = &mut Multiset::from_row_slice_u(&[2, 1, 3, 4, 0, 0]);
        let test_rng2 = &mut StdRng::seed_from_u64(10);
        let result2: Multiset<U6> = Multiset::from_row_slice_u(&[2, 0, 0, 0, 0, 0]);
        b.choose(test_rng2);
        assert_eq!(*b, result2)
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use nalgebra::U6;
    use MultisetTrait2;

    type Multiset2 = VectorN<u16, U6>;
    // type Multiset2 = Vec<u16>;

    #[test]
    fn test_from_iter_u2() {
        // From iterators
        Multiset2::from_iter_u2(vec![1, 0, 1].into_iter());
        Multiset2::from_iter_u2(vec![1, 0, 1, 0].into_iter());
        Multiset2::from_iter_u2(vec![1, 0, 1, 0, 1, 1].into_iter());
        Multiset2::from_iter_u2(vec![1, 0, 1, 0, 1, 1, 0].into_iter());

        // From into iters
        Multiset2::from_iter_u2(vec![1, 0, 1]);
        Multiset2::from_iter_u2(vec![1, 0, 1, 0]);
        Multiset2::from_iter_u2(vec![1, 0, 1, 0, 1, 1]);
        Multiset2::from_iter_u2(vec![1, 0, 1, 0, 1, 1, 0]);
    }

    #[test]
    fn test_contains2() {
        let a = Multiset2::from_row_slice_u2(&[1, 0, 1, 0]);
        assert!(a.contains2(2));
        assert!(!a.contains2(1));
        assert!(!a.contains2(4))
    }

    #[test]
    fn test_union2() {
        let a = Multiset2::from_row_slice_u2(&[1, 0, 1, 0, 0, 0]);
        let b = Multiset2::from_row_slice_u2(&[0, 0, 1, 0, 0, 0]);
        assert_eq!(a, a.union2(&b))
    }

    #[test]
    fn test_intersection2() {
        let a = Multiset2::from_row_slice_u2(&[1, 0, 1, 0, 0, 0]);
        let b = Multiset2::from_row_slice_u2(&[0, 0, 1, 0, 0, 0]);
        assert_eq!(b, a.intersection2(&b))
    }

    #[test]
    fn test_is_subset2() {
        let a = Multiset2::from_row_slice_u2(&[1, 0, 1, 1, 0, 0]);
        let b = Multiset2::from_row_slice_u2(&[0, 0, 1, 1, 0, 0]);
        assert!(b.is_subset2(&a));
        assert!(!a.is_subset2(&b))
    }

    #[test]
    fn test_is_singleton2() {
        let a = Multiset2::from_row_slice_u2(&[1, 0, 1, 1, 0, 0]);
        let b = Multiset2::from_row_slice_u2(&[0, 0, 1, 0, 0, 0]);
        assert!(b.is_singleton2());
        assert!(!a.is_singleton2())
    }

    #[test]
    fn test_is_empty2() {
        let a = Multiset2::from_row_slice_u2(&[0, 0, 0, 0, 0, 0]);
        let b = Multiset2::from_row_slice_u2(&[1, 1, 0, 0, 0, 0]);
        assert!(a.is_empty2());
        assert!(!b.is_empty2())
    }

    #[test]
    fn test_get_non_zero2() {
        let a = Multiset2::from_row_slice_u2(&[0, 0, 3, 0, 0, 6]);
        let b = Multiset2::from_row_slice_u2(&[0, 0, 0]);
        let c = Multiset2::from_row_slice_u2(&[4, 0]);
        assert_eq!(a.get_non_zero2(), Some(5));
        assert_eq!(b.get_non_zero2(), None);
        assert_eq!(c.get_non_zero2(), Some(0))
    }

    #[test]
    fn test_entropy_zero2() {
        let a: &Multiset2 = &Multiset2::from_row_slice_u2(&[200, 0, 0, 0, 0, 0]);
        assert_eq!(a.entropy2(), 0.0)
    }

    #[test]
    fn test_entropy_small2() {
        let a: &Multiset2 = &Multiset2::from_row_slice_u2(&[2, 1, 1, 0, 0, 0]);
        assert_eq!(a.entropy2(), 1.5)
    }

    #[test]
    fn test_entropy_multiple2() {
        let a: &Multiset2 = &Multiset2::from_row_slice_u2(&[4, 6, 1, 6, 0, 0]);
        let entropy = a.entropy2();
        let lt = 1.79219;
        let gt = 1.79220;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_entropy_zero_freq2() {
        let a: &Multiset2 = &Multiset2::from_row_slice_u2(&[4, 6, 0, 6, 0, 0]);
        let entropy = a.entropy2();
        let lt = 1.56127;
        let gt = 1.56128;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_choose2() {
        let a: &mut Multiset2 = &mut Multiset2::from_row_slice_u2(&[2, 1, 3, 4, 0, 0]);
        let test_rng1 = &mut StdRng::seed_from_u64(1);
        let result1: Multiset2 = Multiset2::from_row_slice_u2(&[0, 0, 3, 0, 0, 0]);
        a.choose2(test_rng1);
        assert_eq!(*a, result1);

        let b: &mut Multiset2 = &mut Multiset2::from_row_slice_u2(&[2, 1, 3, 4, 0, 0]);
        let test_rng2 = &mut StdRng::seed_from_u64(10);
        let result2: Multiset2 = Multiset2::from_row_slice_u2(&[2, 0, 0, 0, 0, 0]);
        b.choose2(test_rng2);
        assert_eq!(*b, result2)
    }
}
