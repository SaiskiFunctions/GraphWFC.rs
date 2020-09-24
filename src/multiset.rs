use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, DimName, VectorN};
use rand::prelude::*;

pub type MultisetScalar = u32;
pub type Multiset<D> = VectorN<MultisetScalar, D>;

pub trait MultisetTrait<D: Dim + DimName>
where
    DefaultAllocator: Allocator<MultisetScalar, D>,
{
    fn from_iter_u<I>(iter: I) -> Multiset<D>
    where
        I: IntoIterator<Item = MultisetScalar>;

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

impl<D: Dim + DimName> MultisetTrait<D> for VectorN<MultisetScalar, D>
where
    DefaultAllocator: Allocator<MultisetScalar, D>,
{
    fn from_iter_u<I>(iter: I) -> Multiset<D>
    where
        I: IntoIterator<Item = MultisetScalar>,
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
