use generic_array::{GenericArray, ArrayLength};
use num_traits::{Zero, One};
use rand::distributions::uniform::SampleUniform;
use rand::prelude::*;
use std::fmt::Debug;
use std::mem;
use std::slice;
use std::ops::AddAssign;

pub trait ElemScalar: Clone + PartialEq + Debug {
    #[inline(always)]
    /// Performance hack: Clone doesn't get inlined for Copy types in debug
    /// mode, so make it inline anyway.
    fn inlined_clone(&self) -> Self {
        self.clone()
    }
}

impl<T: Copy + PartialEq + Debug> ElemScalar for T {
    #[inline(always)]
    fn inlined_clone(&self) -> T {
        *self
    }
}

#[derive(Debug)]
struct Multiset<N: ElemScalar, E: ArrayLength<N>> {
    data: GenericArray<N, E>
}

impl<N: ElemScalar, E: ArrayLength<N>> Default for Multiset<N, E>
    where
        N: Default,
{
    fn default() -> Self {
        Multiset {
            data: Default::default()
        }
    }
}

impl<N: ElemScalar, E: ArrayLength<N>> Multiset<N, E>
{
    pub fn from_array(data: GenericArray<N, E>) -> Multiset<N, E> {
        Multiset { data }
    }

    #[inline]
    pub fn iter(&self) -> slice::Iter<N> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> slice::IterMut<N> {
        self.data.iter_mut()
    }
}

/// # Constructors
impl<N: ElemScalar, E: ArrayLength<N>> Multiset<N, E>
{
    pub unsafe fn new_uninitialized() -> Self {
        Multiset {
            data: mem::MaybeUninit::<GenericArray<N, E>>::uninit().assume_init()
        }
    }

    pub fn from_exact_iter<I>(iter: I) -> Option<Self>
        where
            I: IntoIterator<Item=N>
    {
        GenericArray::from_exact_iter(iter).map(|data| Self::from_array(data))
    }

    pub fn repeat(elem: N) -> Self {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..E::USIZE {
            unsafe { *res.data.get_unchecked_mut(i) = elem.inlined_clone() }
        }
        res
    }

    pub fn empty() -> Self
        where
            N: Zero
    {
        Self::repeat(N::zero())
    }

    pub fn from_iter<I>(iter: I) -> Self
        where
            N: Zero,
            I: IntoIterator<Item=N>
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        let mut it = iter.into_iter();

        for i in 0..E::USIZE {
            let elem = match it.next() {
                Some(v) => v,
                None => N::zero(),
            };
            unsafe { *res.data.get_unchecked_mut(i) = elem }
        }
        res
    }

    pub fn from_slice(slice: &[N]) -> Self {
        assert_eq!(slice.len(), E::USIZE);
        let mut res = unsafe { Multiset::new_uninitialized() };
        let mut iter = slice.iter();

        for i in 0..E::USIZE {
            unsafe {
                *res.data.get_unchecked_mut(i) = iter.next().unwrap().inlined_clone()
            }
        }
        res
    }
}

/// # Set functionality
impl<N: ElemScalar, E: ArrayLength<N>> Multiset<N, E>
    where
        N: PartialOrd + Zero
{
    #[inline]
    pub fn contains(&self, elem: usize) -> bool {
        unsafe { elem < E::USIZE && self.data.get_unchecked(elem) > &N::zero() }
    }

    #[inline]
    pub fn count_non_zero(&self) -> usize {
        self.iter().fold(0, |acc, e| if e != &N::zero() { acc + 1 } else { acc })
    }

    #[inline]
    pub fn count_zero(&self) -> usize {
        self.iter().fold(0, |acc, e| if e == &N::zero() { acc + 1 } else { acc })
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        self.zip_map(other, |e1, e2| if e1 < e2 { e1 } else { e2 })
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.iter().all(|e| e == &N::zero())
    }

    #[inline]
    pub fn is_singleton(&self) -> bool {
        self.count_non_zero() == 1
    }

    #[inline]
    pub fn is_subset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a <= b)
    }

    #[inline]
    pub fn is_superset(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a >= b)
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        self.zip_map(other, |e1, e2| if e1 > e2 { e1 } else { e2 })
    }

    #[inline]
    pub fn total_elements(&self) -> N {
        self.fold(N::zero(), |a, b| a + b)
    }

    #[inline]
    pub fn choose(&mut self, elem: usize) {
        for i in 0..E::USIZE {
            if i != elem {
                unsafe { *self.data.get_unchecked_mut(i) = N::zero() }
            }
        }
    }

    //noinspection DuplicatedCode
    #[inline]
    pub fn largest(&self) -> (usize, N) {
        let mut the_max = unsafe { self.data.get_unchecked(0) };
        let mut the_i = 0;

        for i in 1..E::USIZE {
            let val = unsafe { self.data.get_unchecked(i) };
            if val > the_max {
                the_max = val;
                the_i = i;
            }
        }
        (the_i, the_max.inlined_clone())
    }

    //noinspection DuplicatedCode
    #[inline]
    pub fn smallest(&self) -> (usize, N) {
        let mut the_min = unsafe { self.data.get_unchecked(0) };
        let mut the_i = 0;

        for i in 1..E::USIZE {
            let val = unsafe { self.data.get_unchecked(i) };
            if val < the_min {
                the_min = val;
                the_i = i;
            }
        }
        (the_i, the_min.inlined_clone())
    }
}

/// # Set functionality; further constraints 1
impl<N: ElemScalar, E: ArrayLength<N>> Multiset<N, E>
    where
        N: PartialOrd + Zero,
        f64: From<N>,
{
    #[inline]
    pub fn entropy(&self) -> f64 {
        let total = f64::from(self.total_elements());
        -self.fold(0.0, |acc, frequency| {
            if frequency > N::zero() {
                let prob = f64::from(frequency) / total;
                acc + prob * prob.log2()
            } else {
                acc
            }
        })
    }
}

/// # Set functionality; further constraints 2
impl<N: ElemScalar, E: ArrayLength<N>> Multiset<N, E>
    where
        N: AddAssign + Copy + One + PartialOrd + SampleUniform + Zero,
{
    #[inline]
    pub fn choose_random(&mut self, rng: &mut StdRng) {
        let choice = rng.gen_range(N::one(), self.total_elements() + N::one());
        let mut acc = N::zero();
        let mut chosen = false;
        self.iter_mut().for_each(|elem| {
            if chosen {
                *elem = N::zero()
            } else {
                acc += *elem;
                if acc < choice {
                    *elem = N::zero()
                } else {
                    chosen = true;
                }
            }
        })
    }
}

/// # Maps and folds
impl<N: ElemScalar, E: ArrayLength<N>> Multiset<N, E>
{
    #[inline]
    pub fn map<N2, F>(&self, mut f: F) -> Multiset<N2, E>
        where
            N2: ElemScalar,
            F: FnMut(N) -> N2,
            E: ArrayLength<N2>,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..E::USIZE {
            unsafe {
                let e = self.data.get_unchecked(i).inlined_clone();
                *res.data.get_unchecked_mut(i) = f(e)
            }
        }
        res
    }

    #[inline]
    pub fn fold<Acc, F>(&self, init: Acc, mut f: F) -> Acc
        where
            F: FnMut(Acc, N) -> Acc
    {
        let mut res = init;
        for i in 0..E::USIZE {
            unsafe {
                let e = self.data.get_unchecked(i).inlined_clone();
                res = f(res, e)
            }
        }
        res
    }

    #[inline]
    pub fn zip_map<N2, N3, F>(&self, other: &Multiset<N2, E>, mut f: F) -> Multiset<N3, E>
        where
            N2: ElemScalar,
            N3: ElemScalar,
            F: FnMut(N, N2) -> N3,
            E: ArrayLength<N2> + ArrayLength<N3>,
    {
        let mut res = unsafe { Multiset::new_uninitialized() };
        for i in 0..E::USIZE {
            unsafe {
                let e1 = self.data.get_unchecked(i).inlined_clone();
                let e2 = other.data.get_unchecked(i).inlined_clone();
                *res.data.get_unchecked_mut(i) = f(e1, e2)
            }
        }
        res
    }
}

impl<'a, N: 'a + ElemScalar, E: ArrayLength<N>> IntoIterator for &'a Multiset<N, E>
{
    type Item = &'a N;
    type IntoIter = slice::Iter<'a, N>;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}

impl<'a, N: 'a + ElemScalar, E: ArrayLength<N>> IntoIterator for &'a mut Multiset<N, E>
{
    type Item = &'a mut N;
    type IntoIter = slice::IterMut<'a, N>;

    fn into_iter(self) -> Self::IntoIter { self.iter_mut() }
}

impl<N: ElemScalar, E: ArrayLength<N>> PartialEq for Multiset<N, E>
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl<N: ElemScalar, E: ArrayLength<N>> AddAssign for Multiset<N, E>
    where
        N: AddAssign,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..E::USIZE {
            unsafe {
                let e = rhs.data.get_unchecked(i).inlined_clone();
                *self.data.get_unchecked_mut(i) += e;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use typenum::U4;

    type MS4<N> = Multiset<N, U4>;
    type MS4u8 = MS4<u8>;

    #[test]
    fn test_from_exact_iter() {
        MS4u8::from_exact_iter(vec![1, 2, 3, 4].into_iter());
    }

    #[test]
    fn test_repeat() {
        let result = MS4u8::repeat(3);
        let expected = MS4u8::from_iter(vec![3; 4].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_zeroes() {
        let result = MS4u8::empty();
        let expected = MS4u8::from_iter(vec![0; 4].into_iter());
        assert_eq!(result, expected)
    }

    #[test]
    fn test_contains() {
        let set = MS4u8::from_slice(&[1, 0, 1, 0]);
        assert!(set.contains(2));
        assert!(!set.contains(1));
        assert!(!set.contains(4))
    }

    #[test]
    fn test_union() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[0, 0, 3, 1]);
        let c = MS4u8::from_slice(&[2, 0, 4, 1]);
        assert_eq!(c, a.union(&b))
    }

    #[test]
    fn test_intersection() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[0, 0, 3, 1]);
        let c = MS4u8::from_slice(&[0, 0, 3, 0]);
        assert_eq!(c, a.intersection(&b))
    }

    #[test]
    fn test_is_subset() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[2, 0, 4, 1]);
        assert!(a.is_subset(&b));
        assert!(!b.is_subset(&a));

        let c = MS4u8::from_slice(&[1, 3, 4, 5]);
        assert!(!a.is_subset(&c));
        assert!(!c.is_subset(&a));
    }

    #[test]
    fn test_is_superset() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[2, 0, 4, 1]);
        assert!(!a.is_superset(&b));
        assert!(b.is_superset(&a));

        let c = MS4u8::from_slice(&[1, 3, 4, 5]);
        assert!(!a.is_superset(&c));
        assert!(!c.is_superset(&a));
    }

    #[test]
    fn test_is_singleton() {
        let a = MS4u8::from_slice(&[1, 0, 0, 0]);
        assert!(a.is_singleton());

        let b = MS4u8::from_slice(&[0, 0, 0, 5]);
        assert!(b.is_singleton());

        let c = MS4u8::from_slice(&[1, 0, 0, 5]);
        assert!(!c.is_singleton());

        let d = MS4u8::from_slice(&[0, 0, 0, 0]);
        assert!(!d.is_singleton());
    }

    #[test]
    fn test_is_empty() {
        let a = MS4u8::from_slice(&[2, 0, 4, 0]);
        let b = MS4u8::from_slice(&[0, 0, 0, 0]);
        assert!(!a.is_empty());
        assert!(b.is_empty());
    }

    #[test]
    fn test_entropy1() {
        let a = MS4u8::from_slice(&[200, 0, 0, 0]);
        let b = MS4u8::from_slice(&[2, 1, 1, 0]);
        assert_eq!(a.entropy(), 0.0);
        assert_eq!(b.entropy(), 1.5);
    }

    #[test]
    fn test_entropy2() {
        let a = MS4u8::from_slice(&[4, 6, 1, 6]);
        let entropy = a.entropy();
        let lt = 1.79219;
        let gt = 1.79220;
        assert!(lt < entropy && entropy < gt);

        let b = MS4u8::from_slice(&[4, 6, 0, 6]);
        let entropy = b.entropy();
        let lt = 1.56127;
        let gt = 1.56128;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_choose() {
        let mut set = MS4u8::from_slice(&[2, 1, 3, 4]);
        let expected = MS4u8::from_slice(&[0, 0, 3, 0]);
        set.choose(2);
        assert_eq!(set, expected)
    }

    #[test]
    fn test_choose_random() {
        let mut result1 = MS4u8::from_slice(&[2, 1, 3, 4]);
        let expected1 = MS4u8::from_slice(&[0, 0, 3, 0]);
        let test_rng1 = &mut StdRng::seed_from_u64(1);
        result1.choose_random(test_rng1);
        assert_eq!(result1, expected1);

        let mut result2 = MS4u8::from_slice(&[2, 1, 3, 4]);
        let expected2 = MS4u8::from_slice(&[2, 0, 0, 0]);
        let test_rng2 = &mut StdRng::seed_from_u64(10);
        result2.choose_random(test_rng2);
        assert_eq!(result2, expected2);
    }

    #[test]
    fn test_count_zero() {
        let set = MS4u8::from_slice(&[0, 0, 3, 0]);
        assert_eq!(set.count_zero(), 3)
    }

    #[test]
    fn test_count_non_zero() {
        let set = MS4u8::from_slice(&[0, 2, 3, 0]);
        assert_eq!(set.count_non_zero(), 2)
    }

    #[test]
    fn test_map() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let result: MS4<f32> = set.map(|e| e as f32 * 1.5);
        let expected: MS4<f32> = Multiset::from_slice(&[1.5, 7.5, 3.0, 12.0]);
        assert_eq!(result, expected)
    }

    #[test]
    fn test_largest() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let expected = (3, 8);
        assert_eq!(set.largest(), expected)
    }

    #[test]
    fn test_smallest() {
        let set = MS4u8::from_slice(&[1, 5, 2, 8]);
        let expected = (0, 1);
        assert_eq!(set.smallest(), expected)
    }
}
