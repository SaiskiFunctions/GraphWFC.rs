use hashbrown::HashMap;
use lazy_static::*;
use nalgebra::DVector;
use rand::prelude::*;
use std::sync::RwLock;

type MultisetScalar = u32;
pub type Multiset = DVector<MultisetScalar>;
type EntropyCache = HashMap<Multiset, f32>;

lazy_static! {
    static ref CACHE: RwLock<EntropyCache> = RwLock::new(HashMap::with_capacity(32));
}

pub trait MultisetTrait {
    fn contains(&self, elem: MultisetScalar) -> bool;

    fn union(&self, other: &Multiset) -> Multiset;

    fn intersection(&self, other: &Multiset) -> Multiset;

    fn is_subset(&self, other: &Multiset) -> bool;

    fn empty(&self) -> bool;

    fn entropy(&self) -> f32;

    fn component_zero_not(&self, not: &[u32]) -> Multiset;

    fn choose(&mut self, rng: &mut StdRng);
}

impl MultisetTrait for DVector<MultisetScalar> {
    fn contains(&self, elem: MultisetScalar) -> bool {
        if let Some(i) = self.get(elem as usize) {
            return i > &0
        }
        false
    }

    #[inline]
    fn union(&self, other: &Multiset) -> Multiset {
        self.sup(other)
    }

    #[inline]
    fn intersection(&self, other: &Multiset) -> Multiset {
        self.inf(other)
    }

    fn is_subset(&self, other: &Multiset) -> bool {
        &(self.intersection(other)) == self
    }

    fn empty(&self) -> bool {
        self.sum() == 0
    }

    fn entropy(&self) -> f32 {
        let total = self.sum() as f32;
        - self.fold(0.0, |acc, frequency| {
            if frequency > 0 {
                let prob = frequency as f32 / total;
                acc + prob * prob.log2()
            } else { acc }
        })
    }

    fn component_zero_not(&self, not: &[u32]) -> Multiset {
        let iterator = (0..self.len()).map(|index| {
            if not.contains(&(index as u32)) { 1 } else { 0 }
        });
        let zero_not = Multiset::from_iterator(self.len(), iterator);
        self.component_mul(&zero_not)
    }

    fn choose(&mut self, rng: &mut StdRng) {
        let total = self.sum();
        let choice = rng.gen_range(1, total + 1);
        let mut acc = 0;
        let mut chosen = false;
        self.iter_mut().for_each(|elem| {
            if chosen { *elem = 0 }
            else {
                acc += *elem;
                if acc < choice { *elem = 0 }
                else { chosen = true; }
            }
        });
    }
}

pub struct EntCache {
    cache: EntropyCache
}

impl EntCache {
    pub fn new() -> EntCache {
        EntCache { cache: HashMap::new() }
    }

    pub fn entropy(&mut self, ms: &Multiset) -> f32 {
        if let Some(result) = self.cache.get(ms) {
            return *result
        }
        let total = ms.sum() as f32;
        let result = - ms.fold(0.0, |acc, frequency| {
            if frequency > 0 {
                let prob = frequency as f32 / total;
                acc + prob * prob.log2()
            } else { acc }
        });
        self.cache.insert(ms.clone(), result.clone());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains() {
        let a = Multiset::from_row_slice(&[1, 0, 1]);
        assert!(a.contains(2));
        assert!(!a.contains(1));
        assert!(!a.contains(4))
    }

    #[test]
    fn test_union() {
        let a = Multiset::from_row_slice(&[1, 0, 1]);
        let b = Multiset::from_row_slice(&[0, 0, 1]);
        assert_eq!(a, a.union(&b))
    }

    #[test]
    fn test_intersection() {
        let a = Multiset::from_row_slice(&[1, 0, 1]);
        let b = Multiset::from_row_slice(&[0, 0, 1]);
        assert_eq!(b, a.intersection(&b))
    }

    #[test]
    fn test_is_subset() {
        let a = Multiset::from_row_slice(&[1, 0, 1, 1, 0]);
        let b = Multiset::from_row_slice(&[0, 0, 1, 1, 0]);
        assert!(b.is_subset(&a));
        assert!(!a.is_subset(&b))
    }

    #[test]
    fn test_is_empty() {
        let a = Multiset::from_row_slice(&[0, 0]);
        let b = Multiset::from_row_slice(&[1, 1, 0]);
        assert!(a.empty());
        assert!(!b.empty())
    }

    #[test]
    fn test_entropy_zero() {
        let a: &Multiset = &Multiset::from_row_slice(&[200]);
        assert_eq!(a.entropy(), 0.0)
    }

    #[test]
    fn test_entropy_small() {
        let a: &Multiset = &Multiset::from_row_slice(&[2, 1, 1]);
        assert_eq!(a.entropy(), 1.5)
    }

    #[test]
    fn test_entropy_multiple() {
        let a: &Multiset = &Multiset::from_row_slice(&[4, 6, 1, 6]);
        let entropy = a.entropy();
        let lt = 1.79219;
        let gt = 1.79220;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_entropy_zero_freq() {
        let a: &Multiset = &Multiset::from_row_slice(&[4, 6, 0, 6]);
        let entropy = a.entropy();
        let lt = 1.56127;
        let gt = 1.56128;
        assert!(lt < entropy && entropy < gt);
    }

    #[test]
    fn test_component_zero_not() {
        let a: Multiset = Multiset::from_row_slice(&[4, 6, 0, 6]);
        let expected: Multiset = Multiset::from_row_slice(&[4, 0, 0, 6]);
        let result: Multiset = a.component_zero_not(&[0, 3]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_choose() {
        let a: &mut Multiset = &mut Multiset::from_row_slice(&[2, 1, 3, 4]);
        let test_rng1 = &mut StdRng::seed_from_u64(1);
        let result1 = Multiset::from_row_slice(&[0, 0, 3, 0]);
        a.choose(test_rng1);
        assert_eq!(*a, result1);

        let b: &mut Multiset = &mut Multiset::from_row_slice(&[2, 1, 3, 4]);
        let test_rng2 = &mut StdRng::seed_from_u64(10);
        let result2 = Multiset::from_row_slice(&[2, 0, 0, 0]);
        b.choose(test_rng2);
        assert_eq!(*b, result2)
    }
}
