use hashbrown::HashMap;
use lazy_static::*;
use nalgebra::DVector;
use rand::prelude::*;
use std::sync::RwLock;

type MultisetScalar = u32;
type Multiset = DVector<MultisetScalar>;
type EntropyCache = HashMap<Multiset, f32>;

lazy_static! {
    static ref CACHE: RwLock<EntropyCache> = RwLock::new(HashMap::with_capacity(32));
}

trait MultisetTrait {
    fn contains(&self, elem: MultisetScalar) -> bool;

    fn union(&self, other: &Multiset) -> Multiset;

    fn intersection(&self, other: &Multiset) -> Multiset;

    fn subset(&self, other: &Multiset) -> bool;

    fn entropy(&self) -> f32;

    fn choose(&mut self, rng: &mut StdRng) -> &mut Multiset;
}

impl MultisetTrait for DVector<MultisetScalar> {
    fn contains(&self, elem: MultisetScalar) -> bool {
        if let Some(i) = self.get(elem as usize) {
            return i > &0
        }
        false
    }

    fn union(&self, other: &Multiset) -> Multiset {
        self.sup(other)
    }

    fn intersection(&self, other: &Multiset) -> Multiset {
        self.inf(other)
    }

    fn subset(&self, other: &Multiset) -> bool {
        &(self.intersection(other)) == self
    }

    fn entropy(&self) -> f32 {
        if let Some(result) = CACHE.read().unwrap().get(self) {
            return *result
        }
        let total = self.sum() as f32;
        let result = - self.fold(0.0, |acc, frequency| {
            let prob = frequency as f32 / total;
            acc + prob * prob.log2()
        });
        let mut write_cache = CACHE.write().unwrap();
        (*write_cache).insert(self.clone(), result);
        result.clone()
    }

    fn choose(&mut self, rng: &mut StdRng) -> &mut Multiset {
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
        self
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
    fn test_subset() {
        let a = Multiset::from_row_slice(&[1, 0, 1, 1, 0]);
        let b = Multiset::from_row_slice(&[0, 0, 1, 1, 0]);
        assert!(b.subset(&a));
        assert!(!a.subset(&b))
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
    fn test_choose() {
        let a: &mut Multiset = &mut Multiset::from_row_slice(&[2, 1, 3, 4]);

        let test_rng1 = &mut StdRng::seed_from_u64(1);
        let result1 = &mut Multiset::from_row_slice(&[0, 0, 3, 0]);
        assert_eq!(a.clone().choose(test_rng1), result1);

        let test_rng2 = &mut StdRng::seed_from_u64(10);
        let result2 = &mut Multiset::from_row_slice(&[2, 0, 0, 0]);
        assert_eq!(a.choose(test_rng2), result2)
    }
}
