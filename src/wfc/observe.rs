use std::cmp::Ordering;
use rand::prelude::*;
use crate::graph::graph::VertexIndex;
use std::fmt::Debug;


// Lower and upper bounds for use in generating slightly different
// entropy values for the initial set of Observe structs generated
// before running collapse.
static FUZZ_LB: f32 = 0.000001;
static FUZZ_UB: f32 = 0.0005;

#[derive(Debug)]
pub struct Observe {
    entropy: f32,
    pub index: VertexIndex
}

impl Observe {
    pub fn new(index: &VertexIndex, entropy: f32) -> Observe {
        Observe { entropy, index: *index }
    }

    pub fn new_fuzz(rng: &mut StdRng, index: &VertexIndex, entropy: f32) -> Observe {
        let fuzz = rng.gen_range(FUZZ_LB, FUZZ_UB);
        Observe { entropy: entropy + fuzz, index: *index }
    }
}

impl Ord for Observe {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.entropy.partial_cmp(&other.entropy).unwrap() {
            Ordering::Greater => Ordering::Less,
            Ordering::Less => Ordering::Greater,
            ordering => ordering
        }
    }
}

impl PartialOrd for Observe {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Observe {}

impl PartialEq for Observe {
    fn eq(&self, other: &Self) -> bool {
        self.entropy == other.entropy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;
    use crate::multiset::{Multiset, MultisetTrait};
    use nalgebra::U6;

    #[test]
    fn test_observe_cmp() {
        let observe_less_entropy = Observe {
            entropy: 2.0,
            index: 0
        };

        let observe_more_entropy = Observe {
            entropy: 4.0,
            index: 1
        };

        assert!(observe_less_entropy > observe_more_entropy);
    }

    #[test]
    fn test_observe_cmp_eq() {
        let observe_a = Observe {
            entropy: 2.0,
            index: 0
        };

        let observe_b = Observe {
            entropy: 2.0,
            index: 1
        };

        assert_eq!(observe_a, observe_b);
    }

    #[test]
    fn test_observe_heap() {
        let mut test_heap = BinaryHeap::new();

        test_heap.push(Observe {entropy: 4.1, index: 0});
        test_heap.push(Observe {entropy: 2.5, index: 0});
        test_heap.push(Observe {entropy: 3.7, index: 0});

        assert_eq!(test_heap.pop().unwrap().entropy, 2.5);
        assert_eq!(test_heap.pop().unwrap().entropy, 3.7);
        assert_eq!(test_heap.pop().unwrap().entropy, 4.1);
    }

    #[test]
    fn test_new_fuzz() {
        let ms: Multiset<U6> = Multiset::from_row_slice_u(&[2, 1, 1]);
        let mut rng = StdRng::seed_from_u64(10);
        let observe = Observe::new(&0, ms.entropy());
        let observe_fuzz = Observe::new_fuzz(&mut rng, &0, ms.entropy());

        assert!(observe.entropy < observe_fuzz.entropy);
    }
}
