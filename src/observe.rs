use crate::graph::{Graph, VertexIndex, Labels, Frequencies};
use std::cmp::Ordering;
use rand::prelude::*;
use crate::utils::{hash_map, hash_set};

static FUZZ_LB: f32 = 0.000001;
static FUZZ_UB: f32 = 0.0005;

#[derive(Debug)]
pub struct Observe {
    entropy: f32,
    index: VertexIndex
}

impl Observe {
    pub fn new(index: &VertexIndex, labels: &Labels, frequencies: &Frequencies) -> Observe {
        Observe { entropy: calculate_entropy(labels, frequencies), index: *index }
    }

    pub fn new_fuzz(rng: &mut StdRng, index: &VertexIndex, labels: &Labels, frequencies: &Frequencies) -> Observe {
        let entropy = calculate_entropy(labels, frequencies);
        Observe { entropy: entropy + rng.gen_range(FUZZ_LB, FUZZ_UB), index: *index }
    }

    fn constrain(&self, graph: &Graph) {
        // code to do collapse or propagate
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

impl Eq for Observe {}

impl PartialOrd for Observe {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Observe {
    fn eq(&self, other: &Self) -> bool {
        self.entropy == other.entropy
    }
}

fn calculate_entropy(labels: &Labels, frequencies: &Frequencies) -> f32 {
    let label_frequencies =  labels.iter().map(|label| frequencies.get(label).unwrap());
    let total: i32 = label_frequencies.clone().sum();
    - label_frequencies.fold(0.0, |mut acc, frequency| {
        let prob = *frequency as f32 / total as f32;
        acc + prob * prob.log2()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn test_calculate_entropy_one() {
        let test_labels = hash_set(&[1]);
        let test_frequencies = hash_map(&[(1, 200)]);
        let entropy = calculate_entropy(&test_labels, &test_frequencies);

        assert_eq!(entropy, 0.0);
    }

    #[test]
    fn test_calculate_entropy_small() {
        let test_labels = hash_set(&[0, 1, 2]);
        let test_frequencies = hash_map(&[(0, 2), (1, 1), (2, 1)]);
        let entropy = calculate_entropy(&test_labels, &test_frequencies);

        assert_eq!(entropy, 1.5);
    }

    #[test]
    fn test_calculate_entropy_multiple() {
        [
            (hash_set(&[0, 1, 2, 3]), hash_map(&[(0, 4), (1, 6), (2, 1), (3, 6)]), 1.79219, 1.79220)
        ].iter().for_each(|(labels, frequencies, lt, gt)| {
            let entropy = calculate_entropy(labels, frequencies);
            assert!(*lt < entropy && entropy < *gt);
        });
    }

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
        let test_labels = hash_set(&[0, 1, 2]);
        let test_frequencies = hash_map(&[(0, 2), (1, 1), (2, 1)]);

        let mut rng = StdRng::seed_from_u64(10);
        let observe = Observe::new(&0, &test_labels, &test_frequencies);
        let observe_fuzz = Observe::new_fuzz(&mut rng, &0, &test_labels, &test_frequencies);

        assert!(observe.entropy < observe_fuzz.entropy);
    }
}
