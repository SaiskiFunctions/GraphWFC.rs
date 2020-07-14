use crate::graph::{Graph, VertexIndex, Labels, Frequencies};
use std::cmp::Ordering;
use crate::utils::{hash_map, hash_set};

#[derive(Debug)]
pub struct Observe {
    entropy: f32,
    index: VertexIndex
}

impl Observe {
    pub fn new(index: &VertexIndex, labels: &Labels, frequencies: &Frequencies) -> Observe {
        Observe { entropy: Observe::calculate_entropy(labels, frequencies), index: *index }
    }

    fn constrain(&self, graph: &Graph) {
        // code to do collapse or propagate

    }

    fn calculate_entropy(labels: &Labels, frequencies: &Frequencies) -> f32 {
        let label_frequencies =  labels.iter().map(|label| frequencies.get(label).unwrap());
        let total: i32 = label_frequencies.clone().sum();
        - label_frequencies.map(|frequency| {
            let P = *frequency as f32 / total as f32;
            P * P.log2()
        }).sum::<f32>()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BinaryHeap;

    #[test]
    fn test_calculate_entropy_one() {
        let test_labels = hash_set(&[1]);
        let test_frequencies = hash_map(&[(1, 200)]);
        let test_observe = Observe::new(&0, &test_labels, &test_frequencies);

        assert_eq!(test_observe.entropy, 0.0);
    }

    #[test]
    fn test_calculate_entropy_small() {
        let test_labels = hash_set(&[0, 1, 2]);
        let test_frequencies = hash_map(&[(0, 2), (1, 1), (2, 1)]);
        let test_observe = Observe::new(&0, &test_labels, &test_frequencies);

        assert_eq!(test_observe.entropy, 1.5);
    }

    /*
    #[test]
    fn test_calculate_entropy_multiple() {
        [
            (hash_set(&[0, 1, 2, 3]), hash_map(&[(0, 4), (1, 6), (2, 1), (3, 6)]), 1.7921953)
        ].iter().for_each(|(labels, frequencies, expected)| {
            assert_eq!(calculate_entropy(labels.clone(), frequencies.clone()), *expected);
        });
    }
    */

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

        assert!(observe_a == observe_b);
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
}
