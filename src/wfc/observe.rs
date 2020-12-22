use crate::graph::graph::VertexIndex;
use std::cmp::Ordering;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct Observe {
    entropy: f64,
    pub index: VertexIndex,
}

impl Observe {
    pub fn new(index: VertexIndex, entropy: f64) -> Observe {
        Observe {
            entropy,
            index,
        }
    }
}

impl Ord for Observe {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self.entropy <= other.entropy, self.entropy >= other.entropy) {
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            _ => Ordering::Equal,
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

    #[test]
    fn test_observe_cmp() {
        let observe_less_entropy = Observe {
            entropy: 2.0,
            index: 0,
        };

        let observe_more_entropy = Observe {
            entropy: 4.0,
            index: 1,
        };

        assert!(observe_less_entropy > observe_more_entropy);
    }

    #[test]
    fn test_observe_cmp_eq() {
        let observe_a = Observe {
            entropy: 2.0,
            index: 0,
        };

        let observe_b = Observe {
            entropy: 2.0,
            index: 1,
        };

        assert_eq!(observe_a, observe_b);
    }

    #[test]
    fn test_observe_heap() {
        let mut test_heap = BinaryHeap::new();

        test_heap.push(Observe {
            entropy: 4.1,
            index: 0,
        });
        test_heap.push(Observe {
            entropy: 2.5,
            index: 0,
        });
        test_heap.push(Observe {
            entropy: 3.7,
            index: 0,
        });

        assert_eq!(test_heap.pop().unwrap().entropy, 2.5);
        assert_eq!(test_heap.pop().unwrap().entropy, 3.7);
        assert_eq!(test_heap.pop().unwrap().entropy, 4.1);
    }
}
