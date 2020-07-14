use crate::graph::{Graph, VertexIndex};
use std::cmp::Ordering;

#[derive(Debug)]
pub struct Observe {
    entropy: f32,
    index: VertexIndex
}

impl Observe {
    pub fn new(entropy: f32, index: i32) -> Observe {
        Observe { entropy, index }
    }

    fn constrain(&self, graph: &Graph) {
        // code to do collapse or propagate

    }
}

impl Ord for Observe {
    fn cmp(&self, other: &Self) -> Ordering {
        self.entropy.partial_cmp(&other.entropy).unwrap()
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
