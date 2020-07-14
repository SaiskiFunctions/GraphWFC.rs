mod utils;
use utils::{hash_map, hash_set};
mod graph;
use graph::*;

use std::collections::{HashSet, HashMap, BinaryHeap};
use std::iter::FromIterator;
use std::hash::Hash;
use std::cmp::{Eq, Ordering};
use std::clone::Clone;
use std::error::Error;


fn main() {
    // turn input into a graph
    // index number represents each vertex position
    // generate the rules
    // Set up uncollapsed output labels
    // generate output graph from input graph

    // let output_graph_vertices = vec![hash_set(&[0, 1, 2]); 4];
    //
    // let output_graph_edges: HashMap<vertexIndex, Vec<(vertexIndex, EdgeDirection)>> = hash_map(&[
    //     (0, vec![(1, 0), (3, 2)]),
    //     (1, vec![(0, 1), (2, 2)]),
    //     (2, vec![(3, 1), (1, 3)]),
    //     (3, vec![(0, 3), (2, 0)])
    // ]);
    //
    // let output_graph = Graph::new(
    //     output_graph_vertices,
    //     output_graph_edges
    // );
}

fn collapse_algorithm(rules: Rules, out_graph: Graph) -> Option<Graph> {
    let mut heap: BinaryHeap<> = BinaryHeap::new();
    let mut gen_observe = HashSet::new();
}

#[derive(Debug)]
struct ConstraintAction {
    entropy: f32,
    to: i32,
    from: i32,
    direction: u16
}

impl ConstraintAction {
    fn new(entropy: f32, from: i32, to: i32, direction: u16) -> ConstraintAction {
        ConstraintAction {
            entropy,
            from,
            to,
            direction
        }
    }

    fn constrain(&self, graph: &Graph) {
        // code to do collapse or propagate
        
    }
}

impl Ord for ConstraintAction {
    fn cmp(&self, other: &Self) -> Ordering {
        self.entropy.partial_cmp(&other.entropy).unwrap()
    }
}

impl Eq for ConstraintAction {}

impl PartialOrd for ConstraintAction {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ConstraintAction {
    fn eq(&self, other: &Self) -> bool {
        self.entropy == other.entropy
    }
}
