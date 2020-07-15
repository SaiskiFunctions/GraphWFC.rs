mod observe;
mod propagate;
mod utils;
mod graph;

use std::collections::{HashSet, BinaryHeap, HashMap};
use rand::prelude::*;
use crate::observe::Observe;
use crate::propagate::Propagate;
use crate::graph::{Rules, Graph, VertexIndex, VertexLabel, Frequencies, Labels};
use crate::utils::{hash_set, hash_map};

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

fn collapse_algorithm(rng: &mut StdRng, rules: &Rules, frequencies: &Frequencies, all_labels: &Labels, out_graph: Graph) -> Option<Graph> {
    let mut heap: BinaryHeap<Observe> = BinaryHeap::new();
    let mut gen_observe: HashSet<VertexIndex> = HashSet::new();
    let mut observed: HashSet<VertexIndex> = HashSet::new();
    let mut propagations: Vec<Propagate> = Vec::new();

    initialize(rng, frequencies, all_labels, &out_graph, &mut propagations, &mut observed, &mut heap);

    loop {
        if observed.len() == out_graph.vertices.len() || heap.is_empty() { return Some(out_graph) }
        if propagations.is_empty() {
            gen_observe.drain().for_each(|index| {  // algo: 4.2
                let labels = out_graph.vertices.get(index as usize).unwrap();
                heap.push(Observe::new(&index, labels, frequencies))
            });
            
            //heap.pop().unwrap()
        } else {
            // do propagate
        }
    }

    None
}

fn initialize(
    rng: &mut StdRng, 
    frequencies: &Frequencies, 
    all_labels: &Labels, 
    out_graph: &Graph, 
    propagations: &mut Vec<Propagate>,
    observed: &mut HashSet<VertexIndex>,
    heap: &mut BinaryHeap<Observe>
) {
    for (_index, labels) in out_graph.vertices.iter().enumerate() {
        let from_index = _index as i32;
        if labels.is_subset(all_labels) && labels != all_labels {
            out_graph.connections(&from_index).iter().for_each(|(to_index, direction)| {
                propagations.push(Propagate::new(from_index, *to_index, *direction))
            });

            if labels.len() == 1 {
                observed.insert(from_index);
                continue
            }
        }
        heap.push(Observe::new_fuzz(rng, &from_index, labels, frequencies))
    }
}
