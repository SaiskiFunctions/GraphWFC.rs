mod observe;
mod propagate;
mod utils;
mod graph;

use std::collections::{HashSet, BinaryHeap, HashMap};
use crate::observe::Observe;
use crate::propagate::Propagate;
use crate::graph::{Rules, Graph, VertexIndex, VertexLabel};


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

fn calculate_entropy(labels: HashSet<VertexLabel>, frequencies: HashMap<VertexLabel, i32>) -> f32 {
    let label_frequencies =  labels.iter().map(|label| frequencies.get(label).unwrap());
    let total: i32 = label_frequencies.clone().sum();
    - label_frequencies.map(|frequency| {
        let P = *frequency as f32 / total as f32;
        P * P.log2()
    }).sum::<f32>()
}

fn collapse_algorithm(rules: Rules, out_graph: Graph) -> Option<Graph> {
    let mut heap: BinaryHeap<Observe> = BinaryHeap::new();
    let mut gen_observe: HashSet<VertexIndex> = HashSet::new();
    let mut observed: HashSet<VertexIndex> = HashSet::new();
    let mut propagations: Vec<Propagate> = Vec::new();

    // for vertex in out_graph.vertices.iter().for_each(|vertex| {

    // });

    None

}
