mod collapse;
mod graph;
mod observe;
mod propagate;
mod utils;

use collapse::collapse;
use utils::{hash_set, hash_map};
use graph::{Graph, Labels, Edges};


fn main() {

    let input_edges = hash_map(&[
        (0, vec![(1, 2), (4, 1)]),
        (1, vec![(0, 3), (5, 1), (2, 2)]),
        (2, vec![(1, 3)]),
        (3, vec![(4, 2)]),
        (4, vec![(3, 3), (0, 0), (5, 2)]),
        (5, vec![(4, 3), (1, 0)])
    ]);

    let input_vertices: Vec<Labels> = vec![
        hash_set(&[0]),
        hash_set(&[1]),
        hash_set(&[1]),
        hash_set(&[0]),
        hash_set(&[0]),
        hash_set(&[1])
    ];

    let input_graph = Graph::new(input_vertices, input_edges);

    let output_edges: Edges = hash_map(&[
        (0, vec![(1, 2), (4, 1)]),
        (1, vec![(0, 3), (5, 1), (2, 2)]),
        (2, vec![(1, 3), (6, 1), (3, 2)]),
        (3, vec![(2, 3), (7, 1)]),
        (4, vec![(0, 0), (8, 1), (5, 2)]),
        (5, vec![(4, 3), (1, 0), (6, 2), (9, 1)]),
        (6, vec![(5, 3), (2, 0), (7, 2), (10, 1)]),
        (7, vec![(6, 3), (3, 0), (11, 1)]),
        (8, vec![(4, 0), (9, 2)]),
        (9, vec![(8, 3), (5, 0), (10, 2)]),
        (10, vec![(9, 3), (6, 0), (11, 2)]),
        (11, vec![(10, 3), (7, 0)])
    ]);

    let output_vertices: Vec<Labels> = vec![
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
        hash_set(&[0, 1]),
    ];

    let output_graph = Graph::new(output_vertices, output_edges);

    collapse(input_graph, output_graph, Some(134522), None);
}
