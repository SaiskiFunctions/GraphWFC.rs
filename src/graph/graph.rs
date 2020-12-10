use crate::multiset::Multiset;
use hashbrown::HashMap;
use num_traits::Zero;
use std::ops::Index;

pub type VertexIndex = u32; // each unique vertex in a graph
pub type EdgeDirection = u16; // the directional relationship between two vertices
pub type Edges = HashMap<VertexIndex, Vec<(VertexIndex, EdgeDirection)>>;

//                           vertex label (index of LabelFrequencies vector)
//                                            |
//                                            v
pub type Rules<S> = HashMap<(EdgeDirection, usize), S>;

#[derive(Debug, Clone)]
pub struct Graph<S: Multiset> {
    pub vertices: Vec<S>, // index of vec == vertex index
    pub edges: Edges,
    pub all_labels: S,
}

impl<S: Multiset> Graph<S> {
    pub fn new(vertices: Vec<S>, edges: Edges, all_labels: S) -> Graph<S> {
        Graph {
            vertices,
            edges,
            all_labels,
        }
    }

    /// Construct HashMap of rules for this graph.
    /// Rules connect a tuple of direction and vertex label to a set of labels.
    pub fn rules(&self) -> Rules<S> {
        self.edges
            .iter()
            .fold(HashMap::new(), |mut rules, (from_vertex_index, edges)| {
                edges.iter().for_each(|(to_vertex_index, direction)| {
                    self.vertices
                        .index(*from_vertex_index as usize)
                        .iter_m()
                        .enumerate()
                        .filter(|(_, &label)| label > Zero::zero())
                        .for_each(|(from_vertex_label, _)| {
                            let rules_key = (*direction, from_vertex_label);
                            let union_labels = self.vertices.index(*to_vertex_index as usize);
                            rules
                                .entry(rules_key)
                                .and_modify(|to_labels| to_labels.add_assign_m(union_labels))
                                // .and_modify(|to_labels| *to_labels = to_labels.union(union_labels))
                                .or_insert(union_labels.clone());
                        })
                });
                rules
            })
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;
    use crate::utils::hash_map;
    use nalgebra::{VectorN, U6};

    type MS = VectorN<u16, U6>;

    fn graph_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)]),
        ])
    }

    #[test]
    fn test_rules() {
        /*
        1b --- 2c
        |      |
        0a --- 3b

        North = 0, South = 1, East = 2, West = 3
        */

        let graph_vertices: Vec<MS> = vec![
            MS::from_row_slice_u(&[1, 0, 0]),
            MS::from_row_slice_u(&[0, 2, 0]),
            MS::from_row_slice_u(&[0, 0, 1]),
            MS::from_row_slice_u(&[0, 2, 0]),
        ];

        let test_graph = Graph::<MS> {
            vertices: graph_vertices,
            edges: graph_edges(),
            all_labels: MS::from_row_slice_u(&[1, 2, 1]),
        };

        // (0: N, 0: a) -> (1: b)
        // (0: N, 1: b) -> (2: c)
        // (1: S, 1: b) -> (0: a)
        // (1: S, 2: c) -> (1: b)
        // (2: E, 0: a) -> (1: b)
        // (2: E, 1: b) -> (2: c)
        // (3: W, 1: b) -> (0: a)
        // (3: W, 2: c) -> (1: b)

        let result: Rules<MS> = hash_map(&[
            //                              a  b  c  <-- the labels
            ((0, 0), MS::from_row_slice_u(&[0, 2, 0])),
            ((0, 1), MS::from_row_slice_u(&[0, 0, 1])),
            ((1, 1), MS::from_row_slice_u(&[1, 0, 0])),
            ((1, 2), MS::from_row_slice_u(&[0, 2, 0])),
            ((2, 0), MS::from_row_slice_u(&[0, 2, 0])),
            ((2, 1), MS::from_row_slice_u(&[0, 0, 1])),
            ((3, 1), MS::from_row_slice_u(&[1, 0, 0])),
            ((3, 2), MS::from_row_slice_u(&[0, 2, 0])),
        ]);

        assert_eq!(test_graph.rules(), result);
    }

    #[test]
    fn test_rules_multiple() {
        /*
        1b---2c
        |    |
        0a---3a

        North = 0, South = 1, East = 2, West = 3
        */

        let graph_vertices: Vec<MS> = vec![
            MS::from_row_slice_u(&[2, 0, 0]),
            MS::from_row_slice_u(&[0, 1, 0]),
            MS::from_row_slice_u(&[0, 0, 1]),
            MS::from_row_slice_u(&[2, 0, 0]),
        ];

        let test_graph = Graph::<MS> {
            vertices: graph_vertices,
            edges: graph_edges(),
            all_labels: MS::from_row_slice_u(&[2, 1, 1]),
        };

        /*
        (0: N, 0: a) -> (1: b, 2: c)
        (1: S, 1: b) -> (0: a)
        (1: S, 2: c) -> (0: a)
        (2: E, 0: a) -> (0: a)
        (2: E, 1: b) -> (2: c)
        (3: W, 0: a) -> (0: a)
        (3: W, 2: c) -> (1: b)
        */

        let result: Rules<MS> = hash_map(&[
            ((0, 0), MS::from_row_slice_u(&[0, 1, 1])),
            ((1, 1), MS::from_row_slice_u(&[2, 0, 0])),
            ((1, 2), MS::from_row_slice_u(&[2, 0, 0])),
            ((2, 0), MS::from_row_slice_u(&[2, 0, 0])),
            ((2, 1), MS::from_row_slice_u(&[0, 0, 1])),
            ((3, 0), MS::from_row_slice_u(&[2, 0, 0])),
            ((3, 2), MS::from_row_slice_u(&[0, 1, 0])),
        ]);

        assert_eq!(test_graph.rules(), result);
    }

    #[test]
    fn test_rules_partially_collapsed() {
        /*
        1b ---- 2c
        |       |
        0ab --- 3a

        North = 0, South = 1, East = 2, West = 3
        */

        let graph_vertices: Vec<MS> = vec![
            MS::from_row_slice_u(&[2, 2, 0]),
            MS::from_row_slice_u(&[0, 2, 0]),
            MS::from_row_slice_u(&[0, 0, 1]),
            MS::from_row_slice_u(&[2, 0, 0]),
        ];

        let test_graph = Graph::<MS> {
            vertices: graph_vertices,
            edges: graph_edges(),
            all_labels: MS::from_row_slice_u(&[2, 2, 1]),
        };

        /*
        (0: N, 0: a) -> (1: b, 2: c)
        (0: N, 1: b) -> (1: b)
        (1: S, 1: b) -> (0: a, 1: b)
        (1: S, 2: c) -> (0: a)
        (2: E, 0: a) -> (0: a)
        (2: E, 1: b) -> (0: a, 2: c)
        (3: W, 0: a) -> (0: a, 1: b)
        (3: W, 2: c) -> (1: b)
        */

        let result: Rules<MS> = hash_map(&[
            ((0, 0), MS::from_row_slice_u(&[0, 2, 1])),
            ((0, 1), MS::from_row_slice_u(&[0, 2, 0])),
            ((1, 1), MS::from_row_slice_u(&[2, 2, 0])),
            ((1, 2), MS::from_row_slice_u(&[2, 0, 0])),
            ((2, 0), MS::from_row_slice_u(&[2, 0, 0])),
            ((2, 1), MS::from_row_slice_u(&[2, 0, 1])),
            ((3, 0), MS::from_row_slice_u(&[2, 2, 0])),
            ((3, 2), MS::from_row_slice_u(&[0, 2, 0])),
        ]);

        assert_eq!(test_graph.rules(), result);
    }
}
