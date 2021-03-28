use hashbrown::HashMap;
use std::ops::{Index, AddAssign};
use std::fmt::{Debug, Formatter, Result};
use crate::MSu16xNU;

pub type VertexIndex = u32; // each unique vertex in a graph
pub type EdgeDirection = u16; // the directional relationship between two vertices
pub type Edges = HashMap<VertexIndex, Vec<(VertexIndex, EdgeDirection)>>;

//                        vertex label (index of LabelFrequencies vector)
//                                         |
//                                         v
pub type Rules = HashMap<(EdgeDirection, usize), MSu16xNU>;

// It will always be true that: |rules| <= |directions| x |labels|

pub struct Graph {
    pub vertices: Vec<MSu16xNU>, // index of vec == vertex index
    pub edges: Edges,
    pub all_labels: MSu16xNU,
}

impl Debug for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f
            .debug_struct("Graph")
            .field("vertices", &self.vertices)
            .field("edges", &self.edges)
            .field("all_labels", &self.all_labels)
            .finish()
    }
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Graph {
            vertices: self.vertices.clone(),
            edges: self.edges.clone(),
            all_labels: self.all_labels
        }
    }
}

impl Graph {
    pub fn new(vertices: Vec<MSu16xNU>, edges: Edges, all_labels: MSu16xNU) -> Graph {
        Graph {
            vertices,
            edges,
            all_labels,
        }
    }

    /// Construct HashMap of rules for this graph.
    /// Rules connect a tuple of direction and vertex label to a set of labels.
    pub fn rules(&self) -> Rules {
        self.edges
            .iter()
            .fold(HashMap::new(), |mut rules, (from_vertex_index, edges)| {
                edges.iter().for_each(|(to_vertex_index, direction)| {
                    let union_labels = self.vertices.index(*to_vertex_index as usize);
                    self.vertices
                        .index(*from_vertex_index as usize)
                        .into_iter()
                        .enumerate()
                        .filter(|(_, label)| label > &0)
                        .for_each(|(from_vertex_label, _)| {
                            let rules_key = (*direction, from_vertex_label);
                            rules
                                .entry(rules_key)
                                .and_modify(|to_labels| to_labels.add_assign(*union_labels))
                                .or_insert(*union_labels);
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
    use std::iter::FromIterator;

    //noinspection DuplicatedCode
    fn graph_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)]),
        ])
    }

    //noinspection DuplicatedCode
    #[test]
    fn test_rules() {
        /*
        1b --- 2c
        |      |
        0a --- 3b

        North = 0, South = 1, East = 2, West = 3
        */

        let graph_vertices: Vec<MSu16xNU> = vec![
            [1, 0, 0].iter().collect(),
            [0, 2, 0].iter().collect(),
            [0, 0, 1].iter().collect(),
            [0, 2, 0].iter().collect(),
        ];

        let test_graph = Graph {
            vertices: graph_vertices,
            edges: graph_edges(),
            all_labels: MSu16xNU::from_iter([1, 2, 1].iter().cloned()),
        };

        // (0: N, 0: a) -> (1: b)
        // (0: N, 1: b) -> (2: c)
        // (1: S, 1: b) -> (0: a)
        // (1: S, 2: c) -> (1: b)
        // (2: E, 0: a) -> (1: b)
        // (2: E, 1: b) -> (2: c)
        // (3: W, 1: b) -> (0: a)
        // (3: W, 2: c) -> (1: b)

        let result: Rules = hash_map(&[
            //        a  b  c  <-- the labels
            ((0, 0), [0, 2, 0].iter().collect()),
            ((0, 1), [0, 0, 1].iter().collect()),
            ((1, 1), [1, 0, 0].iter().collect()),
            ((1, 2), [0, 2, 0].iter().collect()),
            ((2, 0), [0, 2, 0].iter().collect()),
            ((2, 1), [0, 0, 1].iter().collect()),
            ((3, 1), [1, 0, 0].iter().collect()),
            ((3, 2), [0, 2, 0].iter().collect()),
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

        let graph_vertices: Vec<MSu16xNU> = vec![
            [2, 0, 0].iter().collect(),
            [0, 1, 0].iter().collect(),
            [0, 0, 1].iter().collect(),
            [2, 0, 0].iter().collect(),
        ];

        let test_graph = Graph {
            vertices: graph_vertices,
            edges: graph_edges(),
            all_labels: [2, 1, 1].iter().collect(),
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

        let result: Rules = hash_map(&[
            ((0, 0), [0, 1, 1].iter().collect()),
            ((1, 1), [2, 0, 0].iter().collect()),
            ((1, 2), [2, 0, 0].iter().collect()),
            ((2, 0), [2, 0, 0].iter().collect()),
            ((2, 1), [0, 0, 1].iter().collect()),
            ((3, 0), [2, 0, 0].iter().collect()),
            ((3, 2), [0, 1, 0].iter().collect()),
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

        let graph_vertices: Vec<MSu16xNU> = vec![
            [2, 2, 0].iter().collect(),
            [0, 2, 0].iter().collect(),
            [0, 0, 1].iter().collect(),
            [2, 0, 0].iter().collect(),
        ];

        let test_graph = Graph {
            vertices: graph_vertices,
            edges: graph_edges(),
            all_labels: [2, 2, 1].iter().collect(),
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

        let result: Rules = hash_map(&[
            ((0, 0), [0, 2, 1].iter().collect()),
            ((0, 1), [0, 2, 0].iter().collect()),
            ((1, 1), [2, 2, 0].iter().collect()),
            ((1, 2), [2, 0, 0].iter().collect()),
            ((2, 0), [2, 0, 0].iter().collect()),
            ((2, 1), [2, 0, 1].iter().collect()),
            ((3, 0), [2, 2, 0].iter().collect()),
            ((3, 2), [0, 2, 0].iter().collect()),
        ]);

        assert_eq!(test_graph.rules(), result);
    }
}
