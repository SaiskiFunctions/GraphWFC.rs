use rand::prelude::*;
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use std::ops::Index;
use crate::utils::hash_set;
use nalgebra::DVector;


pub type VertexLabel = i32;     // labels that a vertex can contain
pub type VertexIndex = i32;     // each unique vertex in a graph
pub type EdgeDirection = u16;   // the directional relationship between two vertices
pub type Edges = HashMap<VertexIndex, Vec<(VertexIndex, EdgeDirection)>>;
pub type Rules = HashMap<(EdgeDirection, VertexLabel), HashSet<VertexLabel>>;
pub type Labels = HashSet<VertexLabel>;
pub type Frequencies = HashMap<VertexLabel, i32>;

pub type LabelFrequencies = DVector<u32>;

#[derive(Debug, Clone)]
pub struct Graph {
    pub vertices: Vec<Labels>, // index of vec == vertex index
    edges: Edges,
}

impl Graph {
    pub fn new(vertices: Vec<Labels>, edges: Edges) -> Graph {
        Graph { vertices, edges }
    }

    pub fn empty() -> Graph {
        Graph { vertices: Vec::new(), edges: HashMap::new() }
    }

    /// Construct HashMap of rules for this graph.
    /// Rules connect a tuple of direction and vertex label to a set of labels.
    pub fn rules(&self) -> Rules {
        let mut rules: Rules = HashMap::new();
        for (from_vertex_index, edges) in self.edges.iter() {
            for (to_vertex_index, direction) in edges.iter() {
                for from_vertex_label in self.vertices[*from_vertex_index as usize].iter() {
                    let rules_key = (*direction, *from_vertex_label);
                    rules.entry(rules_key)
                        .and_modify(|to_labels| to_labels.extend(&self.vertices[*to_vertex_index as usize]))
                        .or_insert(self.vertices[*to_vertex_index as usize].clone());
                }
            }
        }
        rules
    }

    /// Construct HashMap of label frequencies for this graph.
    pub fn frequencies(&self) -> Frequencies {
        self.vertices.iter().fold(HashMap::new(), |mut map, labels| {
            labels.iter().for_each(|label| {
                map.entry(*label).and_modify(|n| *n += 1).or_insert(1);
            });
            map
        })
    }

    /// Construct the set of all labels for this graph.
    pub fn all_labels(&self) -> Labels {
        self.vertices.iter().fold(HashSet::new(), |mut all, labels| {
            all.extend(labels);
            all
        })
    }

    /// Return all pairs of vertex indexes and directions connected to the
    /// given index for this graph.
    pub fn connections(&self, index: &VertexIndex) -> &Vec<(VertexIndex, EdgeDirection)> {
        self.edges.index(index)
    }

    /// Collapses the set of vertex labels at the given index to a singleton set.
    pub fn observe(&mut self, rng: &mut StdRng, index: &VertexIndex, frequencies: &Frequencies) {
        let labels = &mut self.vertices[*index as usize];
        let total: i32 = labels.iter().fold(0, |acc, label| {
            &acc + frequencies.index(label)
        });
        let choice = rng.gen_range(1, total + 1);
        let mut acc = 0;

        // We have to sort the labels to ensure deterministic choice of collapsed label.
        let mut sorted_labels = Vec::from_iter(labels.iter());
        sorted_labels.sort();

        *labels = hash_set(&[**sorted_labels.iter().skip_while(|label| {
            acc += *frequencies.index(label);
            acc < choice
        }).next().unwrap()]);
    }

    /// Constrain the vertex labels at the given index by intersecting the
    /// vertex labels with the constraint set.
    /// Return Some(labels) if the labels set was changed else return None.
    pub fn constrain(&mut self, index: &VertexIndex, constraint: &Labels) -> Option<&Labels> {
        let labels = &mut self.vertices[*index as usize];
        if labels.is_subset(constraint) { return None; }
        *labels = labels.intersection(constraint).map(|x| *x).collect();
        Some(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::hash_map;
    use std::iter::FromIterator;
    use nalgebra::{Vector3, sup, inf};

    #[test]
    fn test_matrix_intersect() {
        let a = DVector::from_iterator(3, vec![1, 0, 1].into_iter());
        let b = DVector::from_iterator(3, vec![0, 0, 1].into_iter());
        assert_eq!(b, a.component_mul(&b))
    }

    #[test]
    fn test_matrix_union() {
        let a = DVector::from_iterator(3, vec![1, 0, 1].into_iter());
        let b = DVector::from_iterator(3, vec![0, 0, 1].into_iter());
        assert_eq!(a, sup(&a, &b))
    }

    fn graph_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)])
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

        let test_graph_vertices: Vec<Labels> = Vec::from_iter(
            [0, 1, 2, 1].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: graph_edges(),
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
            ((0, 0), hash_set(&[1])),
            ((0, 1), hash_set(&[2])),
            ((1, 1), hash_set(&[0])),
            ((1, 2), hash_set(&[1])),
            ((2, 0), hash_set(&[1])),
            ((2, 1), hash_set(&[2])),
            ((3, 1), hash_set(&[0])),
            ((3, 2), hash_set(&[1])),
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

        let test_graph_vertices: Vec<Labels> = Vec::from_iter(
            [0, 1, 2, 0].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: graph_edges(),
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
            ((0, 0), hash_set(&[1, 2])),
            ((1, 1), hash_set(&[0])),
            ((1, 2), hash_set(&[0])),
            ((2, 0), hash_set(&[0])),
            ((2, 1), hash_set(&[2])),
            ((3, 0), hash_set(&[0])),
            ((3, 2), hash_set(&[1])),
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

        let test_graph_vertices: Vec<Labels> = vec![
            hash_set(&[0, 1]),
            hash_set(&[1]),
            hash_set(&[2]),
            hash_set(&[0])
        ];

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: graph_edges(),
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
            ((0, 0), hash_set(&[1, 2])),
            ((0, 1), hash_set(&[1])),
            ((1, 1), hash_set(&[0, 1])),
            ((1, 2), hash_set(&[0])),
            ((2, 0), hash_set(&[0])),
            ((2, 1), hash_set(&[0, 2])),
            ((3, 0), hash_set(&[0, 1])),
            ((3, 2), hash_set(&[1])),
        ]);

        assert_eq!(test_graph.rules(), result);
    }

    #[test]
    fn test_frequencies() {
        let test_graph_vertices: Vec<Labels> = Vec::from_iter(
            [0, 1, 2, 1].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: HashMap::new(),
        };

        let result = hash_map(&[(0, 1), (1, 2), (2, 1)]);

        assert_eq!(test_graph.frequencies(), result);
    }

    #[test]
    fn test_frequencies_complex() {
        let test_graph_vertices: Vec<Labels> = Vec::from_iter(
            [0, 1, 2, 1, 1, 1, 2, 3, 4, 5, 5, 0, 0, 1, 2, 4, 5, 6, 0].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: HashMap::new(),
        };

        let result = hash_map(&[(0, 4), (1, 5), (2, 3), (3, 1), (4, 2), (5, 3), (6, 1)]);

        assert_eq!(test_graph.frequencies(), result);
    }

    #[test]
    fn test_observe() {
        let test_graph_vertices: Vec<Labels> = vec![
            hash_set(&[0, 1]),
            hash_set(&[0, 1, 2, 3]),
            hash_set(&[1, 3]),
            hash_set(&[0])
        ];

        let test_frequencies = hash_map(&[(0, 1), (1, 2), (2, 100), (3, 1)]);

        let mut test_graph = Graph {
            vertices: test_graph_vertices,
            edges: HashMap::new(),
        };

        let mut test_rng = StdRng::seed_from_u64(10);
        let index = 1;

        test_graph.observe(&mut test_rng, &index, &test_frequencies);

        let expected: HashSet<i32> = hash_set(&[2]);

        assert_eq!(*test_graph.vertices.get(index as usize).unwrap(), expected);
    }

    #[test]
    fn test_constrain_true() {
        let test_graph_vertices: Vec<Labels> = vec![
            hash_set(&[0, 1, 2, 3])
        ];

        let test_constraint = hash_set(&[0, 1]);

        let mut test_graph = Graph {
            vertices: test_graph_vertices,
            edges: HashMap::new(),
        };

        assert_eq!(test_graph.constrain(&0, &test_constraint), Some(&hash_set(&[0, 1])));

        assert_eq!(*test_graph.vertices.get(0).unwrap(), test_constraint);
    }

    #[test]
    fn test_constrain_false() {
        let test_graph_vertices: Vec<Labels> = vec![
            hash_set(&[0, 1])
        ];

        let test_constraint = hash_set(&[0, 1, 2]);

        let mut test_graph = Graph {
            vertices: test_graph_vertices,
            edges: HashMap::new(),
        };

        assert_eq!(test_graph.constrain(&0, &test_constraint), None);

        assert_eq!(*test_graph.vertices.get(0).unwrap(), hash_set(&[0, 1]));
    }

    #[test]
    fn test_all_labels() {
        let test_graph_vertices: Vec<Labels> = vec![
            hash_set(&[0, 1]),
            hash_set(&[1, 2]),
            hash_set(&[0, 2, 3, 4]),
            hash_set(&[1, 2, 3, 0]),
            hash_set(&[5, 6])
        ];

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: HashMap::new(),
        };

        assert_eq!(test_graph.all_labels(), hash_set(&[0, 1, 2, 3, 4, 5, 6]));
    }

    #[test]
    fn test_connections() {
        let test_graph = Graph {
            vertices: Vec::new(),
            edges: graph_edges(),
        };

        assert_eq!(test_graph.connections(&3), &vec![(0, 3), (2, 0)])
    }
}
