use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;
use rand::{thread_rng, Rng};

use crate::utils::{hash_map, hash_set};

pub type VertexLabel = i32;           // labels that a vertex can contain
pub type VertexIndex = i32;           // each unique vertex in a graph
pub type EdgeDirection = u16;       // the directional relationship between two vertices
pub type Edges = HashMap<VertexIndex, Vec<(VertexIndex, EdgeDirection)>>;
pub type Rules = HashMap<(EdgeDirection, VertexLabel), HashSet<VertexLabel>>;
pub type Labels = HashSet<VertexLabel>;
pub type Frequencies = HashMap<VertexLabel, i32>;

#[derive(Debug)]
pub struct Graph {
    pub vertices: Vec<HashSet<VertexLabel>>,
    edges: Edges
}

impl Graph {
    pub fn new(vertices: Vec<HashSet<VertexLabel>>, edges: Edges) -> Graph {
        Graph { vertices, edges }
    }

    /*
    1. Construct rules hash map
    2. loop through graph directed edges
    2. for each edge:
        try add (dir, vertexLabel) to rules, if already in rules, union set labels
    */
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

    pub fn frequencies(&self) -> Frequencies {
        let mut frequencies = HashMap::new();
        self.vertices.iter().for_each(|labels| {
            labels.iter().for_each(|label| {
                let frequency = frequencies.entry(*label).or_insert(0);
                *frequency += 1
            });
        });
        frequencies
    }

    /*
    TODO:
        1. Seedable randomness
        2. Dependency injected random module
        3. Test
        4. cache calculation of total.
        5. should observe and constrain return a bool?
    */
    pub fn observe(&mut self, index: &VertexIndex, frequencies: &Frequencies) {
        let mut rng = thread_rng();
        self.vertices.get_mut(*index as usize).map(|labels| {
            let total: i32 = labels.iter().fold(0, |mut acc, label| {
                acc + *frequencies.get(label).unwrap()
            });
            let choice = rng.gen_range(1, total + 1);
            let mut acc = 0;

            *labels.iter().skip_while(|label| {
                acc += *frequencies.get(label).unwrap();
                acc < choice
            }).next().unwrap()
        });
    }

    pub fn constrain(&mut self, index: &VertexIndex, labels: &Labels) {
        self.vertices.get_mut(*index as usize).map(|set| {
            set.intersection(labels).collect::<HashSet<&VertexLabel>>()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let test_graph_vertices: Vec<HashSet<VertexLabel>> = Vec::from_iter(
            [0, 1, 2, 1].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: graph_edges()
        };

        // (0: N, 0: a) -> (1: b)
        // (0: N, 1: b) -> (2: c)
        // (1: S, 1: b) -> (0: a)
        // (1: S, 2: c) -> (1: b)
        // (2: E, 0: a) -> (1: b)
        // (2: E, 1: b) -> (2: c)
        // (3: W, 1: b) -> (0: a)
        // (3: W, 2: c) -> (1: b)

        let result: HashMap<(EdgeDirection, VertexLabel), HashSet<VertexLabel>> = hash_map(&[
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

        let test_graph_vertices: Vec<HashSet<VertexLabel>> = Vec::from_iter(
            [0, 1, 2, 0].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: graph_edges()
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

        let result: HashMap<(EdgeDirection, VertexLabel), HashSet<VertexLabel>> = hash_map(&[
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

        let test_graph_vertices: Vec<HashSet<VertexLabel>> = vec![
            hash_set(&[0, 1]),
            hash_set(&[1]),
            hash_set(&[2]),
            hash_set(&[0])
        ];

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: graph_edges()
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

        let result: HashMap<(EdgeDirection, VertexLabel), HashSet<VertexLabel>> = hash_map(&[
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
        let test_graph_vertices: Vec<HashSet<VertexLabel>> = Vec::from_iter(
            [0, 1, 2, 1].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: HashMap::new()
        };

        let result = hash_map(&[(0, 1), (1, 2), (2, 1)]);

        assert_eq!(test_graph.frequencies(), result);
    }

    #[test]
    fn test_frequencies_complex() {
        let test_graph_vertices: Vec<HashSet<VertexLabel>> = Vec::from_iter(
            [0, 1, 2, 1, 1, 1, 2, 3, 4, 5, 5, 0, 0, 1, 2, 4, 5, 6, 0].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let test_graph = Graph {
            vertices: test_graph_vertices,
            edges: HashMap::new()
        };

        let result = hash_map(&[(0, 4), (1, 5), (2, 3), (3, 1), (4, 2), (5, 3), (6, 1)]);

        assert_eq!(test_graph.frequencies(), result);
    }
}
