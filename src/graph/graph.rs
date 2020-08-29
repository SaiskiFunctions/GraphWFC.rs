use rand::prelude::*;
use hashbrown::HashMap;
use std::ops::{Index, IndexMut};
use crate::multiset::{Multiset, MultisetTrait, MultisetScalar};
use nalgebra::{DimName, Dim, DefaultAllocator};
use nalgebra::allocator::Allocator;


pub type VertexIndex = u32;     // each unique vertex in a graph
pub type EdgeDirection = u16;   // the directional relationship between two vertices
pub type Edges = HashMap<VertexIndex, Vec<(VertexIndex, EdgeDirection)>>;


//                           vertex label (index of LabelFrequencies vector)
//                                            |
//                                            v
pub type Rules<D> = HashMap<(EdgeDirection, usize), Multiset<D>>;

#[derive(Debug, Clone)]
pub struct Graph<D>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    pub vertices: Vec<Multiset<D>>, // index of vec == vertex index
    pub edges: Edges,
    pub all_labels: Multiset<D>
}

impl<D> Graph<D>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    pub fn new(vertices: Vec<Multiset<D>>, edges: Edges, all_labels: Multiset<D>) -> Graph<D> {
        Graph { vertices, edges, all_labels }
    }

    pub fn empty() -> Graph<D> {
        Graph::new(Vec::new(), HashMap::new(), Multiset::<D>::zeros())
    }

    /// Construct HashMap of rules for this graph.
    /// Rules connect a tuple of direction and vertex label to a set of labels.
    pub fn rules(&self) -> Rules<D> {
        self.edges.iter().fold(HashMap::new(), |mut rules, (from_vertex_index, edges)| {
            edges.iter().for_each(|(to_vertex_index, direction)| {
                self.vertices.index(*from_vertex_index as usize).iter()
                    .enumerate()
                    .filter(|(_, &label)| label > 0)
                    .for_each(|(from_vertex_label, _)| {
                        let rules_key = (*direction, from_vertex_label);
                        let union_labels = self.vertices.index(*to_vertex_index as usize);
                        rules.entry(rules_key)
                            .and_modify(|to_labels| *to_labels = to_labels.union(union_labels))
                            .or_insert(union_labels.clone());
                    })
            });
            rules
        })
    }

    /// Collapses the set of vertex labels at the given index to a singleton set.
    pub fn observe(&mut self, rng: &mut StdRng, index: &VertexIndex) {
        assert!(self.vertices.len() >= *index as usize);
        let labels_multiset = self.vertices.index_mut(*index as usize);
        labels_multiset.choose(rng);
    }

    /// Constrain the vertex labels at the given index by intersecting the
    /// vertex labels with the constraint set.
    /// Return Some(labels) if the labels set was changed else return None.
    pub fn constrain(&mut self, index: &VertexIndex, constraint: &Multiset<D>) -> Option<&Multiset<D>> {
        assert!(self.vertices.len() >= *index as usize);
        let labels = self.vertices.index_mut(*index as usize);
        let inter = labels.intersection(constraint);
        if labels == &inter { return None }
        *labels = inter;
        Some(labels)
    }
}

#[cfg(test)]
mod graph2_tests {
    use super::*;
    use crate::utils::hash_map;
    use nalgebra::U6;

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

        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0])
        ];

        let test_graph = Graph::<U6> {
            vertices: graph_vertices,
            edges: graph_edges(),
            all_labels: Multiset::from_row_slice_u(&[1, 2, 1])
        };

        // (0: N, 0: a) -> (1: b)
        // (0: N, 1: b) -> (2: c)
        // (1: S, 1: b) -> (0: a)
        // (1: S, 2: c) -> (1: b)
        // (2: E, 0: a) -> (1: b)
        // (2: E, 1: b) -> (2: c)
        // (3: W, 1: b) -> (0: a)
        // (3: W, 2: c) -> (1: b)

        let result: Rules<U6> = hash_map(&[
            ((0, 0), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((0, 1), Multiset::from_row_slice_u(&[0, 0, 1])),
            ((1, 1), Multiset::from_row_slice_u(&[1, 0, 0])),
            ((1, 2), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((2, 0), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((2, 1), Multiset::from_row_slice_u(&[0, 0, 1])),
            ((3, 1), Multiset::from_row_slice_u(&[1, 0, 0])),
            ((3, 2), Multiset::from_row_slice_u(&[0, 2, 0])),
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

        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[2, 0, 0]),
            Multiset::from_row_slice_u(&[0, 1, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[2, 0, 0])
        ];

        let test_graph = Graph::<U6> {
            vertices: graph_vertices,
            edges: graph_edges(),
            all_labels: Multiset::from_row_slice_u(&[2, 1, 1])
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

        let result: Rules<U6> = hash_map(&[
            ((0, 0), Multiset::from_row_slice_u(&[0, 1, 1])),
            ((1, 1), Multiset::from_row_slice_u(&[2, 0, 0])),
            ((1, 2), Multiset::from_row_slice_u(&[2, 0, 0])),
            ((2, 0), Multiset::from_row_slice_u(&[2, 0, 0])),
            ((2, 1), Multiset::from_row_slice_u(&[0, 0, 1])),
            ((3, 0), Multiset::from_row_slice_u(&[2, 0, 0])),
            ((3, 2), Multiset::from_row_slice_u(&[0, 1, 0])),
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

        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[2, 2, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[2, 0, 0])
        ];

        let test_graph = Graph::<U6> {
            vertices: graph_vertices,
            edges: graph_edges(),
            all_labels: Multiset::from_row_slice_u(&[2, 2, 1])
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

        let result: Rules<U6> = hash_map(&[
            ((0, 0), Multiset::from_row_slice_u(&[0, 2, 1])),
            ((0, 1), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((1, 1), Multiset::from_row_slice_u(&[2, 2, 0])),
            ((1, 2), Multiset::from_row_slice_u(&[2, 0, 0])),
            ((2, 0), Multiset::from_row_slice_u(&[2, 0, 0])),
            ((2, 1), Multiset::from_row_slice_u(&[2, 0, 1])),
            ((3, 0), Multiset::from_row_slice_u(&[2, 2, 0])),
            ((3, 2), Multiset::from_row_slice_u(&[0, 2, 0])),
        ]);

        assert_eq!(test_graph.rules(), result);
    }

    #[test]
    fn test_observe() {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[3, 3, 0, 0]),
            Multiset::from_row_slice_u(&[3, 3, 1, 2]),
            Multiset::from_row_slice_u(&[0, 3, 0, 2]),
            Multiset::from_row_slice_u(&[3, 0, 0, 0])
        ];

        let mut test_graph = Graph::<U6> {
            vertices: graph_vertices,
            edges: HashMap::new(),
            all_labels: Multiset::from_row_slice_u(&[3, 3, 1, 2])
        };

        let mut test_rng = StdRng::seed_from_u64(2);
        let index = 1;

        test_graph.observe(&mut test_rng, &index);

        let expected: Multiset<U6> = Multiset::from_row_slice_u(&[0, 0, 1, 0]);

        assert_eq!(*test_graph.vertices.index(index as usize), expected);
    }

    #[test]
    fn test_constrain_true() {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 1, 1, 1])
        ];

        let test_constraint = Multiset::from_row_slice_u(&[1, 1, 0, 0]);

        let mut test_graph = Graph::<U6> {
            vertices: graph_vertices,
            edges: HashMap::new(),
            all_labels: Multiset::<U6>::from_row_slice_u(&[1, 1, 1, 1])
        };

        let result: Multiset<U6> = Multiset::from_row_slice_u(&[1, 1, 0, 0]);
        assert_eq!(test_graph.constrain(&0, &test_constraint), Some(&result));
        assert_eq!(*test_graph.vertices.index(0), test_constraint);
    }

    #[test]
    fn test_constrain_false() {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 1, 0])
        ];

        let test_constraint: Multiset<U6> = Multiset::from_row_slice_u(&[1, 1, 1]);

        let mut test_graph = Graph::<U6> {
            vertices: graph_vertices,
            edges: HashMap::new(),
            all_labels: Multiset::from_row_slice_u(&[1, 1, 0])
        };

        let vertex_result: Multiset<U6> =  Multiset::from_row_slice_u(&[1, 1, 0]);

        assert_eq!(test_graph.constrain(&0, &test_constraint), None);
        assert_eq!(*test_graph.vertices.index(0), vertex_result);
    }
}
