use crate::graph::graph::{EdgeDirection, Edges, Graph, Rules, VertexIndex};
use crate::multiset::{Multiset, MultisetScalar, MultisetTrait};
use crate::wfc::observe::Observe;
use crate::wfc::propagate::Propagate;
use hashbrown::HashSet;
use nalgebra::allocator::Allocator;
use nalgebra::{DefaultAllocator, Dim, DimName};
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::BinaryHeap;
use std::ops::{Index, IndexMut};

type InitCollapse = (
    HashSet<VertexIndex>, // observed
    Vec<Propagate>,       // propagations
    Vec<VertexIndex>,     // to_observe
    BinaryHeap<Observe>,  // heap
);

fn init_collapse<D>(rng: &mut StdRng, out_graph: &Graph<D>) -> InitCollapse
where
    D: Dim + DimName,
    DefaultAllocator: Allocator<MultisetScalar, D>,
{
    let mut observed: HashSet<VertexIndex> = HashSet::new();
    let mut propagations: Vec<Propagate> = Vec::new();
    let mut init_propagations: Vec<VertexIndex> = Vec::new();
    let mut heap: BinaryHeap<Observe> = BinaryHeap::new();

    // initialize heap and observed
    out_graph
        .vertices
        .iter()
        .enumerate()
        .for_each(|(_index, labels)| {
            assert!(labels.is_subset(&out_graph.all_labels));
            let from_index = _index as VertexIndex;
            if labels.is_singleton() {
                observed.insert(from_index);
            } else if labels != &out_graph.all_labels {
                init_propagations.push(from_index);
                heap.push(Observe::new(&from_index, labels.entropy()))
            }
        });

    // Ensure that output graph is fully propagated before collapse.
    init_propagations.drain(..).for_each(|index| {
        generate_propagations(&mut propagations, &observed, &out_graph.edges, &index);
    });

    let to_observe_len = out_graph.vertices.len() as VertexIndex;
    let mut to_observe: Vec<VertexIndex> = {
        (0..to_observe_len)
            .filter(|i| !observed.contains(i))
            .collect()
    };
    to_observe.shuffle(rng);

    (observed, propagations, to_observe, heap)
}

fn exec_collapse<D>(
    rng: &mut StdRng,
    rules: &Rules<D>,
    edges: &Edges,
    init: InitCollapse,
    mut vertices: Vec<Multiset<D>>,
) -> Option<Vec<Multiset<D>>>
where
    D: Dim + DimName,
    DefaultAllocator: Allocator<MultisetScalar, D>,
{
    let (mut observed, mut propagations, mut to_observe, mut heap) = init;
    let mut gen_observe: HashSet<VertexIndex> = HashSet::new();
    loop {
        // propagate constraints
        while !propagations.is_empty() {
            let propagate = propagations.pop().unwrap();
            assert!(vertices.len() >= propagate.from as usize);
            let prop_labels = vertices.index(propagate.from as usize);
            let constraint = build_constraint(prop_labels, propagate.direction, rules);
            assert!(vertices.len() >= propagate.to as usize);
            let labels = vertices.index_mut(propagate.to as usize);
            let constrained = labels.intersection(&constraint);
            if labels != &constrained {
                if constrained.empty() {
                    // No possible value for this vertex, indicating contradiction!
                    return None;
                } else if constrained.is_singleton() {
                    observed.insert(propagate.to);
                } else {
                    gen_observe.insert(propagate.to);
                }
                generate_propagations(&mut propagations, &observed, edges, &propagate.to);
                *labels = constrained
            }
        }

        // check if all vertices observed, if so we have finished
        if observed.len() == vertices.len() {
            return Some(vertices);
        }

        // generate observes for constrained vertices
        gen_observe.drain().for_each(|index| {
            assert!(vertices.len() >= index as usize);
            let labels = vertices.index(index as usize);
            heap.push(Observe::new(&index, labels.entropy()))
        });

        // try to find a vertex index to observe
        let mut observe_index: Option<VertexIndex> = None;
        // check the heap first
        while !heap.is_empty() {
            let observe = heap.pop().unwrap();
            if !observed.contains(&observe.index) {
                observe_index = Some(observe.index);
                break;
            }
        }
        // if no index to check in heap, check vec of initial vertices to observe
        if observe_index.is_none() {
            while !to_observe.is_empty() {
                let index = to_observe.pop().unwrap();
                if !observed.contains(&index) {
                    observe_index = Some(index);
                    break;
                }
            }
        }
        match observe_index {
            None => return Some(vertices), // Nothing left to observe, therefore we've finished
            Some(index) => {
                assert!(vertices.len() >= index as usize);
                let labels_multiset = vertices.index_mut(index as usize);
                labels_multiset.choose(rng);
                observed.insert(index);
                generate_propagations(&mut propagations, &observed, edges, &index);
            }
        }
    }
}

pub fn build_constraint<D>(
    labels: &Multiset<D>,
    direction: EdgeDirection,
    rules: &Rules<D>,
) -> Multiset<D>
where
    D: Dim + DimName,
    DefaultAllocator: Allocator<MultisetScalar, D>,
{
    labels
        .iter()
        .enumerate()
        .fold(Multiset::<D>::zeros(), |mut acc, (index, frq)| {
            if frq > &0 {
                if let Some(a) = rules.get(&(direction, index)) {
                    acc = acc.union(a)
                }
            }
            acc
        })
}

fn generate_propagations(
    propagations: &mut Vec<Propagate>,
    observed: &HashSet<VertexIndex>,
    edges: &Edges,
    from_index: &VertexIndex,
) {
    assert!(edges.len() >= *from_index as usize);
    edges
        .index(from_index)
        .iter()
        .for_each(|(to_index, direction)| {
            if !observed.contains(to_index) {
                propagations.push(Propagate::new(*from_index, *to_index, *direction))
            }
        });
}

pub fn collapse<D>(
    input_graph: &Graph<D>,
    mut output_graph: Graph<D>,
    seed: Option<u64>,
    tries: Option<usize>,
) -> Option<Graph<D>>
where
    D: Dim + DimName,
    DefaultAllocator: Allocator<MultisetScalar, D>,
{
    let rng = &mut StdRng::seed_from_u64(seed.unwrap_or_else(|| thread_rng().next_u64()));

    let rules = &input_graph.rules();
    let init = init_collapse(rng, &output_graph);

    for _ in 0..tries.unwrap_or(10) {
        if let Some(vertices) = exec_collapse(
            rng,
            rules,
            &output_graph.edges,
            init.clone(),
            output_graph.vertices.clone(),
        ) {
            output_graph.vertices = vertices;
            return Some(output_graph);
        }
    }
    println!("CONTRADICTION!");
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph::Edges;
    use crate::utils::hash_map;
    use nalgebra::{U2, U6};

    //noinspection DuplicatedCode
    fn simple_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)]),
        ])
    }

    //noinspection DuplicatedCode
    fn simple_vertices() -> Vec<Multiset<U6>> {
        vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
        ]
    }

    //noinspection DuplicatedCode
    fn simple_rules() -> Rules<U6> {
        hash_map(&[
            ((0, 0), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((0, 1), Multiset::from_row_slice_u(&[0, 0, 1])),
            ((1, 1), Multiset::from_row_slice_u(&[1, 0, 0])),
            ((1, 2), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((2, 0), Multiset::from_row_slice_u(&[0, 2, 0])),
            ((2, 1), Multiset::from_row_slice_u(&[0, 0, 1])),
            ((3, 1), Multiset::from_row_slice_u(&[1, 0, 0])),
            ((3, 2), Multiset::from_row_slice_u(&[0, 2, 0])),
        ])
    }

    #[test]
    fn test_constraint() {
        let a: Multiset<U6> = Multiset::from_row_slice_u(&[1, 0, 0, 0, 0, 0]);
        let b: Multiset<U6> = Multiset::from_row_slice_u(&[0, 0, 1, 0, 0, 0]);
        let c: Multiset<U6> = Multiset::from_row_slice_u(&[1, 1, 1, 0, 0, 0]);
        let rules: Rules<U6> = hash_map(&[((0, 0), a), ((0, 1), b), ((0, 2), c)]);

        let labels: &Multiset<U6> = &Multiset::from_row_slice_u(&[2, 4, 0, 0, 0, 0]);
        let direction: EdgeDirection = 0;

        let result = build_constraint(labels, direction, &rules);
        let expected: Multiset<U6> = Multiset::from_row_slice_u(&[1, 0, 1, 0, 0, 0]);
        assert_eq!(result, expected)
    }

    //noinspection DuplicatedCode
    #[test]
    fn test_exec_simple() {
        /* Seed Values:
            3 -> Some([{0}, {1}, {2}, {1}])
            1 -> None
        */
        let mut rng = StdRng::seed_from_u64(3);

        let edges = simple_edges();
        let all_labels = Multiset::<U6>::from_row_slice_u(&[1, 2, 1]);
        let out_graph = Graph::<U6>::new(simple_vertices(), edges, all_labels);
        let rules: &Rules<U6> = &simple_rules();

        let init = init_collapse::<U6>(&mut rng, &out_graph);

        let result =
            exec_collapse::<U6>(&mut rng, rules, &out_graph.edges, init, simple_vertices())
                .unwrap();
        let expected: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_exec_med() {
        /* Seed Values:
            INPUT:
            0b --- 1b --- 2a
            |      |      |
            3a --- 4b --- 5a

            Output structure same as input structure.
            North = 0, South = 1, East = 2, West = 3
        */
        let mut rng = StdRng::seed_from_u64(234);

        let edges = hash_map(&[
            (0, vec![(3, 1), (1, 2)]),
            (1, vec![(0, 3), (4, 1), (2, 2)]),
            (2, vec![(1, 3), (5, 1)]),
            (3, vec![(0, 0), (4, 2)]),
            (4, vec![(3, 3), (1, 0), (5, 2)]),
            (5, vec![(4, 3), (2, 0)]),
        ]);

        let vertices: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
        ];

        let rules: Rules<U2> = hash_map(&[
            ((0, 0), Multiset::from_row_slice_u(&[3, 3])),
            ((0, 1), Multiset::from_row_slice_u(&[3, 3])),
            ((1, 0), Multiset::from_row_slice_u(&[3, 3])),
            ((1, 1), Multiset::from_row_slice_u(&[3, 3])),
            ((2, 0), Multiset::from_row_slice_u(&[3, 3])),
            ((2, 1), Multiset::from_row_slice_u(&[3, 3])),
            ((3, 0), Multiset::from_row_slice_u(&[3, 3])),
            ((3, 1), Multiset::from_row_slice_u(&[3, 3])),
        ]);

        let all_labels = Multiset::from_row_slice_u(&[3, 3]);
        let out_graph = Graph::<U2>::new(vertices, edges, all_labels);
        let init = init_collapse::<U2>(&mut rng, &out_graph);

        let result = exec_collapse::<U2>(
            &mut rng,
            &rules,
            &out_graph.edges,
            init,
            out_graph.vertices.clone(),
        )
        .unwrap();

        let expected: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
        ];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_exec_complex() {
        /*
            INPUT graph:
                   0a --- 1b --- 2b
                   |      |
            3a --- 4a --- 5b

            Directions: North = 0, South = 1, East = 2, West = 3
        */

        let mut rng = StdRng::seed_from_u64(1);

        let input_edges = hash_map(&[
            (0, vec![(1, 2), (4, 1)]),
            (1, vec![(0, 3), (5, 1), (2, 2)]),
            (2, vec![(1, 3)]),
            (3, vec![(4, 2)]),
            (4, vec![(3, 3), (0, 0), (5, 2)]),
            (5, vec![(4, 3), (1, 0)]),
        ]);

        let input_vertices: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
        ];

        let all_labels: Multiset<U2> = Multiset::from_row_slice_u(&[3, 3]);

        let input_graph = Graph::<U2>::new(input_vertices, input_edges, all_labels);

        let rules = input_graph.rules();

        /*
            OUTPUT STRUCTURE:
            0 ---- 1 ---- 2 ---- 3
            |      |      |      |
            4 ---- 5 ---- 6 ---- 7
            |      |      |      |
            8 ---- 9 ---- 10 --- 11
        */

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
            (11, vec![(10, 3), (7, 0)]),
        ]);

        let output_vertices: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
            Multiset::from_row_slice_u(&[3, 3]),
        ];

        // should be same as all labels but Multiset.clone() is outputs false
        // positive errors, so recreating here to avoid.
        let out_all_labels: Multiset<U2> = Multiset::from_row_slice_u(&[3, 3]);

        let output_graph = Graph::<U2>::new(output_vertices, output_edges, out_all_labels);
        let init = init_collapse::<U2>(&mut rng, &output_graph);

        let result = exec_collapse::<U2>(
            &mut rng,
            &rules,
            &output_graph.edges,
            init,
            output_graph.vertices.clone(),
        )
        .unwrap();

        let expected: Vec<Multiset<U2>> = vec![
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[3, 0]),
            Multiset::from_row_slice_u(&[0, 3]),
            Multiset::from_row_slice_u(&[0, 3]),
        ];

        assert_eq!(result, expected);
    }
}
