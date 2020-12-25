use crate::graph::graph::{EdgeDirection, Edges, Graph, Rules, VertexIndex};
use crate::multiset::Multiset;
use crate::wfc::observe::Observe;
use crate::wfc::propagate::Propagate;
use num_traits::Zero;
use rand::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::BinaryHeap;
use std::mem::replace;
use std::ops::{Index, IndexMut};
use crate::utils::Metrics;
use bit_set::BitSet;
use generic_array::{GenericArray, ArrayLength};
use typenum::{Prod, Unsigned};
use nalgebra::{U6, U8, DimName};

type InitCollapse = (
    BitSet,               // observed
    Vec<Propagate>,       // propagations
    Vec<VertexIndex>,     // to_observe
    BinaryHeap<Observe>,  // heap
);

fn rules_stack<S, L, D>(rules: &Rules<S>) -> GenericArray<Option<S>, Prod<L, D>>
where
    S: Multiset,
    L: std::ops::Mul<D> + Unsigned,
    <L as std::ops::Mul<D>>::Output: ArrayLength<Option<S>>,
{
    let mut array = GenericArray::default();
    rules.into_iter().for_each(|((dir, label), set)| {
        let index = *dir as usize * L::to_usize() + *label;
        *array.index_mut(index) = Some(set.clone());
    });
    array
}

fn init_collapse<S: Multiset>(rng: &mut StdRng, out_graph: &Graph<S>) -> InitCollapse {
    let mut observed: BitSet = BitSet::new();
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
                init_propagations.push(from_index);
                observed.insert(from_index as usize);
            } else if labels != &out_graph.all_labels {
                init_propagations.push(from_index);
                heap.push(Observe::new(from_index, labels.entropy()))
            }
        });

    // Ensure that output graph will be fully propagated before further collapse.
    init_propagations.drain(..).for_each(|index| {
        generate_propagations(&mut propagations, &observed, &out_graph.edges, index);
    });

    let to_observe_len = out_graph.vertices.len() as VertexIndex;
    let mut to_observe: Vec<VertexIndex> = (0..to_observe_len)
        .filter(|i| !observed.contains(*i as usize))
        .collect();
    to_observe.shuffle(rng);

    (observed, propagations, to_observe, heap)
}

const METRICS: bool = false;
const OBSERVE_CHANCE: usize = 33;

fn exec_collapse<S: Multiset>(
    rng: &mut StdRng,
    rules: &Rules<S>,
    edges: &Edges,
    init: InitCollapse,
    mut vertices: Vec<S>,
) -> Vec<S> {
    let (mut observed, mut propagations, mut to_observe, mut heap) = init;
    let mut to_propagate: Vec<Propagate> = Vec::new();
    let vertices_len = vertices.len();
    let mut observed_counter: usize = 0;

    let mut metrics = Metrics::new();

    let stack_rules = rules_stack::<S, <U6 as DimName>::Value, <U8 as DimName>::Value>(rules);

    if METRICS {
        metrics.avg("props/obs", ("props", "obs"));
        metrics.avg("props/loops", ("props", "loops"));
    }

    loop {
        // propagate constraints
        while !propagations.is_empty() {
            if METRICS { metrics.inc("loops") }

            for propagate in propagations.drain(..) {
                if METRICS { metrics.inc("props") }

                assert!(vertices.len() >= propagate.from as usize);
                let prop_labels = vertices.index(propagate.from as usize);
                if prop_labels.is_empty_m() {
                    continue
                }

                let constraint = build_constraint::<S, <U6 as DimName>::Value, <U8 as DimName>::Value>(prop_labels, propagate.direction, &stack_rules);

                assert!(vertices.len() >= propagate.to as usize);
                let labels = vertices.index_mut(propagate.to as usize);

                let constrained = labels.intersection(&constraint);
                if labels != &constrained {
                    if constrained.is_singleton() || constrained.is_empty_m() {
                        observed.insert(propagate.to as usize);
                        observed_counter += 1
                    } else if rng.gen_range(0, 100) < OBSERVE_CHANCE {
                        heap.push(Observe::new(propagate.to, constrained.entropy()))
                    }
                    generate_propagations(&mut to_propagate, &observed, edges, propagate.to);
                    *labels = constrained
                }
            }
            to_propagate = replace(&mut propagations, to_propagate);
        }

        // check if all vertices observed, if so we have finished
        if observed_counter == vertices_len {
            if METRICS { metrics.print(Some("All Observed")) }
            return vertices;
        }

        // try to find a vertex index to observe
        let mut observe_index: Option<VertexIndex> = None;
        // check the heap first
        while !heap.is_empty() {
            let observe = heap.pop().unwrap();
            if !observed.contains(observe.index as usize) {
                observe_index = Some(observe.index);
                break;
            }
        }
        // if no index to check in heap, check vec of initial vertices to observe
        if observe_index.is_none() {
            while !to_observe.is_empty() {
                let index = to_observe.pop().unwrap();
                if !observed.contains(index as usize) {
                    observe_index = Some(index);
                    break;
                }
            }
        }
        match observe_index {
            None => {
                if METRICS { metrics.print(Some("All Observed")) }
                // Nothing left to observe, therefore we've finished
                return vertices;
            }
            Some(index) => {
                if METRICS { metrics.inc("obs") }

                assert!(vertices.len() >= index as usize);
                let labels_multiset = vertices.index_mut(index as usize);
                labels_multiset.choose(rng);
                observed.insert(index as usize);
                generate_propagations(&mut propagations, &observed, edges, index);
            }
        }
    }
}

pub fn build_constraint<S, L, D>(labels: &S, direction: EdgeDirection, rules: &GenericArray<Option<S>, Prod<L, D>>) -> S
where
    S: Multiset,
    L: std::ops::Mul<D> + Unsigned,
    <L as std::ops::Mul<D>>::Output: ArrayLength<Option<S>>,
{
    labels
        .iter_m()
        .enumerate()
        .fold(S::empty(labels.num_elems()), |mut acc, (label, frequency)| {
            if frequency > &Zero::zero() {
                let arr_index = direction as usize * L::to_usize() + label;
                assert!(arr_index <= rules.len());
                if let Some(a) = rules.index(arr_index) {
                    acc = acc.union(a)
                }
            }
            acc
        })
}

fn generate_propagations(
    propagations: &mut Vec<Propagate>,
    observed: &BitSet,
    edges: &Edges,
    from_index: VertexIndex,
) {
    assert!(from_index as usize <= edges.len());
    for (to_index, direction) in edges.index(&from_index) {
        if !observed.contains(*to_index as usize) {
            propagations.push(Propagate::new(from_index, *to_index, *direction))
        }
    }
}

pub fn collapse<S: Multiset>(
    input_graph: &Graph<S>,
    mut output_graph: Graph<S>,
    seed: Option<u64>,
) -> Graph<S> {
    let rng = &mut StdRng::seed_from_u64(seed.unwrap_or_else(|| thread_rng().next_u64()));
    let rules = &input_graph.rules();
    let init = init_collapse(rng, &output_graph);

    let collapsed_vertices = exec_collapse(
        rng,
        rules,
        &output_graph.edges,
        init.clone(),
        output_graph.vertices,
    );
    output_graph.vertices = collapsed_vertices;
    output_graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph::Edges;
    use crate::utils::hash_map;
    use nalgebra::{VectorN, U2, U6};

    type MS6 = VectorN<u16, U6>;
    type MS2 = VectorN<u16, U2>;

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
    fn simple_vertices() -> Vec<MS6> {
        vec![
            MS6::from_row_slice_u(&[1, 0, 0]),
            MS6::from_row_slice_u(&[0, 2, 0]),
            MS6::from_row_slice_u(&[0, 0, 1]),
            MS6::from_row_slice_u(&[0, 2, 0]),
        ]
    }

    //noinspection DuplicatedCode
    fn simple_rules() -> Rules<MS6> {
        hash_map(&[
            ((0, 0), MS6::from_row_slice_u(&[0, 2, 0])),
            ((0, 1), MS6::from_row_slice_u(&[0, 0, 1])),
            ((1, 1), MS6::from_row_slice_u(&[1, 0, 0])),
            ((1, 2), MS6::from_row_slice_u(&[0, 2, 0])),
            ((2, 0), MS6::from_row_slice_u(&[0, 2, 0])),
            ((2, 1), MS6::from_row_slice_u(&[0, 0, 1])),
            ((3, 1), MS6::from_row_slice_u(&[1, 0, 0])),
            ((3, 2), MS6::from_row_slice_u(&[0, 2, 0])),
        ])
    }

    #[test]
    fn test_constraint() {
        let a: MS6 = MS6::from_row_slice_u(&[1, 0, 0, 0, 0, 0]);
        let b: MS6 = MS6::from_row_slice_u(&[0, 0, 1, 0, 0, 0]);
        let c: MS6 = MS6::from_row_slice_u(&[1, 1, 1, 0, 0, 0]);
        let rules: Rules<MS6> = hash_map(&[((0, 0), a), ((0, 1), b), ((0, 2), c)]);
        let stack_rules = rules_stack::<MS6, <U6 as DimName>::Value, <U8 as DimName>::Value>(&rules);

        let labels: &MS6 = &MS6::from_row_slice_u(&[2, 4, 0, 0, 0, 0]);
        let direction: EdgeDirection = 0;

        let result = build_constraint::<MS6, <U6 as DimName>::Value, <U8 as DimName>::Value>(labels, direction, &stack_rules);
        let expected: MS6 = MS6::from_row_slice_u(&[1, 0, 1, 0, 0, 0]);
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
        let all_labels = MS6::from_row_slice_u(&[1, 2, 1]);
        let out_graph = Graph::<MS6>::new(simple_vertices(), edges, all_labels);
        let rules: &Rules<MS6> = &simple_rules();

        let init = init_collapse::<MS6>(&mut rng, &out_graph);

        let result = exec_collapse::<MS6>(&mut rng, rules, &out_graph.edges, init, simple_vertices());
        let expected: Vec<MS6> = vec![
            MS6::from_row_slice_u(&[1, 0, 0]),
            MS6::from_row_slice_u(&[0, 2, 0]),
            MS6::from_row_slice_u(&[0, 0, 1]),
            MS6::from_row_slice_u(&[0, 2, 0]),
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
        let mut rng = StdRng::seed_from_u64(23456);

        let edges = hash_map(&[
            (0, vec![(3, 1), (1, 2)]),
            (1, vec![(0, 3), (4, 1), (2, 2)]),
            (2, vec![(1, 3), (5, 1)]),
            (3, vec![(0, 0), (4, 2)]),
            (4, vec![(3, 3), (1, 0), (5, 2)]),
            (5, vec![(4, 3), (2, 0)]),
        ]);

        let vertices: Vec<MS2> = vec![
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
        ];

        let rules: Rules<MS2> = hash_map(&[
            ((0, 0), MS2::from_row_slice_u(&[3, 3])),
            ((0, 1), MS2::from_row_slice_u(&[3, 3])),
            ((1, 0), MS2::from_row_slice_u(&[3, 3])),
            ((1, 1), MS2::from_row_slice_u(&[3, 3])),
            ((2, 0), MS2::from_row_slice_u(&[3, 3])),
            ((2, 1), MS2::from_row_slice_u(&[3, 3])),
            ((3, 0), MS2::from_row_slice_u(&[3, 3])),
            ((3, 1), MS2::from_row_slice_u(&[3, 3])),
        ]);

        let all_labels = MS2::from_row_slice_u(&[3, 3]);
        let out_graph = Graph::<MS2>::new(vertices, edges, all_labels);
        let init = init_collapse::<MS2>(&mut rng, &out_graph);

        let result = exec_collapse::<MS2>(
            &mut rng,
            &rules,
            &out_graph.edges,
            init,
            out_graph.vertices.clone(),
        );

        let expected: Vec<MS2> = vec![
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[0, 3]),
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

        let mut rng = StdRng::seed_from_u64(4);

        let input_edges = hash_map(&[
            (0, vec![(1, 2), (4, 1)]),
            (1, vec![(0, 3), (5, 1), (2, 2)]),
            (2, vec![(1, 3)]),
            (3, vec![(4, 2)]),
            (4, vec![(3, 3), (0, 0), (5, 2)]),
            (5, vec![(4, 3), (1, 0)]),
        ]);

        let input_vertices: Vec<MS2> = vec![
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[0, 3]),
        ];

        let all_labels: MS2 = MS2::from_row_slice_u(&[3, 3]);

        let input_graph = Graph::<MS2>::new(input_vertices, input_edges, all_labels);

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

        let output_vertices: Vec<MS2> = vec![
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
            MS2::from_row_slice_u(&[3, 3]),
        ];

        // should be same as all labels but Multiset.clone() is outputs false
        // positive errors, so recreating here to avoid.
        let out_all_labels: MS2 = MS2::from_row_slice_u(&[3, 3]);

        let output_graph = Graph::<MS2>::new(output_vertices, output_edges, out_all_labels);
        let init = init_collapse::<MS2>(&mut rng, &output_graph);

        let result = exec_collapse::<MS2>(
            &mut rng,
            &rules,
            &output_graph.edges,
            init,
            output_graph.vertices.clone(),
        );

        let expected: Vec<MS2> = vec![
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[3, 0]),
            MS2::from_row_slice_u(&[0, 3]),
            MS2::from_row_slice_u(&[0, 3]),
        ];

        assert_eq!(result, expected);
    }
}
