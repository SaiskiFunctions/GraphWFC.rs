use rand::prelude::*;
use rand::thread_rng;
use rand::seq::SliceRandom;
use std::collections::{BinaryHeap};
use hashbrown::{HashSet, HashMap};
use std::ops::{Index, IndexMut};
use crate::graph::graph::{VertexIndex, Rules, Graph, Edges};
use crate::wfc::observe::Observe;
use crate::wfc::propagate::Propagate;
use crate::multiset::{MultisetTrait, Multiset, MultisetScalar};
use nalgebra::{Dim, DimName, DefaultAllocator};
use nalgebra::allocator::Allocator;


fn init_collapse<D>(
    rng: &mut StdRng,
    out_graph: &mut Graph<D>,
) -> (HashSet<VertexIndex>, Vec<Propagate>, Vec<VertexIndex>, BinaryHeap<Observe>)
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    let mut observed: HashSet<VertexIndex> = HashSet::new();
    let mut propagations: Vec<Propagate> = Vec::new();
    let mut init_propagations: Vec<u32> = Vec::new();
    let mut heap: BinaryHeap<Observe> = BinaryHeap::new();

    // initialize heap and observed
    out_graph.vertices.iter().enumerate().for_each(|(_index, labels)| {
        assert!(labels.is_subset(&out_graph.all_labels));
        let from_index = _index as u32;
        if labels.is_singleton() { observed.insert(from_index); }
        else if labels != &out_graph.all_labels {
            init_propagations.push(from_index);
            heap.push(Observe::new(&from_index, labels.entropy()))
        }
    });

    // Ensure that output graph is fully propagated before collapse.
    init_propagations.drain(..).for_each(|index| {
        generate_propagations(&mut propagations, &observed, &out_graph.edges, &index);
    });

    let mut to_observe: Vec<VertexIndex> = (0..out_graph.vertices.len() as u32).collect();
    to_observe.shuffle(rng);

    (observed, propagations, to_observe, heap)
}

fn exec_collapse<D>(
    rng: &mut StdRng,
    rules: &Rules<D>,
    edges: &Edges,
    observed: &mut HashSet<VertexIndex>,
    propagations: &mut Vec<Propagate>,
    to_observe: &mut Vec<VertexIndex>,
    heap: &mut BinaryHeap<Observe>,
    vertices: &mut Vec<Multiset<D>>,
) -> Option<Vec<Multiset<D>>>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    let gen_observe: &mut HashSet<VertexIndex> = &mut HashSet::new();
    // let mut vertices_map: HashMap<usize, Multiset<D>> = HashMap::new();

    loop {
        if let Some(propagate) = propagations.pop() {
            assert!(vertices.len() >= propagate.from as usize);

            let prop_labels = vertices.index(propagate.from as usize);
            let constraint = build_constraint(prop_labels, &propagate.direction, rules);
            let labels = vertices.index_mut(propagate.to as usize);
            let constrained = labels.intersection(&constraint);
            if labels != &constrained {
                if constrained.is_empty() { return false; }
                else if constrained.is_singleton() { observed.insert(propagate.to); }
                else { gen_observe.insert(propagate.to); }
                generate_propagations(propagations, &observed, edges, &propagate.to);
                *labels = constrained
            }
        } else {
            gen_observe.drain().for_each(|index| {
                assert!(vertices.len() >= index as usize);
                let labels = vertices.index(index as usize);
                heap.push(Observe::new(&index, labels.entropy()))
            });

            let observe_index = if let Some(ob) = heap.pop() {
                ob.index
            } else {
                let mut _index: u32;
                loop {
                    if let Some(new) = to_observe.pop() {
                        if observed.contains(&new) { continue; }
                        _index = new;
                        break;
                    } else { return true; }
                }
                _index
            };

            if observed.contains(&observe_index) { continue; }
            let labels_multiset = vertices.index_mut(observe_index as usize);
            labels_multiset.choose(rng);
            observed.insert(observe_index);
            generate_propagations(propagations, &observed, edges, &observe_index);
        }
    }
}

pub fn build_constraint<D>(labels: &Multiset<D>, direction: &u16, rules: &Rules<D>) -> Multiset<D>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    labels.iter().enumerate()
        .fold(Multiset::<D>::zeros(), |mut acc, (index, frq)| {
            if frq > &0 {
                if let Some(a) = rules.get(&(*direction, index)) {
                    acc = acc.sup(a)
                }
            }
            acc
        })
}

fn generate_propagations(propagations: &mut Vec<Propagate>, observed: &HashSet<VertexIndex>, edges: &Edges, from_index: &VertexIndex) {
    assert!(edges.len() >= *from_index as usize);
    edges.index(from_index).iter().for_each(|(to_index, direction)| {
        if !observed.contains(to_index) {
            propagations.push(Propagate::new(*from_index, *to_index, *direction))
        }
    });
}

pub fn collapse<D>(input_graph: &Graph<D>, mut output_graph: Graph<D>, seed: Option<u64>) -> Option<Graph<D>>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    let rng = &mut StdRng::seed_from_u64(seed.unwrap_or_else(|| {
        thread_rng().next_u64()
    }));

    let rules = &input_graph.rules();
    let (mut observed, mut propagations, mut to_observe, mut heap) = init_collapse(rng, &mut output_graph);

    let out_vertices = &mut output_graph.vertices;

    if exec_collapse(rng,
                     rules,
                     &output_graph.edges,
                     &mut observed,
                     &mut propagations,
                     &mut to_observe,
                     &mut heap,
                     out_vertices) {
        Some(output_graph)
    } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::graph::Edges;
    use crate::utils::hash_map;
    use nalgebra::{U6, U2};

    //noinspection DuplicatedCode
    fn simple_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)])
        ])
    }

    //noinspection DuplicatedCode
    fn simple_vertices() -> Vec<Multiset<U6>> {
        vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0])
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
        let rules: Rules<U6> = hash_map(&[
            ((0, 0), a),
            ((0, 1), b),
            ((0, 2), c)
        ]);

        let labels: &Multiset<U6> = &Multiset::from_row_slice_u(&[2, 4, 0, 0, 0, 0]);
        let direction: &u16 = &0;

        let result = build_constraint(labels, direction, &rules);
        let expected: Multiset<U6> = Multiset::from_row_slice_u(&[1, 0, 1, 0, 0, 0]);
        assert_eq!(result, expected)
    }

    //noinspection DuplicatedCode
    // #[test]
    // fn test_exec_simple() {
    //     /* Seed Values:
    //         3 -> Some([{0}, {1}, {2}, {1}])
    //         1 -> None
    //     */
    //     let mut rng = StdRng::seed_from_u64(3);
    //
    //     let edges = simple_edges();
    //     let all_labels = Multiset::<U6>::from_row_slice_u(&[1, 2, 1]);
    //     let mut out_graph = Graph::<U6>::new(simple_vertices(), edges.clone(), all_labels);
    //     let rules: &Rules<U6> = &simple_rules();
    //
    //     let result = exec_collapse::<U6>(&mut rng, rules, &all_labels, &edges, &mut out_graph).unwrap();
    //     let expected: Vec<Multiset<U6>> = vec![
    //         Multiset::from_row_slice_u(&[1, 0, 0]),
    //         Multiset::from_row_slice_u(&[0, 2, 0]),
    //         Multiset::from_row_slice_u(&[0, 0, 1]),
    //         Multiset::from_row_slice_u(&[0, 2, 0])
    //     ];
    //
    //     assert_eq!(result.vertices, expected);
    // }
    //
    // #[test]
    // fn test_exec_med() {
    //     /* Seed Values:
    //         INPUT:
    //         0b --- 1b --- 2a
    //         |      |      |
    //         3a --- 4b --- 5a
    //
    //         Output structure same as input structure.
    //         North = 0, South = 1, East = 2, West = 3
    //     */
    //     let mut rng = StdRng::seed_from_u64(234);
    //
    //     let edges = hash_map(&[
    //         (0, vec![(3, 1), (1, 2)]),
    //         (1, vec![(0, 3), (4, 1), (2, 2)]),
    //         (2, vec![(1, 3), (5, 1)]),
    //         (3, vec![(0, 0), (4, 2)]),
    //         (4, vec![(3, 3), (1, 0), (5, 2)]),
    //         (5, vec![(4, 3), (2, 0)])
    //     ]);
    //
    //     let vertices: Vec<Multiset<U2>> = vec![
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //     ];
    //
    //     let rules: Rules<U2> = hash_map(&[
    //         ((0, 0), Multiset::from_row_slice_u(&[3, 3])),
    //         ((0, 1), Multiset::from_row_slice_u(&[3, 3])),
    //         ((1, 0), Multiset::from_row_slice_u(&[3, 3])),
    //         ((1, 1), Multiset::from_row_slice_u(&[3, 3])),
    //         ((2, 0), Multiset::from_row_slice_u(&[3, 3])),
    //         ((2, 1), Multiset::from_row_slice_u(&[3, 3])),
    //         ((3, 0), Multiset::from_row_slice_u(&[3, 3])),
    //         ((3, 1), Multiset::from_row_slice_u(&[3, 3])),
    //     ]);
    //
    //     let all_labels = Multiset::from_row_slice_u(&[3, 3]);
    //     let out_graph = &mut Graph::<U2>::new(vertices, edges.clone(), all_labels);
    //
    //     let result = exec_collapse::<U2>(&mut rng,
    //                                      &rules,
    //                                      &all_labels,
    //                                      &edges,
    //                                      out_graph).unwrap();
    //
    //     let expected: Vec<Multiset<U2>> = vec![
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //     ];
    //
    //
    //     assert_eq!(result.vertices, expected);
    // }
    //
    // #[test]
    // fn test_exec_complex() {
    //     /*
    //         INPUT graph:
    //                0a --- 1b --- 2b
    //                |      |
    //         3a --- 4a --- 5b
    //
    //         Directions: North = 0, South = 1, East = 2, West = 3
    //     */
    //
    //     let mut rng = StdRng::seed_from_u64(1);
    //
    //     let input_edges = hash_map(&[
    //         (0, vec![(1, 2), (4, 1)]),
    //         (1, vec![(0, 3), (5, 1), (2, 2)]),
    //         (2, vec![(1, 3)]),
    //         (3, vec![(4, 2)]),
    //         (4, vec![(3, 3), (0, 0), (5, 2)]),
    //         (5, vec![(4, 3), (1, 0)])
    //     ]);
    //
    //     let input_vertices: Vec<Multiset<U2>> = vec![
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[0, 3])
    //     ];
    //
    //     let all_labels: Multiset<U2> = Multiset::from_row_slice_u(&[3, 3]);
    //
    //     let input_graph = Graph::<U2>::new(input_vertices, input_edges, all_labels);
    //
    //     let rules = input_graph.rules();
    //
    //     /*
    //         OUTPUT STRUCTURE:
    //         0 ---- 1 ---- 2 ---- 3
    //         |      |      |      |
    //         4 ---- 5 ---- 6 ---- 7
    //         |      |      |      |
    //         8 ---- 9 ---- 10 --- 11
    //     */
    //
    //     let output_edges: Edges = hash_map(&[
    //         (0, vec![(1, 2), (4, 1)]),
    //         (1, vec![(0, 3), (5, 1), (2, 2)]),
    //         (2, vec![(1, 3), (6, 1), (3, 2)]),
    //         (3, vec![(2, 3), (7, 1)]),
    //         (4, vec![(0, 0), (8, 1), (5, 2)]),
    //         (5, vec![(4, 3), (1, 0), (6, 2), (9, 1)]),
    //         (6, vec![(5, 3), (2, 0), (7, 2), (10, 1)]),
    //         (7, vec![(6, 3), (3, 0), (11, 1)]),
    //         (8, vec![(4, 0), (9, 2)]),
    //         (9, vec![(8, 3), (5, 0), (10, 2)]),
    //         (10, vec![(9, 3), (6, 0), (11, 2)]),
    //         (11, vec![(10, 3), (7, 0)])
    //     ]);
    //
    //     let output_vertices: Vec<Multiset<U2>> = vec![
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3]),
    //         Multiset::from_row_slice_u(&[3, 3])
    //     ];
    //
    //     // should be same as all labels but Multiset.clone() is outputs false
    //     // positive errors, so recreating here to avoid.
    //     let out_all_labels: Multiset<U2> = Multiset::from_row_slice_u(&[3, 3]);
    //
    //     let output_graph = &mut Graph::<U2>::new(output_vertices, output_edges.clone(), out_all_labels);
    //
    //     let result = exec_collapse::<U2>(&mut rng,
    //                                      &rules,
    //                                      &all_labels,
    //                                      &output_edges,
    //                                      output_graph).unwrap();
    //
    //     let expected: Vec<Multiset<U2>> = vec![
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[3, 0]),
    //         Multiset::from_row_slice_u(&[0, 3]),
    //         Multiset::from_row_slice_u(&[0, 3])
    //     ];
    //
    //     assert_eq!(result.vertices, expected);
    // }
}
