#[macro_use]
extern crate bencher;
extern crate nalgebra;

mod collapse {
    use bencher::Bencher;
    use nalgebra::{U4, VectorN};
    use wfc_rust::graph::graph::Graph;
    use wfc_rust::io::text_parser::parse;
    use wfc_rust::io::utils::{make_edges_cardinal_grid, make_edges_8_way_grid};
    use wfc_rust::wfc::collapse::collapse;

    type MS4 = VectorN<u32, U4>;

    pub fn bench_collapse(bench: &mut Bencher) {
        let out_width = 100;
        let out_depth = 100;

        if let Ok((input_graph, _)) = parse::<MS4>("resources/test/tosashimizu_model.txt") {
            let all_labels = input_graph.all_labels;
            let output_vertices: Vec<MS4> = vec![all_labels; out_width * out_depth];
            let output_edges = make_edges_8_way_grid(out_width, out_depth);
            let output_graph = Graph::new(output_vertices, output_edges, all_labels);

            bench.iter(|| collapse(&input_graph, output_graph.clone(), None))
        }
    }
}

mod graphs {
    use bencher::Bencher;
    use nalgebra::{U6, VectorN};
    use wfc_rust::graph::graph::*;
    use wfc_rust::multiset::Multiset;
    use wfc_rust::utils::hash_map;

    type MS6 = VectorN<u16, U6>;

    fn graph_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)]),
        ])
    }

    pub fn graph_rules(bench: &mut Bencher) {
        let graph_vertices: Vec<MS6> = vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
        ];

        let graph = Graph::new(
            graph_vertices,
            graph_edges(),
            Multiset::from_row_slice_u(&[1, 2, 1]),
        );

        bench.iter(|| graph.rules())
    }
}

mod multiset {
    use bencher::{black_box, Bencher};
    use nalgebra::{U4, VectorN};
    use wfc_rust::multiset::Multiset as MS1;
    use wfc_rust::multiset2::Multiset as MS2;

    type MS1d4u8 = VectorN<u8, U4>;
    type MS2d4u8 = MS2<u8, typenum::U4>;

    pub fn ms1_union(bench: &mut Bencher) {
        let sets: Vec<MS1d4u8> = vec![MS1::from_row_slice_u(&[1, 5, 6, 2]); 1000];
        let mut out: MS1d4u8 = MS1::empty(4);

        bench.iter(|| sets.iter().for_each(|set| {
            out = black_box(out.union(set));
        }));
    }

    pub fn ms2_union(bench: &mut Bencher) {
        let sets: Vec<MS2d4u8> = vec![MS2::from_slice(&[1, 5, 6, 2]); 1000];
        let mut out: MS2d4u8 = MS2::empty();

        bench.iter(|| sets.iter().for_each(|set| {
            out = black_box(out.union(set));
        }));
    }

    pub fn ms1_is_subset(bench: &mut Bencher) {
        let sets: Vec<MS1d4u8> = vec![MS1::from_row_slice_u(&[1, 5, 6, 2]); 1000];
        let out: MS1d4u8 = MS1::empty(4);

        bench.iter(|| sets.iter().for_each(|set| {
            black_box(out.is_subset(set));
        }));
    }

    pub fn ms2_is_subset(bench: &mut Bencher) {
        let sets: Vec<MS2d4u8> = vec![MS2::from_slice(&[1, 5, 6, 2]); 1000];
        let out: MS2d4u8 = MS2::empty();

        bench.iter(|| sets.iter().for_each(|set| {
            black_box(out.is_subset(set));
        }));
    }

    pub fn ms1_contains(bench: &mut Bencher) {
        let indices: Vec<usize> = vec![2; 1000];
        let out: MS1d4u8 = MS1::from_row_slice_u(&[1, 5, 6, 2]);

        bench.iter(|| indices.iter().for_each(|index| {
            black_box(out.contains(*index));
        }));
    }

    pub fn ms2_contains(bench: &mut Bencher) {
        let indices: Vec<usize> = vec![2; 1000];
        let out: MS2d4u8 = MS2::from_slice(&[1, 5, 6, 2]);

        bench.iter(|| indices.iter().for_each(|index| {
            // unsafe { black_box(out.contains_unchecked(*index)) };
            black_box(out.contains(*index));
        }));
    }

    pub fn ms2_shannon(bench: &mut Bencher) {
        let sets: Vec<MS2d4u8> = vec![MS2::from_slice(&[1, 5, 6, 2]); 1000];
        let mut out: f64 = 0.0;

        bench.iter(|| sets.iter().for_each(|set| {
            out = black_box(set.shannon_entropy());
        }));
    }

    pub fn ms2_collision(bench: &mut Bencher) {
        let sets: Vec<MS2d4u8> = vec![MS2::from_slice(&[1, 5, 6, 2]); 1000];
        let mut out: f64 = 0.0;

        bench.iter(|| sets.iter().for_each(|set| {
            out = black_box(set.collision_entropy());
        }));
    }
}

benchmark_group!(
    benches,
    // collapse::bench_collapse,
    // graphs::graph_rules,
    // multiset::ms1_union,
    // multiset::ms2_union,
    // multiset::ms1_is_subset,
    // multiset::ms2_is_subset,
    // multiset::ms1_contains,
    // multiset::ms2_contains,
    multiset::ms2_shannon,
    multiset::ms2_collision,
);
benchmark_main!(benches);
