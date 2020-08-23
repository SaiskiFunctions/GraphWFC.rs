#[macro_use]
extern crate bencher;
extern crate nalgebra;


mod collapse {
    use bencher::Bencher;
    use wfc_rust::wfc::collapse::collapse;
    use wfc_rust::io::text_parser::{parse, make_nsew_grid_edges};
    use wfc_rust::graph::graph::{Graph, Rules};
    use wfc_rust::multiset::{Multiset, MultisetTrait};
    use wfc_rust::utils::hash_map;
    use nalgebra::U6;

    pub fn bench_collapse(bench: &mut Bencher) {
        let out_width = 100;
        let out_depth = 100;

        if let Ok((input_graph, _)) = parse::<U6>("resources/test/tosashimizu_model.txt") {
            let all_labels = input_graph.all_labels.clone();
            let output_vertices: Vec<Multiset<U6>> = vec![all_labels.clone(); out_width * out_depth];
            let output_edges = make_nsew_grid_edges(out_width, out_depth);
            let output_graph = Graph::<U6>::new(output_vertices, output_edges, all_labels.clone());

            bench.iter(|| {
                collapse::<U6>(&input_graph, output_graph.clone(), None, None)
            })
        }
    }
}

mod entropy_cache {
    use bencher::Bencher;
    use wfc_rust::multiset::{Multiset, MultisetTrait};
    use nalgebra::U6;

    pub fn cached_ent_01(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_02(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 100]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_03(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 100, 33]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_04(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 100, 33, 28]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_05(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 100, 33, 28, 99]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_06(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 100, 33, 28, 99, 11]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_07(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 100, 33, 28, 99, 11, 76]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_08(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 100, 33, 28, 99, 11, 76, 43]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_16(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 100, 33, 28, 99, 11, 76, 43, 200, 100, 33, 28, 99, 11, 76, 43]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_32(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[
            200, 100, 33, 28, 99, 11, 76, 43, 200, 100, 33, 28, 99, 11, 76, 43,
            200, 100, 33, 28, 99, 11, 76, 43, 200, 100, 33, 28, 99, 11, 76, 43
        ]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_zeroes_08(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[200, 0, 0, 0, 0, 0, 0, 0]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent_zeroes_32(bench: &mut Bencher) {
        let a: &Multiset<U6> = &Multiset::from_row_slice_u(&[
            200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        ]);
        bench.iter(|| {
            a.entropy()
        })
    }
}

mod graphs {
    use bencher::Bencher;
    use wfc_rust::graph::graph::*;
    use wfc_rust::utils::{hash_map, hash_set};
    use wfc_rust::multiset::{Multiset, MultisetTrait};
    use std::iter::FromIterator;
    use hashbrown::HashMap;
    use rand::prelude::*;
    use nalgebra::U6;

    fn graph_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)])
        ])
    }

    pub fn graph_rules(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 0, 0]),
            Multiset::from_row_slice_u(&[0, 2, 0]),
            Multiset::from_row_slice_u(&[0, 0, 1]),
            Multiset::from_row_slice_u(&[0, 2, 0])
        ];

        let graph = Graph::<U6>::new(graph_vertices, graph_edges(), Multiset::from_row_slice_u(&[1, 2, 1]));

        bench.iter(|| {
            graph.rules()
        })
    }

    pub fn graph_observe(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[3, 3, 0, 0]),
            Multiset::from_row_slice_u(&[3, 3, 1, 2]),
            Multiset::from_row_slice_u(&[0, 3, 0, 2]),
            Multiset::from_row_slice_u(&[3, 0, 0, 0])
        ];

        let graph = Graph::<U6>::new(graph_vertices, HashMap::new(), Multiset::from_row_slice_u(&[3, 3, 1, 2]));
        let mut test_rng = StdRng::seed_from_u64(2);
        let index = 1;

        bench.iter(|| {
            graph.clone().observe(&mut test_rng, &index)
        });
    }

    pub fn graph_constrain_true(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 1, 1, 1])
        ];

        let constraint = Multiset::from_row_slice_u(&[1, 1, 0, 0]);
        let graph = Graph::<U6>::new(graph_vertices, HashMap::new(), Multiset::from_row_slice_u(&[1, 1, 1, 1]));

        bench.iter(|| {
            graph.clone().constrain(&0, &constraint);
        });
    }

    pub fn graph_constrain_false(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset<U6>> = vec![
            Multiset::from_row_slice_u(&[1, 1, 0])
        ];

        let constraint = Multiset::from_row_slice_u(&[1, 1, 1]);
        let graph = Graph::<U6>::new(graph_vertices, HashMap::new(), Multiset::from_row_slice_u(&[1, 1, 1]));

        bench.iter(|| {
            graph.clone().constrain(&0, &constraint);
        });
    }
}


benchmark_group!(
    benches,
    collapse::bench_collapse,
    // entropy_cache::cached_ent_01,
    // entropy_cache::cached_ent_02,
    // entropy_cache::cached_ent_03,
    // entropy_cache::cached_ent_04,
    // entropy_cache::cached_ent_05,
    // entropy_cache::cached_ent_06,
    // entropy_cache::cached_ent_07,
    // entropy_cache::cached_ent_08,
    // entropy_cache::cached_ent_16,
    // entropy_cache::cached_ent_32,
    // entropy_cache::cached_ent_zeroes_08,
    // entropy_cache::cached_ent_zeroes_32,
    // graphs::graph_rules,
    // graphs::graph_observe,
    // graphs::graph_constrain_true,
    // graphs::graph_constrain_false
);
benchmark_main!(benches);
