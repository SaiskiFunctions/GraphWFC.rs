#[macro_use]
extern crate bencher;
extern crate nalgebra;


mod collapse {
    use bencher::Bencher;
    use wfc_rust::wfc::collapse::{ConstraintCache, collapse, collapse2};
    use wfc_rust::io::text_parser::{parse, make_nsew_grid_edges, parse2};
    use wfc_rust::graph::graph::{Labels, Graph, Rules2, Graph2};
    use wfc_rust::multiset::Multiset;
    use wfc_rust::utils::hash_map;

    pub fn bench_collapse(bench: &mut Bencher) {
        let out_width = 100;
        let out_depth = 100;

        if let Ok((input_graph, _)) = parse("resources/test/tosashimizu_model.txt") {
            let all_labels = input_graph.all_labels();
            let output_vertices: Vec<Labels> = vec![all_labels; out_width * out_depth];
            let output_edges = make_nsew_grid_edges(out_width, out_depth);
            let output_graph = Graph::new(output_vertices, output_edges);

            bench.iter(|| {
                collapse(&input_graph, output_graph.clone(), None, None)
            })
        }
    }

    pub fn bench_collapse2(bench: &mut Bencher) {
        let out_width = 100;
        let out_depth = 100;

        if let Ok((input_graph, _)) = parse2("resources/test/tosashimizu_model.txt") {
            let all_labels = input_graph.all_labels.clone();
            let output_vertices: Vec<Multiset> = vec![all_labels.clone(); out_width * out_depth];
            let output_edges = make_nsew_grid_edges(out_width, out_depth);
            let output_graph = Graph2::new(output_vertices, output_edges, all_labels.clone());

            bench.iter(|| {
                collapse2(&input_graph, output_graph.clone(), None, None)
            })
        }
    }

    pub fn bench_constraint_build(bench: &mut Bencher) {
        let a: Multiset = Multiset::from_row_slice(&[1, 0, 0]);
        let b: Multiset = Multiset::from_row_slice(&[0, 0, 1]);
        let c: Multiset = Multiset::from_row_slice(&[1, 1, 1]);
        let rules: Rules2 = hash_map(&[
            ((0, 0), a),
            ((0, 1), b),
            ((0, 2), c)
        ]);

        let labels: &Multiset = &Multiset::from_row_slice(&[2, 4, 0]);
        let direction: &u16 = &0;

        let mut constraint_cache = ConstraintCache::new();

        bench.iter(|| {
            constraint_cache.constraint(labels, direction, &rules)
        })
    }

    pub fn bench_clone(bench: &mut Bencher) {
        let all_labels: Multiset = Multiset::from_row_slice(&[10, 22, 3]);
        let length = 10000;
        bench.iter(|| {
            vec![all_labels.clone(); length]
        })
    }

    pub fn bench_ref(bench: &mut Bencher) {
        let all_labels: Multiset = Multiset::from_row_slice(&[10, 22, 3]);
        let length = 10000;
        bench.iter(|| {
            vec![&all_labels; length]
        })
    }
}

mod entropy_cache {
    use bencher::Bencher;
    use wfc_rust::multiset::{Multiset, MultisetTrait};

    pub fn cached_ent3_01(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_02(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200, 100]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_03(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200, 100, 33]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_04(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200, 100, 33, 28]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_05(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200, 100, 33, 28, 99]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_06(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200, 100, 33, 28, 99, 11]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_07(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200, 100, 33, 28, 99, 11, 76]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_08(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200, 100, 33, 28, 99, 11, 76, 43]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_16(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200, 100, 33, 28, 99, 11, 76, 43, 200, 100, 33, 28, 99, 11, 76, 43]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_32(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[
            200, 100, 33, 28, 99, 11, 76, 43, 200, 100, 33, 28, 99, 11, 76, 43,
            200, 100, 33, 28, 99, 11, 76, 43, 200, 100, 33, 28, 99, 11, 76, 43
        ]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_zeroes_08(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[200, 0, 0, 0, 0, 0, 0, 0]);
        bench.iter(|| {
            a.entropy()
        })
    }

    pub fn cached_ent3_zeroes_32(bench: &mut Bencher) {
        let a: &Multiset = &Multiset::from_row_slice(&[
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
    use wfc_rust::multiset::Multiset;
    use std::iter::FromIterator;
    use hashbrown::HashMap;
    use rand::prelude::*;

    fn graph_edges() -> Edges {
        hash_map(&[
            (0, vec![(1, 0), (3, 2)]),
            (1, vec![(0, 1), (2, 2)]),
            (2, vec![(3, 1), (1, 3)]),
            (3, vec![(0, 3), (2, 0)])
        ])
    }

    pub fn graph1_rules(bench: &mut Bencher) {
        let graph_vertices: Vec<Labels> = Vec::from_iter(
            [0, 1, 2, 1].iter().map(|n: &i32| hash_set(&[*n]))
        );

        let graph = Graph::new(graph_vertices, graph_edges());

        bench.iter(|| {
            graph.rules()
        })
    }

    pub fn graph2_rules(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset> = vec![
            Multiset::from_row_slice(&[1, 0, 0]),
            Multiset::from_row_slice(&[0, 2, 0]),
            Multiset::from_row_slice(&[0, 0, 1]),
            Multiset::from_row_slice(&[0, 2, 0])
        ];

        let graph = Graph2::new(graph_vertices, graph_edges(), Multiset::from_row_slice(&[1, 2, 1]));

        bench.iter(|| {
            graph.rules()
        })
    }

    pub fn graph1_observe(bench: &mut Bencher) {
        let graph_vertices: Vec<Labels> = vec![
            hash_set(&[0, 1]),
            hash_set(&[0, 1, 2, 3]),
            hash_set(&[1, 3]),
            hash_set(&[0])
        ];

        let frequencies = hash_map(&[(0, 1), (1, 2), (2, 100), (3, 1)]);
        let graph = Graph::new(graph_vertices, HashMap::new());
        let mut test_rng = StdRng::seed_from_u64(10);
        let index = 1;

        bench.iter(|| {
            graph.clone().observe(&mut test_rng, &index, &frequencies)
        });
    }

    pub fn graph2_observe(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset> = vec![
            Multiset::from_row_slice(&[3, 3, 0, 0]),
            Multiset::from_row_slice(&[3, 3, 1, 2]),
            Multiset::from_row_slice(&[0, 3, 0, 2]),
            Multiset::from_row_slice(&[3, 0, 0, 0])
        ];

        let graph = Graph2::new(graph_vertices, HashMap::new(), Multiset::from_row_slice(&[3, 3, 1, 2]));
        let mut test_rng = StdRng::seed_from_u64(2);
        let index = 1;

        bench.iter(|| {
            graph.clone().observe(&mut test_rng, &index)
        });
    }

    pub fn graph1_constrain_true(bench: &mut Bencher) {
        let graph_vertices: Vec<Labels> = vec![
            hash_set(&[0, 1, 2, 3])
        ];

        let constraint = hash_set(&[0, 1]);
        let graph = Graph::new(graph_vertices, HashMap::new());

        bench.iter(|| {
            graph.clone().constrain(&0, &constraint);
        });
    }

    pub fn graph1_constrain_false(bench: &mut Bencher) {
        let graph_vertices: Vec<Labels> = vec![
            hash_set(&[0, 1])
        ];

        let constraint = hash_set(&[0, 1, 2]);
        let graph = Graph::new(graph_vertices, HashMap::new());

        bench.iter(|| {
            graph.clone().constrain(&0, &constraint);
        });
    }

    pub fn graph2_constrain_true(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset> = vec![
            Multiset::from_row_slice(&[1, 1, 1, 1])
        ];

        let constraint = Multiset::from_row_slice(&[1, 1, 0, 0]);
        let graph = Graph2::new(graph_vertices, HashMap::new(), Multiset::from_row_slice(&[1, 1, 1, 1]));

        bench.iter(|| {
            graph.clone().constrain(&0, &constraint);
        });
    }

    pub fn graph2_constrain_false(bench: &mut Bencher) {
        let graph_vertices: Vec<Multiset> = vec![
            Multiset::from_row_slice(&[1, 1, 0])
        ];

        let constraint = Multiset::from_row_slice(&[1, 1, 1]);
        let graph = Graph2::new(graph_vertices, HashMap::new(), Multiset::from_row_slice(&[1, 1, 1]));

        bench.iter(|| {
            graph.clone().constrain(&0, &constraint);
        });
    }
}


benchmark_group!(
    benches,
    collapse::bench_collapse,
    collapse::bench_collapse2,
    collapse::bench_constraint_build,
    // collapse::bench_clone,
    // collapse::bench_ref,
    // entropy_cache::cached_ent3_01,
    // entropy_cache::cached_ent3_02,
    // entropy_cache::cached_ent3_03,
    // entropy_cache::cached_ent3_04,
    // entropy_cache::cached_ent3_05,
    // entropy_cache::cached_ent3_06,
    // entropy_cache::cached_ent3_07,
    // entropy_cache::cached_ent3_08,
    // entropy_cache::cached_ent3_16,
    // entropy_cache::cached_ent3_32,
    // entropy_cache::cached_ent3_zeroes_08,
    // entropy_cache::cached_ent3_zeroes_32,
    // graphs::graph1_rules,
    // graphs::graph2_rules,
    // graphs::graph1_observe,
    // graphs::graph2_observe,
    // graphs::graph1_constrain_true,
    // graphs::graph2_constrain_true,
    // graphs::graph1_constrain_false,
    // graphs::graph2_constrain_false
);
benchmark_main!(benches);
