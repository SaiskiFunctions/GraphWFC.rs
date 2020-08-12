#[macro_use]
extern crate bencher;
extern crate nalgebra;


mod collapse {
    use bencher::Bencher;
    use wfc_rust::wfc::collapse::collapse;
    use wfc_rust::io::text_parser::{parse, make_nsew_grid_edges};
    use wfc_rust::graph::graph::{Labels, Graph};


    pub fn bench_collapse(bench: &mut Bencher) {
        let out_width = 100;
        let out_depth = 100;

        if let Ok((input_graph, _)) = parse("resources/test/tosashimizu_model.txt") {
            let all_labels = input_graph.all_labels();
            let output_vertices: Vec<Labels> = vec![all_labels; out_width * out_depth];
            let output_edges = make_nsew_grid_edges(out_width, out_depth);
            let output_graph = Graph::new(output_vertices, output_edges);

            bench.iter(|| { collapse(&input_graph, output_graph.clone(), None, None) })
        }
    }
}

mod entropy_cache {
    use bencher::Bencher;
    use wfc_rust::utils::{hash_set, hash_map};
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


benchmark_group!(
    benches,
    // collapse::bench_collapse
    entropy_cache::cached_ent3_01,
    entropy_cache::cached_ent3_02,
    entropy_cache::cached_ent3_03,
    entropy_cache::cached_ent3_04,
    entropy_cache::cached_ent3_05,
    entropy_cache::cached_ent3_06,
    entropy_cache::cached_ent3_07,
    entropy_cache::cached_ent3_08,
    entropy_cache::cached_ent3_16,
    entropy_cache::cached_ent3_32,
    entropy_cache::cached_ent3_zeroes_08,
    entropy_cache::cached_ent3_zeroes_32
);
benchmark_main!(benches);
