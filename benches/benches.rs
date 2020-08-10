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
    use wfc_rust::wfc::observe::*;
    use wfc_rust::utils::{hash_set, hash_map};
    use nalgebra::DVector;

    pub fn uncached_32(bench: &mut Bencher) {
        let test_labels = hash_set(&[
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
        ]);
        let test_frequencies = hash_map(&[
            (1, 200), (2, 100), (3, 33), (4, 28), (5, 99), (6, 11), (7, 76), (8, 43),
            (9, 200), (10, 100), (11, 33), (12, 28), (13, 99), (14, 11), (15, 76), (16, 43),
            (17, 200), (18, 100), (19, 33), (20, 28), (21, 99), (22, 11), (23, 76), (24, 43),
            (25, 200), (26, 100), (27, 33), (28, 28), (29, 99), (30, 11), (31, 76), (32, 43)
        ]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_16(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let test_frequencies = hash_map(&[
            (1, 200), (2, 100), (3, 33), (4, 28), (5, 99), (6, 11), (7, 76), (8, 43),
            (9, 200), (10, 100), (11, 33), (12, 28), (13, 99), (14, 11), (15, 76), (16, 43)
        ]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_08(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28), (5, 99), (6, 11), (7, 76), (8, 43)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_07(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5, 6, 7]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28), (5, 99), (6, 11), (7, 76)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_06(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5, 6]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28), (5, 99), (6, 11)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_05(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28), (5, 99)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_04(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_03(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_02(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_01(bench: &mut Bencher) {
        let test_labels = hash_set(&[1]);
        let test_frequencies = hash_map(&[(1, 200)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_zeros_8(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5]);
        let test_frequencies = hash_map(&[(1, 200), (2, 0), (3, 0), (4, 0), (5, 0)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn cached_ent2_08(bench: &mut Bencher) {
        let a: DVector<u32> = DVector::from_row_slice(&[200, 100, 33, 28, 99, 11, 76, 43]);
        bench.iter(|| {
            calculate_entropy2(a.clone())
        })
    }

    pub fn cached_ent3_01(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_02(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 100]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_03(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 100, 33]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_04(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 100, 33, 28]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_05(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 100, 33, 28, 99]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_06(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 100, 33, 28, 99, 11]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_07(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 100, 33, 28, 99, 11, 76]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_08(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 100, 33, 28, 99, 11, 76, 43]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_16(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 100, 33, 28, 99, 11, 76, 43, 200, 100, 33, 28, 99, 11, 76, 43]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_32(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[
            200, 100, 33, 28, 99, 11, 76, 43, 200, 100, 33, 28, 99, 11, 76, 43,
            200, 100, 33, 28, 99, 11, 76, 43, 200, 100, 33, 28, 99, 11, 76, 43
        ]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }

    pub fn cached_ent3_zeroes_8(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 0, 0, 0, 0, 0, 0, 0]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }
}


benchmark_group!(
    benches,
    // collapse::bench_collapse
    // entropy_cache::uncached_32,
    // entropy_cache::uncached_16,
    // entropy_cache::uncached_08,
    // entropy_cache::uncached_07,
    // entropy_cache::uncached_06,
    // entropy_cache::uncached_05,
    // entropy_cache::uncached_04,
    // entropy_cache::uncached_03,
    // entropy_cache::uncached_02,
    // entropy_cache::uncached_01,
    // entropy_cache::cached_ent2_08,
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
    // entropy_cache::uncached_zeros_8,
    // entropy_cache::cached_ent3_zeroes_8
);
benchmark_main!(benches);
