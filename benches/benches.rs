#[macro_use]
extern crate bencher;
extern crate nalgebra;


mod collapse {
    use bencher::Bencher;
    use wfc_rust::wfc::collapse::collapse;
    use wfc_rust::io::text_parser::{parse, make_nsew_grid_edges};
    use wfc_rust::graph::graph::{Labels, Graph};


    pub fn bench_collapse(bench: &mut Bencher) {
        let out_width = 10;
        let out_depth = 10;

        if let Ok((input_graph, _)) = parse("resources/test/tosashimizu_model.txt") {
            let all_labels = input_graph.all_labels();
            let output_vertices: Vec<Labels> = vec![all_labels; out_width * out_depth];
            let output_edges = make_nsew_grid_edges(out_width, out_depth);
            let output_graph = Graph::new(output_vertices, output_edges);

            bench.iter(|| { collapse(&input_graph, output_graph.clone(), None, None) })
        }
    }
}

mod nalgebra_test {
    use bencher::Bencher;
    use nalgebra::{inf, DVector};


    fn make_vecs() -> (DVector<u32>, DVector<u32>) {
        let a = DVector::from_iterator(5, vec![1, 0, 1, 1, 0].into_iter());
        let b = DVector::from_iterator(5, vec![0, 0, 1, 1, 0].into_iter());
        (a, b)
    }


    pub fn vector_inf(bench: &mut Bencher) {
        let (a, b) = make_vecs();
        bench.iter(|| {
            inf(&a, &b)
        })
    }


    pub fn vector_wise_mul(bench: &mut Bencher) {
        let (a, b) = make_vecs();
        bench.iter(|| {
            a.component_mul(&b)
        })
    }
}

mod entropy_cache {
    use bencher::Bencher;
    use wfc_rust::wfc::observe::*;
    use wfc_rust::utils::{hash_set, hash_map};
    use nalgebra::DVector;

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

    pub fn uncached_8(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28), (5, 99), (6, 11), (7, 76), (8, 43)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_7(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5, 6, 7]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28), (5, 99), (6, 11), (7, 76)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_6(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5, 6]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28), (5, 99), (6, 11)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_5(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4, 5]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28), (5, 99)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_4(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3, 4]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33), (4, 28)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_3(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2, 3]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100), (3, 33)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_2(bench: &mut Bencher) {
        let test_labels = hash_set(&[1, 2]);
        let test_frequencies = hash_map(&[(1, 200), (2, 100)]);
        bench.iter(|| {
            calculate_entropy(&test_labels, &test_frequencies)
        })
    }

    pub fn uncached_1(bench: &mut Bencher) {
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

    pub fn cached_ent2_8(bench: &mut Bencher) {
        let a: DVector<u32> = DVector::from_row_slice(&[200, 100, 33, 28, 99, 11, 76, 43]);
        bench.iter(|| {
            calculate_entropy2(a.clone())
        })
    }

    pub fn cached_ent3_8(bench: &mut Bencher) {
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

    pub fn cached_ent3_zeroes_8(bench: &mut Bencher) {
        let a: &DVector<u32> = &DVector::from_row_slice(&[200, 0, 0, 0, 0, 0, 0, 0]);
        bench.iter(|| {
            calculate_entropy3(a)
        })
    }
}


benchmark_group!(
    benches,
    // collapse::bench_collapse,
    // nalgebra::vector_inf,
    // nalgebra_test::vector_wise_mul,
    entropy_cache::uncached_16,
    entropy_cache::uncached_8,
    entropy_cache::uncached_7,
    entropy_cache::uncached_6,
    entropy_cache::uncached_5,
    entropy_cache::uncached_4,
    entropy_cache::uncached_3,
    entropy_cache::uncached_2,
    entropy_cache::uncached_1,
    entropy_cache::cached_ent2_8,
    entropy_cache::cached_ent3_8,
    entropy_cache::cached_ent3_16,
    entropy_cache::uncached_zeros_8,
    entropy_cache::cached_ent3_zeroes_8
);
benchmark_main!(benches);
