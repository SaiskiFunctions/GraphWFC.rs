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

            bench.iter(|| { collapse(&input_graph, output_graph.clone(), Some(134522), None) })
        }
    }
}

mod nalgebra_tests {
    use bencher::Bencher;
    use wfc_rust::graph::graph::Graph;
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


benchmark_group!(
    benches,
    collapse::bench_collapse,
    nalgebra_tests::vector_inf,
    nalgebra_tests::vector_wise_mul
);
benchmark_main!(benches);
