#[macro_use]
extern crate bencher;
extern crate nalgebra;

use bencher::Bencher;
use wfc_rust::graph::graph::Graph;
use nalgebra::{inf, DVector};


fn make_vecs() -> (DVector<u32>, DVector<u32>) {
    let a = DVector::from_iterator(5, vec![1, 0, 1, 1, 0].into_iter());
    let b = DVector::from_iterator(5, vec![0, 0, 1, 1, 0].into_iter());
    (a, b)
}


fn vector_inf(bench: &mut Bencher) {
    let (a, b) = make_vecs();
    bench.iter(|| {
        inf(&a, &b)
    })
}


fn vector_wise_mul(bench: &mut Bencher) {
    let (a, b) = make_vecs();
    bench.iter(|| {
        a.component_mul(&b)
    })
}


benchmark_group!(benches, vector_inf, vector_wise_mul);
benchmark_main!(benches);
