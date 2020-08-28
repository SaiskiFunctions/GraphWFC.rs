use nalgebra::{U6, Dim, DimName, DefaultAllocator};
use wfc_rust::wfc::collapse::collapse;
use wfc_rust::graph::graph::Graph;
use wfc_rust::io::text_parser::{parse, make_edges_cardinal_grid, render};
use wfc_rust::multiset::{Multiset, MultisetScalar};
use nalgebra::allocator::Allocator;


fn run_collapse<D>(input: &str, output: &str, width: usize, depth: usize)
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    if let Ok((input_graph, keys)) = parse::<D>(input) {
        let all_labels = input_graph.all_labels.clone();
        let output_vertices: Vec<Multiset<D>> = vec![all_labels.clone(); width * depth];
        let output_edges = make_edges_cardinal_grid(width, depth);
        let output_graph = Graph::<D>::new(output_vertices, output_edges, all_labels);
        if let Some(collapsed_graph) = collapse::<D>(&input_graph, output_graph, Some(134522), None) {
            render::<D>(output, &collapsed_graph, &keys, width);
        }
    }
}

// BUG: out_width is divded by 2 when rendered
fn main() {
    let input = "resources/test/tosashimizu_model.txt";
    let output = "resources/test/tosashimizu_output3.txt";
    let out_width = 20;
    let out_depth = 10;

    run_collapse::<U6>(input, output, out_width, out_depth);
}
