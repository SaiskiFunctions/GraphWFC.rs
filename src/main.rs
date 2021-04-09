use wfc_rust::graph::graph::Graph;
use wfc_rust::io::text_parser;
use wfc_rust::io::image_olm_parser;
use wfc_rust::io::utils::{make_edges_cardinal_grid, make_edges_8_way_grid};
use wfc_rust::wfc::collapse::collapse;

// TODO: Save the results of parsing

fn run_tile(input: &str, output: &str, width: usize, depth: usize, intercardinals: bool) {
    if let Ok((input_graph, keys)) = text_parser::parse(input, intercardinals) {
        let all_labels = input_graph.all_labels;
        let output_vertices = vec![all_labels; width * depth];
        let output_edges = if intercardinals {
            make_edges_8_way_grid(width, depth)
        } else {
            make_edges_cardinal_grid(width, depth)
        };
        let output_graph = Graph::new(output_vertices, output_edges, all_labels);
        let collapsed_vertices = collapse(&input_graph.rules(), &output_graph, None, None);
        text_parser::render(output, collapsed_vertices, &keys, width);
    }
}

fn run_olm(input: &str, chunk_size: usize, output: &str, width: usize, depth: usize) {
    if width % chunk_size != 0 || depth % chunk_size != 0 {
        panic!("Output dimensions and N size NOT divisible.");
    }
    let (rules, keys, all_labels, chunks) = image_olm_parser::parse(input, chunk_size);
    let graph_width = width / chunk_size; // in chunks
    let graph_depth = depth / chunk_size; // in chunks
    let output_edges = make_edges_8_way_grid(graph_width, graph_depth);
    let output_vertices = vec![all_labels; graph_width * graph_depth];
    let output_graph = Graph::new(output_vertices, output_edges, all_labels);
    let collapsed_vertices = collapse(&rules, &output_graph, None, None);
    image_olm_parser::render(output, collapsed_vertices, &keys, &chunks, (width, depth), chunk_size as usize);
}

enum RunMode {
    OLM,
    Tile
}

const CHUNK_SIZE: usize = 2;
const MODE: RunMode = RunMode::OLM;

fn main() {
    match MODE {
        RunMode::OLM => {
            let input = "resources/test/City.png";
            let output = "resources/test/test_result_9.png";
            let out_width = 20;
            let out_depth = 20;

            run_olm(input, CHUNK_SIZE, output, out_width, out_depth);
        },
        RunMode::Tile => {
            let input = "resources/test/tosashimizu_model.txt";
            let output = "resources/test/tosashimizu_output3.txt";
            let out_width = 20;
            let out_depth = 20;

            run_tile(input, output, out_width, out_depth, true);
        }
    }
}
