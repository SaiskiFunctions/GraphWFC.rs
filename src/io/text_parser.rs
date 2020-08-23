use hashbrown::HashMap;
use std::fs::{read_to_string, write};
use std::io::Error;
use std::ops::Index;
use crate::graph::graph::{Edges, EdgeDirection, VertexIndex, Graph};
use crate::multiset::{Multiset, MultisetTrait, MultisetScalar};
use nalgebra::{Dim, DimName, DefaultAllocator};
use nalgebra::allocator::Allocator;


pub fn parse<D>(filename: &str) -> Result<(Graph<D>, HashMap<usize, char>), Error>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    read_to_string(filename).map(|string| {
        let lines: Vec<&str> = string.split('\n').filter(|l| !l.is_empty()).collect();
        let num_lines = lines.len();
        let line_len = lines.index(0).chars().count();

        let mut edges: Edges = HashMap::new();
        let mut key_frequency_map: HashMap<char, u32> = HashMap::new();

        lines.iter().enumerate().for_each(|(line_index, line)| {
            line.chars().enumerate().for_each(|(character_index, character)| {
                key_frequency_map.entry(character).and_modify(|freq| *freq += 1).or_insert(1);

                let mut direction_pairs = Vec::new();
                //NORTH = 0
                if line_index > 0 {
                    direction_pairs.push(((line_index - 1) * line_len + character_index, 0))
                }
                //SOUTH = 1
                if line_index < num_lines - 1 {
                    direction_pairs.push(((line_index + 1) * line_len + character_index, 1))
                }
                //WEST = 3
                if character_index > 0 {
                    direction_pairs.push(((character_index - 1) + line_index * line_len, 3))
                }
                //EAST = 2
                if character_index < line_len - 1 {
                    direction_pairs.push(((character_index + 1) + line_index * line_len, 2))
                }

                let direction_pairs = direction_pairs.into_iter().map(|(i, d)| {
                    (i as VertexIndex, d as EdgeDirection)
                }).collect::<Vec<(VertexIndex, EdgeDirection)>>();

                let this_vertex_index = ((line_index * line_len) + character_index) as VertexIndex;
                edges.insert(this_vertex_index, direction_pairs);
            });
        });

        let char_keys: HashMap<usize, char> = key_frequency_map.keys().copied().enumerate().collect();

        let all_labels_vec: Vec<u32> = (0..char_keys.len()).map(|index| {
            *key_frequency_map.index(char_keys.index(&index))
        }).collect();

        let all_labels: Multiset<D> = Multiset::from_iter_u(all_labels_vec.clone());

        let vertices: Vec<Multiset<D>> = string.chars().filter(|c| c != &'\n').map(|c| {
            Multiset::from_iter_u((0..all_labels_vec.len()).map(|index| {
                let char = char_keys.index(&index);
                if char == &c { *key_frequency_map.index(&c) } else { 0 }
            }))
        }).collect();

        (Graph::new(vertices, edges, all_labels), char_keys)
    })
}

pub fn make_nsew_grid_edges(width: usize, depth: usize) -> Edges {
    let mut edges: Edges = HashMap::new();
    for depth_index in 0..depth {
        for width_index in 0..width {

            let mut direction_pairs = Vec::new();
            //NORTH = 0
            if depth_index > 0 {
                direction_pairs.push(((depth_index - 1) * width + width_index, 0))
            }
            //SOUTH = 1
            if depth_index < depth - 1 {
                direction_pairs.push(((depth_index + 1) * width + width_index, 1))
            }
            //WEST = 3
            if width_index > 0 {
                direction_pairs.push(((width_index - 1) + depth_index * width, 3))
            }
            //EAST = 2
            if width_index < width - 1 {
                direction_pairs.push(((width_index + 1) + depth_index * width, 2))
            }

            let direction_pairs = direction_pairs.into_iter().map(|(i, d)| {
                (i as VertexIndex, d as EdgeDirection)
            }).collect::<Vec<(VertexIndex, EdgeDirection)>>();

            let this_vertex_index = ((depth_index * width) + width_index) as VertexIndex;
            edges.insert(this_vertex_index, direction_pairs);
        };
    };
    edges
}

pub fn render<D>(filename: &str, graph: &Graph<D>, key: &HashMap<usize, char>, width: usize)
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    let rendered_vertices: Vec<char> = graph.vertices.iter().map(|labels| {
        let k = labels.get_non_zero().unwrap();
        *key.index(&k)
    }).collect();
    let lines: String = rendered_vertices.chunks_exact(width).map(|chunk| {
        chunk.iter().collect::<String>() + "\n"
    }).collect::<String>();
    if let Ok(_) = write(filename, lines) {};
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::U6;

    #[test]
    fn test_parse_easy() {
        if let Ok((graph, keys)) = parse::<U6>("resources/test/easy_emoji.txt") {
            println!("{:?}", keys);
            println!("{:?}", graph);
            assert_eq!(keys.len(), 3);
            assert_eq!(graph.vertices.len(), 4);
        }
    }

    #[test]
    fn test_parse_medium() {
        if let Ok((graph, keys)) = parse::<U6>("resources/test/medium_emoji.txt") {
            println!("{:?}", keys);
            println!("{:?}", graph);
            assert_eq!(keys.len(), 7);
            assert_eq!(graph.vertices.len(), 108);
        }
    }
}
