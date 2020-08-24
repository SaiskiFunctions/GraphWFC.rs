use hashbrown::HashMap;
use std::fs::{read_to_string, write};
use std::io::Error;
use std::ops::Index;
use crate::graph::graph::{Edges, EdgeDirection, VertexIndex, Graph};
use crate::multiset::{Multiset, MultisetTrait, MultisetScalar};
use nalgebra::{Dim, DimName, DefaultAllocator};
use nalgebra::allocator::Allocator;


pub type CharKeyMap = HashMap<usize, char>;

pub fn parse<D>(filename: &str) -> Result<(Graph<D>, CharKeyMap), Error>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    read_to_string(filename).map(|string| {
        let edges = make_edges(&string);
        let char_frequency: HashMap<char, u32> = char_frequency(&string);
        let char_keys: CharKeyMap = char_frequency.keys().copied().enumerate().collect();
        let all_labels: Multiset<D> = make_all_labels(&char_frequency, &char_keys);
        let vertices: Vec<Multiset<D>> = make_vertices(&string, &char_frequency, &char_keys);

        (Graph::new(vertices, edges, all_labels), char_keys)
    })
}

fn char_frequency(string: &str) -> HashMap<char, u32> {
    string.chars().fold(HashMap::new(), |mut map, char| {
        if char != '\n' {
            map.entry(char).and_modify(|freq| *freq += 1).or_insert(1);
        }
        map
    })
}

fn make_all_labels<D>(char_frequency: &HashMap<char, u32>, char_keys: &CharKeyMap) -> Multiset<D>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    Multiset::from_iter_u((0..char_keys.len()).map(|index| {
        *char_frequency.index(char_keys.index(&index))
    }))
}

fn make_vertices<D>(string: &str, char_frequency: &HashMap<char, u32>, char_keys: &CharKeyMap) -> Vec<Multiset<D>>
    where D: Dim + DimName,
          DefaultAllocator: Allocator<MultisetScalar, D>
{
    string.chars().filter(|c| c != &'\n').map(|c| {
        Multiset::from_iter_u((0..char_keys.len()).map(|index| {
            let char = char_keys.index(&index);
            if char == &c { *char_frequency.index(&c) } else { 0 }
        }))
    }).collect()
}

fn make_edges(string: &str) -> Edges {
    let lines: Vec<&str> = string.split('\n').filter(|l| !l.is_empty()).collect();
    let line_len = lines.index(0).chars().count();
    make_edges_cardinal_grid(line_len, lines.len())
}

pub fn make_edges_cardinal_grid(width: usize, depth: usize) -> Edges {
    let mut edges: Edges = HashMap::new();
    (0..depth).for_each(|depth_index| {
        (0..width).for_each(|width_index| {
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
        });
    });
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
    if write(filename, lines).is_ok() {}
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
