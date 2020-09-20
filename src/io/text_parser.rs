use crate::graph::graph::{Edges, Graph};
use crate::io::utils::{make_edges_cardinal_grid, make_edges_8_way_grid};
use crate::multiset::Multiset;
use hashbrown::HashMap;
use num_traits::{One, Zero};
use std::fs::{read_to_string, write};
use std::io::Error;
use std::ops::Index;

pub type CharKeyMap = HashMap<usize, char>;

pub fn parse<S: Multiset>(filename: &str) -> Result<(Graph<S>, CharKeyMap), Error> {
    read_to_string(filename).map(|string| {
        let edges = make_edges(&string);
        let char_frequency = char_frequency::<S>(&string);
        let char_keys: CharKeyMap = char_frequency.keys().copied().enumerate().collect();
        let all_labels = make_all_labels(&char_frequency, &char_keys);
        let vertices = make_vertices(&string, &char_frequency, &char_keys);

        (Graph::new(vertices, edges, all_labels), char_keys)
    })
}

fn char_frequency<S: Multiset>(string: &str) -> HashMap<char, S::Item> {
    string.chars().fold(HashMap::new(), |mut map, char| {
        if char != '\n' {
            map.entry(char)
                .and_modify(|freq| *freq += One::one())
                .or_insert(One::one());
        }
        map
    })
}

fn make_all_labels<S: Multiset>(
    char_frequency: &HashMap<char, S::Item>,
    char_keys: &CharKeyMap,
) -> S {
    S::from_iter_u((0..char_keys.len()).map(|index| {
        let character = char_keys.index(&index);
        *char_frequency.index(character)
    }))
}

fn make_vertices<S: Multiset>(
    string: &str,
    char_frequency: &HashMap<char, S::Item>,
    char_keys: &CharKeyMap,
) -> Vec<S> {
    string
        .chars()
        .filter(|c| c != &'\n')
        .map(|c| {
            S::from_iter_u((0..char_keys.len()).map(|index| {
                let char = char_keys.index(&index);
                if char == &c {
                    *char_frequency.index(&c)
                    // One::one()
                } else {
                    Zero::zero()
                }
            }))
        })
        .collect()
}

fn make_edges(string: &str) -> Edges {
    let lines: Vec<&str> = string.split('\n').filter(|l| !l.is_empty()).collect();
    let line_len = lines.index(0).chars().count();
    make_edges_cardinal_grid(line_len, lines.len())
    // make_edges_8_way_grid(line_len, lines.len())
}

pub fn render<S: Multiset>(
    filename: &str,
    graph: &Graph<S>,
    key: &HashMap<usize, char>,
    width: usize,
) {
    let rendered_vertices: Vec<char> = graph
        .vertices
        .iter()
        .map(|labels| {
            let k = labels.get_non_zero().unwrap();
            *key.index(&k)
        })
        .collect();
    let lines: String = rendered_vertices
        .chunks_exact(width)
        .map(|chunk| chunk.iter().collect::<String>() + "\n")
        .collect::<String>();
    if write(filename, lines).is_ok() {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{VectorN, U6};

    type MS6 = VectorN<u16, U6>;

    #[test]
    fn test_parse_easy() {
        if let Ok((graph, keys)) = parse::<MS6>("resources/test/easy_emoji.txt") {
            println!("{:?}", keys);
            println!("{:?}", graph);
            assert_eq!(keys.len(), 3);
            assert_eq!(graph.vertices.len(), 4);
        }
    }

    #[test]
    fn test_parse_medium() {
        if let Ok((graph, keys)) = parse::<MS6>("resources/test/medium_emoji.txt") {
            println!("{:?}", keys);
            println!("{:?}", graph);
            assert_eq!(keys.len(), 7);
            assert_eq!(graph.vertices.len(), 108);
        }
    }
}
