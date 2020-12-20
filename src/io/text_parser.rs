use crate::graph::graph::{Edges, Graph};
use crate::io::utils::{make_edges_cardinal_grid, make_edges_8_way_grid};
use crate::multiset::Multiset;
use hashbrown::HashMap;
use num_traits::One;
use std::fs::{read_to_string, write};
use std::io::Error;
use std::ops::Index;
use bimap::BiHashMap;

pub type CharKeyBimap = BiHashMap<usize, char>;

pub fn parse<S: Multiset>(filename: &str) -> Result<(Graph<S>, CharKeyBimap), Error> {
    read_to_string(filename).map(|string| {
        let edges = make_edges(&string);
        let char_frequency = char_frequency::<S>(&string);
        let char_keys: CharKeyBimap = char_keys::<S>(&char_frequency);
        let all_labels = make_all_labels(&char_frequency, &char_keys);
        let vertices = make_vertices(&string, &char_frequency, &char_keys, true);

        (Graph::new(vertices, edges, all_labels), char_keys)
    })
}

fn char_keys<S: Multiset>(char_frequency: &HashMap<char, S::Item>) -> CharKeyBimap {
    char_frequency.keys().copied().enumerate().collect()
}

fn char_frequency<S: Multiset>(string: &str) -> HashMap<char, S::Item> {
    string.chars().filter(|c| c != &'\n').fold(HashMap::new(), |mut map, char| {
        map.entry(char)
            .and_modify(|freq| *freq += One::one())
            .or_insert(One::one());
        map
    })
}

fn make_all_labels<S: Multiset>(
    char_frequency: &HashMap<char, S::Item>,
    char_keys: &CharKeyBimap,
) -> S {
    S::from_iter_u((0..char_keys.len()).map(|index| {
        let character = char_keys.get_by_left(&index).unwrap();
        *char_frequency.index(character)
    }))
}

fn make_vertices<S: Multiset>(
    string: &str,
    char_frequency: &HashMap<char, S::Item>,
    char_keys: &CharKeyBimap,
    directional: bool
) -> Vec<S> {
    string
        .chars()
        .filter(|c| c != &'\n')
        .map(|c| {
            let index = char_keys.get_by_right(&c).unwrap();
            let mut set = S::empty(char_keys.len());
            if directional {
                *set.index_mut(*index) = One::one()
            } else {
                *set.index_mut(*index) = *char_frequency.index(&c)
            }
            set
        })
        .collect()
}

fn make_edges(string: &str) -> Edges {
    let width = string.chars().enumerate()
        .find_map(|t| if t.1 == '\n' { Some(t.0) } else { None })
        .unwrap_or_else(|| string.chars().count());
    let height = string.split('\n').filter(|l| !l.is_empty()).count();
    // make_edges_cardinal_grid(width, height)
    make_edges_8_way_grid(width, height)
}

pub fn render<S: Multiset>(
    filename: &str,
    graph: &Graph<S>,
    key: &CharKeyBimap,
    width: usize,
) {
    let rendered_vertices: Vec<char> = graph
        .vertices
        .iter()
        .map(|labels| {
            if let Some(k) = labels.get_non_zero() {
                *key.get_by_left(&k).unwrap()
            } else {
                '❌'
            }
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
    use nalgebra::{VectorN, U6, U7};

    #[test]
    fn test_parse_easy() {
        if let Ok((graph, keys)) = parse::<VectorN<u16, U6>>("resources/test/easy_emoji.txt") {
            println!("{:?}", keys);
            println!("{:?}", graph);
            assert_eq!(keys.len(), 3);
            assert_eq!(graph.vertices.len(), 4);
        }
    }

    #[test]
    fn test_parse_medium() {
        if let Ok((graph, keys)) = parse::<VectorN<u16, U7>>("resources/test/medium_emoji.txt") {
            println!("{:?}", keys);
            println!("{:?}", graph);
            assert_eq!(keys.len(), 7);
            assert_eq!(graph.vertices.len(), 108);
        }
    }
}
