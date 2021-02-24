use crate::graph::graph::{Edges, VertexIndex};
use hashbrown::HashMap;

pub struct Directions {
    code: u8,
}

impl Directions {
    pub fn new(code: u8) -> Directions {
        Directions { code }
    }

    fn fns(&self) -> Vec<impl Fn(u32, u32, u32, u32) -> Option<(u32, u16)>> {
        let mask = format!("{:b}", self.code);
        let fns = &[
            // north, north_east, east, south_east, south, south_west, west, north_west,
            north_west, north, north_east, west, east, south_west, south, south_east,
        ];
        fns.iter()
            .zip(mask.chars())
            .filter_map(|(f, m)| if m == '1' { Some(f) } else { None })
            .collect()
    }

    pub fn make_edges(&self, width: usize, depth: usize) -> Edges {
        let fns = self.fns();
        let (depth, width) = (depth as VertexIndex, width as VertexIndex);
        let mut edges = HashMap::new();
        (0..depth).for_each(|depth_index| {
            (0..width).for_each(|width_index| {
                let direction_pairs = fns
                    .iter()
                    .filter_map(|f| f(depth, depth_index, width, width_index))
                    .collect();
                let this_vertex_index = depth_index * width + width_index;
                edges.insert(this_vertex_index, direction_pairs);
            });
        });
        edges
    }
}

#[allow(unused_variables)]
//NORTH = 1
fn north(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di > 0 {
        Some(((di - 1) * w + wi, 1))
    } else {
        None
    }
}

#[allow(unused_variables)]
//NORTH EAST = 2
fn north_east(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di > 0 && wi < w - 1 {
        Some(((di - 1) * w + wi + 1, 2))
    } else {
        None
    }
}

#[allow(unused_variables)]
//EAST = 4
fn east(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if wi < w - 1 {
        Some(((wi + 1) + di * w, 4))
    } else {
        None
    }
}

//SOUTH EAST = 7
fn south_east(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di < d - 1 && wi < w - 1 {
        Some(((di + 1) * w + wi + 1, 7))
    } else {
        None
    }
}

//SOUTH = 6
fn south(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di < d - 1 {
        Some(((di + 1) * w + wi, 6))
    } else {
        None
    }
}

//SOUTH WEST = 5
fn south_west(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di < d - 1 && wi > 0 {
        Some(((di + 1) * w + wi - 1, 5))
    } else {
        None
    }
}

#[allow(unused_variables)]
//WEST = 3
fn west(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if wi > 0 {
        Some(((wi - 1) + di * w, 3))
    } else {
        None
    }
}

#[allow(unused_variables)]
//NORTH WEST = 0
fn north_west(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di > 0 && wi > 0 {
        Some(((di - 1) * w + wi - 1, 0))
    } else {
        None
    }
}

pub fn make_edges_cardinal_grid(width: usize, depth: usize) -> Edges {
    Directions::new(170).make_edges(width, depth)
}

pub fn make_edges_8_way_grid(width: usize, depth: usize) -> Edges {
    Directions::new(255).make_edges(width, depth)
}

pub fn index_to_coords(index: u32, width: u32) -> (u32, u32) { (index % width, index / width) }

pub fn coords_to_index((x, y): (u32, u32), width: u32) -> u32 { x + y * width }

pub fn is_inside((x, y): (i32, i32), (w, h): (u32, u32)) -> bool {
    if x < 0 || y < 0 || x > (w as i32 - 1) || y > (h as i32 - 1) { false } else { true }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::hash_map;

    #[test]
    fn test_make_edges_cardinal_grid() {
        let result = make_edges_cardinal_grid(2, 2);
        let expected = hash_map(&[
            (1, vec![(3, 4), (0, 6)]),
            (0, vec![(1, 2), (2, 4)]),
            (3, vec![(1, 0), (2, 6)]),
            (2, vec![(0, 0), (3, 2)]),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_make_edges_8_way_grid() {
        let result = make_edges_8_way_grid(2, 2);
        let expected = hash_map(&[
            (1, vec![(3, 4), (2, 5), (0, 6)]),
            (0, vec![(1, 2), (3, 3), (2, 4)]),
            (3, vec![(1, 0), (2, 6), (0, 7)]),
            (2, vec![(0, 0), (1, 1), (3, 2)]),
        ]);
        assert_eq!(result, expected);
    }
}
