use crate::graph::graph::{Edges, VertexIndex};
use hashbrown::HashMap;

pub struct Directions {
    n: bool, ne: bool, e: bool, se: bool,
    s: bool, sw: bool, w: bool, nw: bool,
}

impl Directions {
    fn new(n: bool, ne: bool, e: bool, se: bool,
           s: bool, sw: bool, w: bool, nw: bool) -> Directions {
        Directions { n, ne, e, se, s, sw, w, nw, }
    }

    fn fns(&self) -> Vec<impl Fn(u32, u32, u32, u32) -> Option<(u32, u16)>> {
        let mask = &[
            self.n, self.ne, self.e, self.se,
            self.s, self.sw, self.w, self.nw
        ];
        let fns = &[
            north, north_east, east, south_east,
            south, south_west, west, north_west
        ];
        fns.into_iter().zip(mask).filter(|(_, &m)| m).map(|(f, _)| f).collect()
    }

    pub fn make_edges(&self, width: usize, depth: usize) -> Edges {
        let fns = self.fns();
        let (depth, width) = (depth as VertexIndex, width as VertexIndex);
        let mut edges: Edges = HashMap::new();
        (0..depth).for_each(|depth_index| {
            (0..width).for_each(|width_index| {
                let direction_pairs = fns.iter().map(|f| {
                    f(depth, depth_index, width, width_index)
                }).flatten().collect();
                let this_vertex_index = depth_index * width + width_index;
                edges.insert(this_vertex_index, direction_pairs);
            });
        });
        edges
    }
}

#[allow(unused_variables)]
//NORTH = 0
fn north(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di > 0 { Some(((di - 1) * w + wi, 0)) } else { None }
}

#[allow(unused_variables)]
//NORTH EAST = 1
fn north_east(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di > 0 && wi < w - 1 { Some(((di - 1) * w + wi + 1, 1)) } else { None }
}

#[allow(unused_variables)]
//EAST = 2
fn east(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if wi < w - 1 { Some(((wi + 1) + di * w, 2)) } else { None }
}

//SOUTH EAST = 3
fn south_east(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di < d - 1 && wi < w - 1 { Some(((di + 1) * w + wi + 1, 3)) } else { None }
}

//SOUTH = 4
fn south(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di < d - 1 { Some(((di + 1) * w + wi, 4)) } else { None }
}

//SOUTH WEST = 5
fn south_west(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di < d - 1 && wi > 0 { Some(((di + 1) * w + wi - 1, 5)) } else { None }
}

#[allow(unused_variables)]
//WEST = 6
fn west(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if wi > 0 { Some(((wi - 1) + di * w, 6)) } else { None }
}

#[allow(unused_variables)]
//NORTH WEST = 7
fn north_west(d: u32, di: u32, w: u32, wi: u32) -> Option<(u32, u16)> {
    if di > 0 && wi > 0 { Some(((di - 1) * w + wi - 1, 7)) } else { None }
}

pub fn make_edges_cardinal_grid(width: usize, depth: usize) -> Edges {
    let d = Directions::new(true, false, true, false, true, false, true, false);
    d.make_edges(width, depth)
}

pub fn make_edges_8_way_grid(width: usize, depth: usize) -> Edges {
    let d = Directions::new(true, true, true, true, true, true, true, true);
    d.make_edges(width, depth)
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
            (2, vec![(0, 0), (3, 2)])
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
            (2, vec![(0, 0), (1, 1), (3, 2)])
        ]);
        assert_eq!(result, expected);
    }
}
