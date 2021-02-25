use crate::graph::graph::{Edges, VertexIndex};
use hashbrown::HashMap;
use nalgebra::DMatrix;

//              +--- index of neighbor
//              |    +-- direction of neighbor
//              |    |
//              v    v
type IdxDir = (u32, u16);

pub struct Directions {
    code: u8,
}

impl Directions {
    pub fn new(code: u8) -> Directions {
        Directions { code }
    }

    fn fns(&self) -> Vec<impl Fn(u32, u32, u32, u32) -> Option<IdxDir>> {
        let mask = format!("{:08b}", self.code);
        let fns = &[
            // north, north_east, east, south_east, south, south_west, west, north_west,
            north_west, north, north_east, west, east, south_west, south, south_east,
        ];
        fns.iter()
            .zip(mask.chars())
            .filter_map(|(f, m)| (m == '1').then(|| f))
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
fn north(d: u32, di: u32, w: u32, wi: u32) -> Option<IdxDir> {
    (di > 0).then(|| ((di - 1) * w + wi, 1))
}

#[allow(unused_variables)]
//NORTH EAST = 2
fn north_east(d: u32, di: u32, w: u32, wi: u32) -> Option<IdxDir> {
    (di > 0 && wi < w - 1).then(|| ((di - 1) * w + wi + 1, 2))
}

#[allow(unused_variables)]
//EAST = 4
fn east(d: u32, di: u32, w: u32, wi: u32) -> Option<IdxDir> {
    (wi < w - 1).then(|| ((wi + 1) + di * w, 4))
}

//SOUTH EAST = 7
fn south_east(d: u32, di: u32, w: u32, wi: u32) -> Option<IdxDir> {
    (di < d - 1 && wi < w - 1).then(|| ((di + 1) * w + wi + 1, 7))
}

//SOUTH = 6
fn south(d: u32, di: u32, w: u32, wi: u32) -> Option<IdxDir> {
    (di < d - 1).then(|| ((di + 1) * w + wi, 6))
}

//SOUTH WEST = 5
fn south_west(d: u32, di: u32, w: u32, wi: u32) -> Option<IdxDir> {
    (di < d - 1 && wi > 0).then(|| ((di + 1) * w + wi - 1, 5))
}

#[allow(unused_variables)]
//WEST = 3
fn west(d: u32, di: u32, w: u32, wi: u32) -> Option<IdxDir> {
    (wi > 0).then(|| ((wi - 1) + di * w, 3))
}

#[allow(unused_variables)]
//NORTH WEST = 0
fn north_west(d: u32, di: u32, w: u32, wi: u32) -> Option<IdxDir> {
    (di > 0 && wi > 0).then(|| ((di - 1) * w + wi - 1, 0))
}

pub fn make_edges_cardinal_grid(width: usize, depth: usize) -> Edges {
    Directions::new(90).make_edges(width, depth)
}

pub fn make_edges_8_way_grid(width: usize, depth: usize) -> Edges {
    Directions::new(255).make_edges(width, depth)
}

pub trait Rotation {
    fn rotate_90(&self) -> Self;
}

impl Rotation for DMatrix<usize> {
    fn rotate_90(&self) -> DMatrix<usize> {
        assert_eq!(self.nrows(), self.ncols());
        let side = self.nrows();
        let mut target_matrix = DMatrix::<usize>::zeros(side, side);

        (0..side).for_each(|i| {
            (0..side).for_each(|j| target_matrix[(j, (side - 1) - i)] = self[(i, j)]);
        });
        target_matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::hash_map;

    #[test]
    fn test_make_edges_cardinal_grid() {
        let result = make_edges_cardinal_grid(2, 2);
        let expected = hash_map(&[
            (1, vec![(0, 3), (3, 6)]),
            (0, vec![(1, 4), (2, 6)]),
            (3, vec![(1, 1), (2, 3)]),
            (2, vec![(0, 1), (3, 4)]),
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_make_edges_8_way_grid() {
        let result = make_edges_8_way_grid(2, 2);
        let expected = hash_map(&[
            (1, vec![(0, 3), (2, 5), (3, 6)]),
            (0, vec![(1, 4), (2, 6), (3, 7)]),
            (3, vec![(0, 0), (1, 1), (2, 3)]),
            (2, vec![(0, 1), (1, 2), (3, 4)]),
        ]);
        assert_eq!(result, expected);
    }
}
