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
    fn rotate_90(&self) -> Self {
        let iter = (0..self.nrows()).rev().flat_map(|i| {
            (0..self.ncols()).map(move |j| self[(i, j)])
        });
        DMatrix::<usize>::from_iterator(self.ncols(), self.nrows(), iter)
    }
}

pub trait Reflection {
    fn reflect_vertical(&self) -> Self;

    fn reflect_horizontal(&self) -> Self;
}

impl Reflection for DMatrix<usize> {
    fn reflect_vertical(&self) -> Self {
        let iter = (0..self.ncols()).flat_map(|j| {
            (0..self.nrows()).rev().map(move |i| self[(i, j)])
        });
        DMatrix::<usize>::from_iterator(self.nrows(), self.ncols(), iter)
    }

    fn reflect_horizontal(&self) -> Self {
        let iter = (0..self.ncols()).rev().flat_map(|j| {
            (0..self.nrows()).map(move |i| self[(i, j)])
        });
        DMatrix::<usize>::from_iterator(self.nrows(), self.ncols(), iter)
    }
}

pub trait DiagonalReflection {
    fn reflect_top_left(&self) -> Self;

    fn reflect_bottom_left(&self) -> Self;
}

impl DiagonalReflection for DMatrix<usize> {
    fn reflect_top_left(&self) -> Self {
        self.rotate_90().reflect_vertical()
    }

    fn reflect_bottom_left(&self) -> Self {
        self.rotate_90().reflect_horizontal()
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

    #[test]
    fn test_rotation_2x2() {
        /*
        0 1  -->  2 0
        2 3       3 1
         */

        let matrix = DMatrix::from_row_slice(2, 2, &vec![0, 1, 2, 3]);
        let result = DMatrix::from_row_slice(2, 2, &vec![2, 0, 3, 1]);
        assert_eq!(matrix.rotate_90(), result)
    }

    #[test]
    fn test_rotation_3x3() {
        /*
        0  1  2  -->  9  4  0
        4  5  6       10 5  1
        9 10 11       11 6  2
         */

        let matrix = DMatrix::from_row_slice(3, 3, &vec![0, 1, 2, 4, 5, 6, 9, 10, 11]);
        let result = DMatrix::from_row_slice(3, 3, &vec![9, 4, 0, 10, 5, 1, 11, 6, 2]);
        assert_eq!(matrix.rotate_90(), result)
    }

    #[test]
    fn test_rotation_2x3() {
        /*
        0  1  2  -->  4  0
        4  5  6       5  1
                      6  2
         */

        let matrix = DMatrix::from_row_slice(2, 3, &vec![0, 1, 2, 4, 5, 6]);
        let result = DMatrix::from_row_slice(3, 2, &vec![4, 0, 5, 1, 6, 2]);
        assert_eq!(matrix.rotate_90(), result)
    }

    #[test]
    fn test_reflect_vertical_2x2() {
        /*
        0 1  -->  2 3
        2 3       0 1
         */

        let matrix = DMatrix::from_row_slice(2, 2, &vec![0, 1, 2, 3]);
        let result = DMatrix::from_row_slice(2, 2, &vec![2, 3, 0, 1]);
        assert_eq!(matrix.reflect_vertical(), result)
    }

    #[test]
    fn test_reflect_vertical_3x3() {
        /*
        0  1  2  -->  9 10 11
        4  5  6       4  5  6
        9 10 11       0  1  2
         */

        let matrix = DMatrix::from_row_slice(3, 3, &vec![0, 1, 2, 4, 5, 6, 9, 10, 11]);
        let result = DMatrix::from_row_slice(3, 3, &vec![9, 10, 11, 4, 5, 6, 0, 1, 2]);
        assert_eq!(matrix.reflect_vertical(), result)
    }

    #[test]
    fn test_reflect_vertical_2x3() {
        /*
        0  1  2  -->  4  5  6
        4  5  6       0  1  2
         */

        let matrix = DMatrix::from_row_slice(2, 3, &vec![0, 1, 2, 4, 5, 6]);
        let result = DMatrix::from_row_slice(2, 3, &vec![4, 5, 6, 0, 1, 2]);
        assert_eq!(matrix.reflect_vertical(), result)
    }

    #[test]
    fn test_reflect_vertical_3x2() {
        /*
        0  1  -->  5  6
        2  4       2  4
        5  6       0  1
         */

        let matrix = DMatrix::from_row_slice(3, 2, &vec![0, 1, 2, 4, 5, 6]);
        let result = DMatrix::from_row_slice(3, 2, &vec![5, 6, 2, 4, 0, 1]);
        assert_eq!(matrix.reflect_vertical(), result)
    }

    #[test]
    fn test_reflect_horizontal_2x2() {
        /*
        0 1 --> 1 0
        2 3     3 2
         */

        let matrix = DMatrix::from_row_slice(2, 2, &vec![0, 1, 2, 3]);
        let result = DMatrix::from_row_slice(2, 2, &vec![1, 0, 3, 2]);
        assert_eq!(matrix.reflect_horizontal(), result)
    }

    #[test]
    fn test_reflect_horizontal_3x3() {
        /*
        0  1  2  -->  2  1  0
        4  5  6       6  5  4
        9 10 11       11 10 9
         */

        let matrix = DMatrix::from_row_slice(3, 3, &vec![0, 1, 2, 4, 5, 6, 9, 10, 11]);
        let result = DMatrix::from_row_slice(3, 3, &vec![2, 1, 0, 6, 5, 4, 11, 10, 9]);
        assert_eq!(matrix.reflect_horizontal(), result)
    }

    #[test]
    fn test_reflect_horizontal_2x3() {
        /*
        0  1  2  -->  2  1  0
        4  5  6       6  5  4
         */

        let matrix = DMatrix::from_row_slice(2, 3, &vec![0, 1, 2, 4, 5, 6]);
        let result = DMatrix::from_row_slice(2, 3, &vec![2, 1, 0, 6, 5, 4]);
        assert_eq!(matrix.reflect_horizontal(), result)
    }

    #[test]
    fn test_reflect_top_left_2x2() {
        /*
        0 1 --> 3 1
        2 3     2 0
         */

        let matrix = DMatrix::from_row_slice(2, 2, &vec![0, 1, 2, 3]);
        let result = DMatrix::from_row_slice(2, 2, &vec![3, 1, 2, 0]);
        assert_eq!(matrix.reflect_top_left(), result)
    }

    #[test]
    fn test_reflect_top_left_3x3() {
        /*
        0  1  2  -->  11 6  2
        4  5  6       10 5  1
        9 10 11       9  4  0
         */

        let matrix = DMatrix::from_row_slice(3, 3, &vec![0, 1, 2, 4, 5, 6, 9, 10, 11]);
        let result = DMatrix::from_row_slice(3, 3, &vec![11, 6, 2, 10, 5, 1, 9, 4, 0]);
        assert_eq!(matrix.reflect_top_left(), result)
    }

    #[test]
    fn test_reflect_top_left_2x3() {
        /*
        0  1  2  -->  6  2
        4  5  6       5  1
                      4  0
         */

        let matrix = DMatrix::from_row_slice(2, 3, &vec![0, 1, 2, 4, 5, 6]);
        let result = DMatrix::from_row_slice(3, 2, &vec![6, 2, 5, 1, 4, 0]);
        assert_eq!(matrix.reflect_top_left(), result)
    }

    #[test]
    fn test_reflect_bottom_left_2x2() {
        /*
        0 1 --> 0 2
        2 3     1 3
         */

        let matrix = DMatrix::from_row_slice(2, 2, &vec![0, 1, 2, 3]);
        let result = DMatrix::from_row_slice(2, 2, &vec![0, 2, 1, 3]);
        assert_eq!(matrix.reflect_bottom_left(), result)
    }

    #[test]
    fn test_reflect_bottom_left_3x3() {
        /*
        0  1  2  -->  0  4  9
        4  5  6       1  5 10
        9 10 11       2  6 11
         */

        let matrix = DMatrix::from_row_slice(3, 3, &vec![0, 1, 2, 4, 5, 6, 9, 10, 11]);
        let result = DMatrix::from_row_slice(3, 3, &vec![0, 4, 9, 1, 5, 10, 2, 6, 11]);
        assert_eq!(matrix.reflect_bottom_left(), result)
    }

    #[test]
    fn test_reflect_bottom_left_2x3() {
        /*
        0  1  2  -->  0  4
        4  5  6       1  5
                      2  6
         */

        let matrix = DMatrix::from_row_slice(2, 3, &vec![0, 1, 2, 4, 5, 6]);
        let result = DMatrix::from_row_slice(3, 2, &vec![0, 4, 1, 5, 2, 6]);
        assert_eq!(matrix.reflect_bottom_left(), result)
    }
}