use image::{GenericImageView, DynamicImage, RgbImage, Rgb, imageops};
use nalgebra::{DMatrix, Matrix2};
use nalgebra::geometry::Rotation2;
use std::collections::HashSet;
use hashbrown::HashMap;
use bimap::BiMap;

// Matrix and image data is in COLUMN MAJOR so:
// [1, 2, 3, 4] is equivalent to:
/*
 1 3
 2 4
 */

static RGB_CHANNELS: u8 = 3;

pub fn parse() {
    let img = image::open("resources/test/City.png").unwrap();
}

pub trait Rotation {
    fn rotate_90(&self) -> DMatrix<u32>;
}

impl Rotation for DMatrix<u32> {
    fn rotate_90(&self) -> DMatrix<u32> {
        assert_eq!(self.nrows(), self.ncols());
        let side = self.nrows();
        let mut target_matrix = DMatrix::<u32>::zeros(side, side);

        (0..side).for_each(|i| {
            (0..side).for_each(|j| {
                target_matrix[(j, (side - 1) - i)] = self[(i, j)]
            });
        });

        target_matrix
    }
}

// A function that returns an iterator of aliased image chunks (unrotated) ->

// alias_image
fn alias_pixels(image: &RgbImage) -> BiMap<u32, Rgb<u8>> {
    image
        .pixels()
        .fold(HashSet::<Rgb<u8>>::new(), |mut acc, pixel| {
            acc.insert(*pixel);
            acc
        })
        .into_iter()
        .enumerate()
        .map(|(i, p)| (i as u32, p))
        .collect()
}

// TODO: variable sub_chunk_size
fn chunk_image(image: &RgbImage, chunk_size: u32, pixel_aliases: &BiMap<u32, Rgb<u8>>) -> HashSet<DMatrix<u32>> {
    let chunk_size_u = chunk_size as usize;
    let (height, width) = image.dimensions();
    let mut chunk_set: HashSet<DMatrix<u32>> = HashSet::new();

    for y in 0..height - (chunk_size - 1) {
        for x in 0..width - (chunk_size - 1) {
            // get vec of pixel aliases
            let pixels: Vec<u32> = imageops::crop_imm(image, x, y, chunk_size, chunk_size).to_image()
                .pixels()
                .map(|p| *pixel_aliases.get_by_right(&p).unwrap())
                .collect();

            let chunk = DMatrix::from_column_slice(chunk_size_u, chunk_size_u, &pixels);

            if chunk_set.contains(&chunk) { continue }
            chunk_set.insert(chunk.clone());
            
            let chunk_r90 = chunk.rotate_90();
            if chunk_set.contains(&chunk_r90) { continue }
            chunk_set.insert(chunk_r90.clone());

            let chunk_r180 = chunk_r90.rotate_90();
            if chunk_set.contains(&chunk_r180) { continue }
            chunk_set.insert(chunk_r180.clone());

            let chunk_r270 = chunk_r180.rotate_90();
            chunk_set.insert(chunk_r270);
        }
    }
    chunk_set
}

pub fn f() {
    /*
        1. For each a chunk, generate all possible chunks that can be adjacent to it
        2. For each adjacent chunk do the intersection of the shared adjacencies
        3. If both shared adjacencies are non empty then add a graph edge
     */
}

pub fn is_sym(matrix: &DMatrix<i32>) -> bool {
    let transposed_matrix = matrix.transpose();
    if matrix.row(0).iter().count() != transposed_matrix.row(0).iter().count() { return false } 
    true
}

#[cfg(test)]
mod tests {

    use super::*;
    use image::ImageBuffer;

    // we need to be more aware of when trait implementations imply things about syntactical constructs

    #[test]
    fn test_alias_pixels() {
        let pixels = vec![255, 255, 255, 0, 0, 0, 122, 122, 122, 96, 96, 96];
        let img = ImageBuffer::from_vec(2, 2, pixels).unwrap();
        let pixel_aliases = alias_pixels(&img);
        assert_eq!(pixel_aliases.len(), 4);
    }

    #[test]
    fn test_move_data() {
        let side = 4;
        let basic_matrix = DMatrix::from_column_slice(side, side, &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        let mut target_matrix = DMatrix::<u32>::zeros(side, side);

        (0..side).for_each(|i| {
            (0..side).for_each(|j| {
                target_matrix[(j, (side - 1) - i)] = basic_matrix[(i, j)]
            });
        });
    }



    #[test]
    fn test_image_loading() {
        let img = image::open("resources/test/City.png");
        assert!(img.is_ok())
    }

    #[test]
    fn test_chunk_image() {
        let pixels = vec![255, 255, 255, 0, 0, 0, 122, 122, 122, 96, 96, 96];
        let img = ImageBuffer::from_vec(2, 2, pixels).unwrap();
        let pixel_aliases = alias_pixels(&img);

        let chunk_set = chunk_image(&img, 2, &pixel_aliases);

        assert_eq!(chunk_set.len(), 4);
        assert!(chunk_set.contains(&DMatrix::from_column_slice(2, 2, &vec![2, 0, 3, 1])));
        assert!(chunk_set.contains(&DMatrix::from_column_slice(2, 2, &vec![3, 2, 1, 0])));
        assert!(chunk_set.contains(&DMatrix::from_column_slice(2, 2, &vec![1, 3, 0, 2])));
        assert!(chunk_set.contains(&DMatrix::from_column_slice(2, 2, &vec![0, 1, 2, 3])));
    }

    #[test]
    fn test_image_crop() {
        let img = image::open("resources/test/City.png").unwrap();
        //println!("{:?}", img.crop_imm(100, 0, 2, 2).to_rgb());
    }

    // #[test]
    // fn test_chunk_image() {
    //     let img = image::open("resources/test/4pix.png").unwrap();
    //     let chunk_set = chunk_image(&img);
    //     println!("{:?}", chunk_set);
        
    // }

    #[test]
    fn test_rotation_matrix() {
        // let basic_matrix = Matrix2::from_row_slice(&vec![2, 3, 4, 5]);
        // let rotation_matrix = Matrix2::from_row_slice(&vec![0, -1, 1, 0]);
        // let rotated_matrix = basic_matrix * rotation_matrix;


        // just alias your RGB channels
        let basic_matrix = DMatrix::from_row_slice(2, 2, &vec![2, 3, 4, 5]);
        let rotation_matrix = DMatrix::from_row_slice(2, 2, &vec![0, -1, 1, 0]);
        let rotated_matrix = basic_matrix * rotation_matrix;
        //println!("{:?}", rotated_matrix);
    }

    #[test]
    fn test_symmetry() {
        let asym_matrix = DMatrix::from_row_slice(2, 3, &vec![2, 1, 2, 0, 2, 1]);
        let trans_asym_matrix = asym_matrix.transpose();
        // asym_matrix.row(0).into_iter().for_each(|r| { println!("{:?}", r)});
        // trans_asym_matrix.row(0).into_iter().for_each(|r| { println!("{:?}", r)});
        //println!("{}", asym_matrix.row(0).iter());

        let sym_matrix = DMatrix::from_row_slice(2, 2, &vec![2, 1, 2, 0]);
        // let trans_sym_matrix = sym_matrix.transpose();
        // println!("{:?}", sym_matrix == trans_sym_matrix);
        // assert!(!is_sym(&asym_matrix));
        // assert!(is_sym(&sym_matrix));
    }
    // 2 1
    // 2 0   2 2 2
    // 2 1   1 0 1

    //cargo test --package wfc-rust --lib io::image_olm_parser::tests -- --nocapture 
    // [(255, 14, 0), (255, 255, 255), (52, 0, 254), (0, 255, 11)] * [0, 1, -1, 0] / [0,-1, 1, 0]
    // 0 1
    // 2 3


    // 2 3
    // 4 5

    // 

    // 3 5
    // 2 4

    // 1 1 1
    // 0 0 0
    // 1 1 1

    // 0 1 0
    // 1 1 1
    // 0 1 0
}
