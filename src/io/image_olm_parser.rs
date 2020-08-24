use image::{GenericImageView, DynamicImage, RgbImage};
use nalgebra::{DMatrix, Matrix2};
use nalgebra::geometry::Rotation2;
use std::collections::HashSet;

static RGB_CHANNELS: u8 = 3;

pub fn parse() {
    let img = image::open("resources/test/City.png").unwrap();
}



fn chunk_image(image: &DynamicImage) -> HashSet<Vec<u8>> {
    let chunk_size = 2;
    let sub_chunk_size = 1; // pixels per sub chunk

    let (height, width) = image.dimensions();

    let mut chunk_set: HashSet<Vec<u8>> = HashSet::new();

    (0..height - (chunk_size - 1)).for_each(|y| {
        (0..width - (chunk_size - 1)).for_each(|x| {
            // get vec of pixel channels
            chunk = DMatrix.for_row_vector(chunk_size, chunk_size, image.crop_imm(x, y, chunk_size, chunk_size).to_rgb().into_vec().chunks(RGB_CHANNELS));

            if !chunk_set.contains(chunk) { 
                chunk_set.insert(chunk);
            };

            // for each cardinality

            // rotate 90
            // insert it
            // rotate 90
            // insert it
        });
    });

    chunk_set
}

pub fn is_sym(matrix: &DMatrix<i32>) -> bool {
    let transposed_matrix = matrix.transpose();
    if matrix.row(0).iter().count() != transposed_matrix.row(0).iter().count() { return false } 
    true
}

#[cfg(test)]
mod tests {

    use super::*;

    // we need to be more aware of when trait implementations imply things about syntactical constructs

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
    fn test_image_bytes() {
        let img = image::open("resources/test/City.png").unwrap();
        let img_rgb = img.into_rgb();
        // img_rgb.pixels().for_each(|pixel| println!("{:?}", pixel));
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
