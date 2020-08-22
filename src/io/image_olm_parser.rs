use image::{GenericImageView, DynamicImage, RgbImage};
use nalgebra::{DMatrix, Matrix2};
use nalgebra::geometry::Rotation2;
use std::collections::HashSet;

static RGB_CHANNELS: u8 = 3;

pub fn parse() {
    let img = image::open("resources/test/City.png").unwrap();
    // 
    // chunk {
    //     vec![4]
    // }
}

// fn chunk_image(image: &DynamicImage) -> HashSet<Vec<u8>> {
//     let chunk_size = 2;
//     let sub_chunk_size = 1; // pixels per sub chunk

//     let (height, width) = image.dimensions();

//     let mut chunk_set: HashSet<Vec<u8>> = HashSet::new();

//     (0..height - (chunk_size - 1)).for_each(|y| {
//         (0..width - (chunk_size - 1)).for_each(|x| {
//             // get vec of pixel channels
//             chunk = DMatrix.for_row_vector(chunk_size, chunk_size, image.crop_imm(x, y, chunk_size, chunk_size).to_rgb().into_vec().chunks(RGB_CHANNELS));

//             if !chunk_set.contains(chunk) { 
//                 insert(chunk);
//             };

//             // for each cardinality
//         });
//     });

//     chunk_set
// }

#[cfg(test)]
mod tests {

    use super::*;

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
        let basic_matrix = DMatrix::from_row_slice(2, 2, &vec![[2, 2], [3, 3], [4, 4], [5, 5]]);
        let rotation_matrix = DMatrix::from_row_slice(2, 2, &vec![0, -1, 1, 0]);
        let rotated_matrix = basic_matrix * rotation_matrix;
        println!("{:?}", rotated_matrix);
    }

    //cargo test --package wfc-rust --lib io::image_olm_parser::tests -- --nocapture 
    // [(255, 14, 0), (255, 255, 255), (52, 0, 254), (0, 255, 11)] * [0, 1, -1, 0] / [0,-1, 1, 0]
    // 0 1
    // 2 3


    // 2 3
    // 4 5

    // 3 5
    // 2 4
}
