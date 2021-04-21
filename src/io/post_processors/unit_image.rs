use image::{Rgb, RgbImage};
use crate::io::post_processors::post_processor::PostProcessor;

// extremely simply image post processor for demonstration purposes
pub struct UnitImage {
    color: Rgb<u8>
}

impl UnitImage {
    pub fn new(color: Rgb<u8>) -> UnitImage { UnitImage { color } }
}

impl PostProcessor<RgbImage> for UnitImage {
    fn process(self, input: &RgbImage) -> RgbImage {
        let (width, height) = input.dimensions();
        RgbImage::from_pixel(width, height, self.color)
    }
}
