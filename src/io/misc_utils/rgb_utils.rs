use image::{Rgb, Pixel, Primitive};

pub trait PixelOperations<D>
where
    D: Primitive
{
    fn add<T: Primitive + 'static>(&self, x: Rgb<T>) -> Rgb<D>;
}

impl<D: 'static> PixelOperations<D> for Rgb<D>
where
    D: Primitive
{
    fn add<T: Primitive + 'static>(&self, x: Rgb<T>) -> Rgb<D>
    {
        Rgb::from([
            self.channels()[0] + D::from(x.channels()[0]).unwrap(),
            self.channels()[1] + D::from(x.channels()[1]).unwrap(),
            self.channels()[2] + D::from(x.channels()[2]).unwrap()
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_pixels() {
        let small_pixel = Rgb::<u8>::from([100, 20, 5]);
        let large_pixel = Rgb::<usize>::from([10, 40, 1000]);
        let combined_pixel = large_pixel.add(small_pixel);
        assert_eq!(combined_pixel.channels()[0], 110);
        assert_eq!(combined_pixel.channels()[1], 60);
        assert_eq!(combined_pixel.channels()[2], 1005);
    }
}