```
the variable formely known as N:
N = tileSize
```

tileChunk instead of tileSize allows for specifying a larger set of pixels to the smallest possible chunk that an image can be broken down in to.

chunkSize = pixel density of the a tile
tileSize = number of chunks in the side of a tile
stepSize?

1. Read in the image.
2. imageWidth % chunkSize * tileSize should eq 0 (same for height if not square)
3. Iterate through the image by in chunkSize * tileSize tiles in steps
4. Add overlapping relationships to vertices and store unique tile indexes
5. 

6. add orphan vertices


chunk
sub_chunk --> pixel dimensions of the sub chunks that make up a chunk
pixel

These are all square

1 --> 2 --> 4


Hi Guys, I wonder if someone could give some guidance on what's preferred stylistically when working with Rust procedures.

I was working on a project with a friend recently which splits an into chunks and then aliases this chunks before rotating them and adding into a hashmap.
```

```

--- 
Why can't you use an `impl Iterator` for a method that returns an Iterator in an `impl trait` block? I had the following code:
```rust
pub trait SubImage {
    fn sub_images(&self, sub_image_dim: u32) -> impl Iterator<Item=RgbImage>;
}

impl SubImage for RgbImage {

    fn sub_images(&self, sub_image_dim: u32) -> impl Iterator<Item=RgbImage>
    {
        let height_iter = (0..image.dimensions().0 - (chunk_size - 1));
        let width_iter = (0..image.dimensions().1 - (chunk_size - 1));
        height_iter.cartesian_product(width_iter)
                   .map(move |(y, x)| imageops::crop_imm(self, x, y, chunk_size, chunk_size).to_image())
    }
}
```

But it complains about the second `impl` inside the `impl` for the `SubImage` trait with `impl Trait not allowed outside of function and inherent method return types`. Is there a way to return an iterator from a trait like this?
