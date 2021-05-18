
use crate::graph::graph::{Graph, Vertices};
use crate::io::frame_padder::pad_frame;
use crate::utils::{index_to_coords, is_inside, coords_to_index};
use crate::io::post_processors::post_processor::PostProcessor;
use crate::io::pixel_utils::PixelOperations;

use bimap::BiMap;
use hashbrown::HashMap;
use image::{imageops, Rgb, RgbImage, Pixel};
use itertools::Itertools;
use nalgebra::DMatrix;
use std::ops::{IndexMut, Index, AddAssign};
use std::convert::TryFrom;
use indexmap::IndexMap;
use std::ops::Not;
use crate::MSu16xNU;

const GREEN: Rgb<u8> = Rgb([0, 255, 0]);

type Chunk = DMatrix<usize>;
type PixelKeys = BiMap<usize, Rgb<u8>>;

pub fn render(
    filename: &str,
    graph: Graph,
    key: &PixelKeys,
    chunks: &IndexMap<Chunk, u16>,
    (width, height): (usize, usize),
    chunk_size: usize,
    opt_post_processors: &Option<Vec<impl PostProcessor<RgbImage>>>
) {
    let mut output_image: RgbImage = image::ImageBuffer::new(width as u32, height as u32);
    let graph_width = width / chunk_size; // in chunks

    output_image = image::ImageBuffer::new(width as u32, height as u32);

    let vertices_as_chunks = vertices_to_chunks(graph.vertices, chunks, chunk_size, key.len());

    for (vertex_index, vertex_chunks) in vertices_as_chunks.into_iter().enumerate() {
        let (vertex_x, vertex_y) = index_to_coords(vertex_index, graph_width);
        // project to pixel coordinates
        let top_left_pix_x = vertex_x * chunk_size;
        let top_left_pix_y = vertex_y * chunk_size;

        let vertex_as_pixels = chunks_to_pixels(vertex_chunks, key, chunk_size);

        for (pixel_index, pixel) in vertex_as_pixels.into_iter().enumerate() {
            let (p_y, p_x) = index_to_coords(pixel_index, chunk_size);
            let pixel_y = (top_left_pix_y + p_y) as u32;
            let pixel_x = (top_left_pix_x + p_x) as u32;
            output_image.put_pixel(pixel_x, pixel_y, pixel);
        }
    }

    // do any post-processing on the image
    match &opt_post_processors {
        Some(post_processors) => {
            post_processors
                .iter()
                .for_each(|post_processor| {
                    output_image = post_processor.process(&output_image)
                })
        },
        _ => println!("No post processors installed for this render.")
    }

    output_image.save(filename).unwrap();
}

pub fn progress(
    filename: &str,
    graphs: Vec<Vertices>,
    key: &PixelKeys,
    chunks: &IndexMap<Chunk, u16>,
    (width, height): (usize, usize),
    chunk_size: usize,
    opt_post_processors: &Option<Vec<impl PostProcessor<RgbImage>>>
) {
    let output_frames = graphs.len();

    graphs
        .into_iter()
        .enumerate()
        .for_each(|(frame, vertices)| {
            let graph = Graph::new(
                vertices,
                // default values to satisfy function input
                HashMap::new(),
                MSu16xNU::empty()
            );

            // generate frame filename
            let filename_parts: Vec<&str> = filename.split(".").collect();

            let file_title = filename_parts[0];
            let file_extension = filename_parts[1];

            let frame_extension = pad_frame(output_frames, frame);

            let frame_name = format!("{}_{}.{}", file_title, frame_extension, file_extension);

            render(
                &frame_name,
                graph,
                key,
                chunks,
                (width, height),
                chunk_size,
                opt_post_processors
            )
        })
}

fn vertices_to_chunks(
    vertices: Vertices,
    chunks: &IndexMap<Chunk, u16>,
    chunk_size: usize,
    contradiction_key: usize,
) -> Vec<Vec<Chunk>> {
    vertices
        .into_iter()
        // Vec<Multiset> => Vec<Vec<DMatrix>>
        // map each non-zero multiset label into a corresponding DMatrix
        .map(|vertex| {
            vertex
                .into_iter()
                .enumerate()
                .fold(Vec::new(), |mut acc, (label, frequency)| {
                    if frequency > 0 {
                        let (chunk, _) = chunks.get_index(label).unwrap();
                        acc.push(chunk.clone())
                    }
                    acc
                })
        })
        .map(|chunks| {
            match chunks.is_empty() {
                true => vec![DMatrix::from_element(chunk_size, chunk_size, contradiction_key)],
                false => chunks
            }
        })
        .collect()
}

fn chunks_to_pixels(chunks: Vec<Chunk>, key: &PixelKeys, chunk_size: usize) -> Vec<Rgb<u8>> {
    chunks
        // a vec of DMatrix of pixel aliases
        .iter()
        // map each DMatrix of pixel aliases into an iterator rgb values
        .map(|chunk| {
            chunk
                .iter()
                .map(|pixel_alias| {
                    key
                        .get_by_left(pixel_alias)
                        .copied()
                        .unwrap_or(GREEN)
                })
        })
        // sum each matching pixel values for each chunk
        // using usize to avoid capping on u8 channel size
        .fold(vec![[0; 3]; chunk_size * chunk_size], |mut acc, chunk| {
            acc
                .iter_mut()
                // zip each pixel in acc with each pixel in chunk
                .zip(chunk)
                // add each pixel together
                .for_each(|(acc_pixel, chunk_pixel)| {
                    acc_pixel[0] += chunk_pixel.channels()[0] as usize;
                    acc_pixel[1] += chunk_pixel.channels()[1] as usize;
                    acc_pixel[2] += chunk_pixel.channels()[2] as usize;
                });
            acc
        })
        .iter()
        // blend summed pixel values by dividing by length of chunks
        // unfortunately its way more complex to do this iteratively so accept code duplication
        .map(|sum_pixel| {
            Rgb::from([
                (sum_pixel[0] / chunks.len()) as u8,
                (sum_pixel[1] / chunks.len()) as u8,
                (sum_pixel[2] / chunks.len()) as u8,
            ])
        })
        .collect()
}