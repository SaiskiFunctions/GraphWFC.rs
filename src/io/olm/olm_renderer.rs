
use crate::graph::graph::{Graph, Vertices};
use crate::io::frame_padder::pad_frame;
use crate::utils::{index_to_coords, is_inside, coords_to_index};
use crate::io::post_processors::post_processor::PostProcessor;

use bimap::BiMap;
use hashbrown::HashMap;
use image::{imageops, Rgb, RgbImage, Pixel};
use itertools::Itertools;
use nalgebra::DMatrix;
use std::ops::{IndexMut, Index, AddAssign};
use std::convert::TryFrom;
use indexmap::IndexMap;
use std::ops::Not;

const GREEN: Rgb<u8> = Rgb([0, 255, 0]);

type Chunk = DMatrix<usize>;
type PixelKeys = BiMap<usize, Rgb<u8>>;

// we've maintained two render functions here because of the complexity
pub fn render(
    filename: &str,
    graph: Graph,
    key: &PixelKeys,
    chunks: &IndexMap<Chunk, u16>,
    (width, height): (usize, usize),
    chunk_size: usize,
    opt_post_processors: Option<Vec<impl PostProcessor<RgbImage>>>
) {
    let mut output_image: RgbImage = image::ImageBuffer::new(width as u32, height as u32);
    let graph_width = width / chunk_size; // in chunks
    let contradiction_key = key.len();

    graph
        .vertices
        .into_iter()
        .map(|labels| {
            labels
                .is_singleton()
                .then(|| chunks.get_index(labels.imax()).map(|t| t.0.clone()))
                .flatten()
        })
        .enumerate()
        .for_each(|(chunk_index, opt_chunk)| {
            let (chunk_x, chunk_y) = index_to_coords(chunk_index, graph_width);
            // project to pixel coordinates
            let top_left_pix_x = chunk_x * chunk_size;
            let top_left_pix_y = chunk_y * chunk_size;

            let chunk = match opt_chunk {
                None => DMatrix::from_element(chunk_size, chunk_size, contradiction_key),
                Some(chunk) => chunk
            };

            chunk
                .iter()
                .enumerate()
                .for_each(|(pixel_index, pixel_alias)| {
                    let (p_y, p_x) = index_to_coords(pixel_index, chunk_size);
                    let pixel_y = (top_left_pix_y + p_y) as u32;
                    let pixel_x = (top_left_pix_x + p_x) as u32;
                    let pixel = key.get_by_left(pixel_alias).copied().unwrap_or(GREEN);
                    output_image.put_pixel(pixel_x, pixel_y, pixel);
                });
        });

    match opt_post_processors {
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

fn vertex_by_alias_to_chunks(
    vertices: Vertices,
    chunks: &IndexMap<Chunk, u16>
) -> Vec<Vec<DMatrix<usize>>> {
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

fn multi_chunks_to_pixels(
    multi_chunks: Vec<Vec<DMatrix<usize>>>
) -> Vec<Rgb<u8>> {
    let multi_pixels = multi_chunks
        .map(|chunks| {
            chunks
            // a vec of DMatrix of pixel aliases
            .iter()
            // map each DMatrix of pixel aliases into an iterator rgb values
            .map(|chunk| {
                let chunk_pixels = chunk
                    .iter()
                    .map(|pixel_alias| {
                        key
                            .get_by_left(pixel_alias)
                            .copied()
                            .unwrap_or(GREEN)
                    });
                blend_chunk(chunk_pixels)
            })
        })
        .collect();
}

fn blend_chunk(
    chunk_pixels: Vec<Rgb<u8>>
) {
    let blend_ratio = 1.0 / chunks.len() as f64;
    chunk_pixels
        .fold(vec![vec![0, 0, 0]; chunk_size * chunk_size], |mut acc, chunk| {
            acc
                .iter_mut()
                // zip each pixel in acc with each pixel in chunk
                .zip(chunk) // iterator rgb values consumed here
                .for_each(|(acc_pixel, chunk_pixel)| {
                    acc_pixel
                        .iter_mut()
                        // zip the RGB channels of each pixel together
                        // i.e. r + r, g + g, b + b
                        .zip(chunk_pixel.channels())
                        .for_each(|(acc_channel, chunk_channel)| {
                            *acc_channel += (*chunk_channel as usize);
                        })
                });
            acc
        })
        .iter()
        .map(|sum_pixel| {
            let mut pixel = Rgb::from([0, 0, 0]);
            sum_pixel
                .iter()
                .zip(pixel.channels_mut())
                .for_each(|(sum_channel, pixel_channel)| {
                    let blend_channel = (sum_channel / chunks.len()) as u8;
                    *pixel_channel = blend_channel;
                });
            pixel
        });

}


pub fn render_2(
    filename: &str,
    graph: Graph,
    key: &PixelKeys,
    chunks: &IndexMap<Chunk, u16>,
    (width, height): (usize, usize),
    chunk_size: usize,
    opt_post_processors: Option<Vec<impl PostProcessor<RgbImage>>>
) {
    let mut output_image: RgbImage = image::ImageBuffer::new(width as u32, height as u32);
    let graph_width = width / chunk_size; // in chunks
    let contradiction_key = key.len();
    let output_frames = graphs.len();

    output_image = image::ImageBuffer::new(width as u32, height as u32);

    // vertices as chunk aliases to vertices as chunks
    let vertices_as_multi_chunks = vertex_by_alias_to_chunks(graph.vertices, chunks);

    let vertices_as_pixels = multi_chunks_to_pixels(vertices_as_multi_chunks);




        .enumerate()
        // for each chunk of the image which may contain multiple overlayed opt_chunks
        .for_each(|(chunk_index, chunks)| {
            let (chunk_x, chunk_y) = index_to_coords(chunk_index, graph_width);
            // project to pixel coordinates
            let top_left_pix_x = chunk_x * chunk_size;
            let top_left_pix_y = chunk_y * chunk_size;

            let blend_ratio = 1.0 / chunks.len() as f64;

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
                // sum each matching pixel for each chunk
                .fold(vec![vec![0, 0, 0]; chunk_size * chunk_size], |mut acc, chunk| {
                    acc
                        .iter_mut()
                        // zip each pixel in acc with each pixel in chunk
                        .zip(chunk) // iterator rgb values consumed here
                        .for_each(|(acc_pixel, chunk_pixel)| {
                            acc_pixel
                                .iter_mut()
                                // zip the RGB channels of each pixel together
                                // i.e. r + r, g + g, b + b
                                .zip(chunk_pixel.channels())
                                .for_each(|(acc_channel, chunk_channel)| {
                                    *acc_channel += (*chunk_channel as usize);
                                })
                        });
                    acc
                })
                .iter()
                .map(|sum_pixel| {
                    let mut pixel = Rgb::from([0, 0, 0]);
                    sum_pixel
                        .iter()
                        .zip(pixel.channels_mut())
                        .for_each(|(sum_channel, pixel_channel)| {
                            let blend_channel = (sum_channel / chunks.len()) as u8;
                            *pixel_channel = blend_channel;
                        });
                    pixel
                })
                .enumerate()
                .for_each(|(pixel_index, pixel)| {
                    let (p_y, p_x) = index_to_coords(pixel_index, chunk_size);
                    let pixel_y = (top_left_pix_y + p_y) as u32;
                    let pixel_x = (top_left_pix_x + p_x) as u32;
                    output_image.put_pixel(pixel_x, pixel_y, pixel);
                });
        });

    let filename_parts: Vec<&str> = filename.split(".").collect();

    let file_title = filename_parts[0];
    let file_extension = filename_parts[1];

    let frame_extension = pad_frame(output_frames, frame);

    let full_filename = format!("{}_{}.{}", file_title, frame_extension, file_extension);

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

    output_image.save(full_filename).unwrap();
}

pub fn progress_2(
    filename: &str,
    graphs: Vec<Vertices>,
    key: &PixelKeys,
    chunks: &IndexMap<Chunk, u16>,
    (width, height): (usize, usize),
    chunk_size: usize,
    opt_post_processors: Option<Vec<impl PostProcessor<RgbImage>>>
) {
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

            // generate filenames from frame
            let frame_name = filename;

            render_2(
                frame_name,
                graph,
                key,
                chunks,
                (width, height),
                chunk_size,
                opt_post_processors
            )
        })
}

pub fn render_3(
    filename: &str,
    graph: Graph,
    key: &PixelKeys,
    chunks: &IndexMap<Chunk, u16>,
    (width, height): (usize, usize),
    chunk_size: usize,
    opt_post_processors: Option<Vec<impl PostProcessor<RgbImage>>>
) {
    let mut output_image: RgbImage = image::ImageBuffer::new(width as u32, height as u32);
    let graph_width = width / chunk_size; // in chunks
    let contradiction_key = key.len();
    let output_frames = graphs.len();

    output_image = image::ImageBuffer::new(width as u32, height as u32);
    graph
        .vertices
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
        .enumerate()
        // for each chunk of the image which may contain multiple overlayed opt_chunks
        .for_each(|(chunk_index, chunks)| {
            let (chunk_x, chunk_y) = index_to_coords(chunk_index, graph_width);
            // project to pixel coordinates
            let top_left_pix_x = chunk_x * chunk_size;
            let top_left_pix_y = chunk_y * chunk_size;

            let blend_ratio = 1.0 / chunks.len() as f64;

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
                // sum each matching pixel for each chunk
                .fold(vec![vec![0, 0, 0]; chunk_size * chunk_size], |mut acc, chunk| {
                    acc
                        .iter_mut()
                        // zip each pixel in acc with each pixel in chunk
                        .zip(chunk) // iterator rgb values consumed here
                        .for_each(|(acc_pixel, chunk_pixel)| {
                            acc_pixel
                                .iter_mut()
                                // zip the RGB channels of each pixel together
                                // i.e. r + r, g + g, b + b
                                .zip(chunk_pixel.channels())
                                .for_each(|(acc_channel, chunk_channel)| {
                                    *acc_channel += (*chunk_channel as usize);
                                })
                        });
                    acc
                })
                .iter()
                .map(|sum_pixel| {
                    let mut pixel = Rgb::from([0, 0, 0]);
                    sum_pixel
                        .iter()
                        .zip(pixel.channels_mut())
                        .for_each(|(sum_channel, pixel_channel)| {
                            let blend_channel = (sum_channel / chunks.len()) as u8;
                            *pixel_channel = blend_channel;
                        });
                    pixel
                })
                .enumerate()
                .for_each(|(pixel_index, pixel)| {
                    let (p_y, p_x) = index_to_coords(pixel_index, chunk_size);
                    let pixel_y = (top_left_pix_y + p_y) as u32;
                    let pixel_x = (top_left_pix_x + p_x) as u32;
                    output_image.put_pixel(pixel_x, pixel_y, pixel);
                });
        });

    let filename_parts: Vec<&str> = filename.split(".").collect();

    let file_title = filename_parts[0];
    let file_extension = filename_parts[1];

    let frame_extension = pad_frame(output_frames, frame);

    let full_filename = format!("{}_{}.{}", file_title, frame_extension, file_extension);

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

    output_image.save(full_filename).unwrap();
}

pub fn progress(
    filename: &str,
    graphs: Vec<Vertices>,
    key: &PixelKeys,
    chunks: &IndexMap<Chunk, u16>,
    (width, height): (usize, usize),
    chunk_size: usize,
    opt_post_processors: Option<Vec<impl PostProcessor<RgbImage>>>
) {
    let mut output_image: RgbImage = image::ImageBuffer::new(width as u32, height as u32);
    let graph_width = width / chunk_size; // in chunks
    let contradiction_key = key.len();
    let output_frames = graphs.len();

    graphs
        .into_iter()
        .enumerate()
        .for_each(|(frame, graph)| {
            output_image = image::ImageBuffer::new(width as u32, height as u32);
            graph
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
                .enumerate()
                // for each chunk of the image which may contain multiple overlayed opt_chunks
                .for_each(|(chunk_index, chunks)| {
                    let (chunk_x, chunk_y) = index_to_coords(chunk_index, graph_width);
                    // project to pixel coordinates
                    let top_left_pix_x = chunk_x * chunk_size;
                    let top_left_pix_y = chunk_y * chunk_size;

                    let blend_ratio = 1.0 / chunks.len() as f64;

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
                        // sum each matching pixel for each chunk
                        .fold(vec![vec![0, 0, 0]; chunk_size * chunk_size], |mut acc, chunk| {
                            acc
                                .iter_mut()
                                // zip each pixel in acc with each pixel in chunk
                                .zip(chunk) // iterator rgb values consumed here
                                .for_each(|(acc_pixel, chunk_pixel)| {
                                    acc_pixel
                                        .iter_mut()
                                        // zip the RGB channels of each pixel together
                                        // i.e. r + r, g + g, b + b
                                        .zip(chunk_pixel.channels())
                                        .for_each(|(acc_channel, chunk_channel)| {
                                            *acc_channel += (*chunk_channel as usize);
                                        })
                                });
                            acc
                        })
                        .iter()
                        .map(|sum_pixel| {
                            let mut pixel = Rgb::from([0, 0, 0]);
                            sum_pixel
                                .iter()
                                .zip(pixel.channels_mut())
                                .for_each(|(sum_channel, pixel_channel)| {
                                    let blend_channel = (sum_channel / chunks.len()) as u8;
                                    *pixel_channel = blend_channel;
                                });
                            pixel
                        })
                        .enumerate()
                        .for_each(|(pixel_index, pixel)| {
                            let (p_y, p_x) = index_to_coords(pixel_index, chunk_size);
                            let pixel_y = (top_left_pix_y + p_y) as u32;
                            let pixel_x = (top_left_pix_x + p_x) as u32;
                            output_image.put_pixel(pixel_x, pixel_y, pixel);
                        });
                });

            let filename_parts: Vec<&str> = filename.split(".").collect();

            let file_title = filename_parts[0];
            let file_extension = filename_parts[1];

            let frame_extension = pad_frame(output_frames, frame);

            let full_filename = format!("{}_{}.{}", file_title, frame_extension, file_extension);

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

            output_image.save(full_filename).unwrap();
        });
}