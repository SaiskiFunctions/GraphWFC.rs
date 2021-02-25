use std::ops::RangeFrom;
use std::iter::{Chain, Repeat, Take, repeat};

#[derive(Clone)]
pub struct Limit {
    iter: Chain<Take<Repeat<usize>>, RangeFrom<usize>>
}

impl Limit {
    pub fn new(base: usize) -> Limit {
        Limit { iter: repeat(0).take(base).chain(1..) }
    }
}

impl Iterator for Limit {
    type Item = usize;

    fn next(&mut self) -> Option<usize> { self.iter.next() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_limit_iter() {
        let wave = Limit::new(3).take(10).collect::<Vec<_>>();
        assert_eq!(wave, vec![0, 0, 0, 1, 2, 3, 4, 5, 6, 7]);
    }
}