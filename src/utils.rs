use hashbrown::{HashMap, HashSet};
use std::hash::Hash;
use std::fmt::{Display, Formatter};
use std::fmt;
use std::collections::BTreeMap;

pub fn hash_set<T>(data: &[T]) -> HashSet<T>
    where
        T: Hash + Eq + Clone,
{
    data.iter().cloned().collect()
}

pub fn hash_map<K, V>(data: &[(K, V)]) -> HashMap<K, V>
    where
        K: Hash + Eq + Clone,
        V: Clone,
{
    data.iter().cloned().collect()
}

#[derive(Clone)]
pub struct Accumulator{
    data: Vec<f64>
}

impl Accumulator {
    pub fn new() -> Accumulator { Accumulator { data: Vec::new() } }

    pub fn from_vec(vec: Vec<f64>) -> Accumulator { Accumulator { data: vec } }

    pub fn push(&mut self, value: f64) { self.data.push(value) }

    fn samples(&self) -> usize { self.data.len() }

    fn sum(&self) -> f64 { self.data.iter().sum() }

    fn avg(&self) -> f64 { self.sum() / self.samples() as f64 }

    fn median(&self) -> f64 { self.data[self.samples() / 2] }

    fn max(&self) -> f64 {
        self.data.iter()
            .copied()
            .min_by(|a, b| b.partial_cmp(a).expect("Tried to compare a NaN"))
            .unwrap()
    }

    fn min(&self) -> f64 {
        self.data.iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).expect("Tried to compare a NaN"))
            .unwrap()
    }

    fn diffs(&self) -> Vec<f64> {
        self.data.windows(2).map(|window| {
            let x = window[0];
            let y = window[1];
            y - x
        }).collect()
    }
}

impl Display for Accumulator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut output = String::new();
        output.push_str(&format!("* samples: {}\n", self.samples()));
        output.push_str(&format!("* avg: {}\n", self.avg()));
        output.push_str(&format!("* median: {}\n", self.median()));
        output.push_str(&format!("* max: {}\n", self.max()));
        output.push_str(&format!("* min: {}\n", self.min()));
        // output.push_str(&format!("* values: {:?}\n", self.data));
        // output.push_str(&format!("* diffs: {:?}\n", self.diffs()));
        output.push_str("---\n");
        f.write_str(&output)
    }
}

pub struct Metrics<'a> {
    counters: BTreeMap<&'a str, usize>,
    accumulators: BTreeMap<&'a str, Accumulator>,
    averages: BTreeMap<&'a str, (&'a str, &'a str)>,
}

impl<'a> Default for Metrics<'a> {
    fn default() -> Self {
        Metrics::new()
    }
}

impl<'a> Metrics<'a> {
    pub fn new() -> Metrics<'a> {
        Metrics {
            counters: BTreeMap::new(),
            accumulators: BTreeMap::new(),
            averages: BTreeMap::new(),
        }
    }

    pub fn inc(&mut self, key: &'a str) {
        self.counters.entry(key).and_modify(|v| *v += 1).or_insert(1);
    }

    pub fn inc_by(&mut self, key: &'a str, value: usize) {
        self.counters.entry(key).and_modify(|v| *v += value).or_insert(value);
    }

    pub fn init_counter(&mut self, key: &'a str, value: usize) {
        self.counters.insert(key, value);
    }

    pub fn get_counter(&self, key: &'a str) -> Option<&usize> {
        self.counters.get(key)
    }

    pub fn acc(&mut self, key: &'a str, value: f64) {
        self.accumulators
            .entry(key)
            .and_modify(|a| a.push(value))
            .or_insert_with(|| Accumulator::from_vec(vec![value]));
    }

    pub fn init_acc(&mut self, key: &'a str, value: Vec<f64>) {
        self.accumulators.insert(key, Accumulator::from_vec(value));
    }

    pub fn get_acc(&self, key: &'a str) -> Option<&Accumulator> {
        self.accumulators.get(key)
    }

    pub fn avg(&mut self, key: &'a str, value: (&'a str, &'a str)) {
        self.averages.insert(key, value);
    }

    pub fn print(&self, msg: Option<&str>) {
        msg.iter().for_each(|string| println!("{}", string));
        println!("{}", self)
    }
}

impl Display for Metrics<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut output = String::new();
        if !self.counters.is_empty() {
            output.push_str("-- Counters --\n");
            self.counters.iter().for_each(|(&k, counter)| {
                output.push_str(&format!("{}: {}\n", k, counter));
            });
            output.push('\n')
        }

        if !self.accumulators.is_empty() {
            output.push_str("-- Accumulators --\n");
            self.accumulators.iter().for_each(|(&k, acc)| {
                output.push_str(&format!("{}:\n{}", k, acc));
            });
            output.push('\n')
        }

        let avg_results: Vec<(String, f64)> = self.averages.iter().filter_map(|(k, &v)| {
            self.counters.get(v.0).zip(self.counters.get(v.1)).map(|(value1, value2)| {
                let result = *value1 as f64 / *value2 as f64;
                (k.to_string(), result)
            })
        }).collect();
        if !avg_results.is_empty() {
            output.push_str("-- Aggregations --\n");
            avg_results.iter().for_each(|(k, result)| {
                output.push_str(&format!("{}: {}\n", k, result));
            });
            output.push('\n')
        }

        f.write_str(&output)
    }
}

pub fn index_to_coords(index: usize, width: usize) -> (usize, usize) {
    (index % width, index / width)
}

pub fn coords_to_index(x: usize, y: usize, width: usize) -> usize {
    x + y * width
}

pub fn is_inside((x, y): (i32, i32), (w, h): (usize, usize)) -> bool {
    x >= 0 && y >= 0 && x < w as i32 && y < h as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_to_coords() {
        assert_eq!(index_to_coords(4, 3), (1, 1));
        assert_eq!(index_to_coords(4, 4), (0, 1));
        assert_eq!(index_to_coords(11, 3), (2, 3));
    }

    #[test]
    fn test_coords_to_index() {
        assert_eq!(coords_to_index(2, 1, 3), 5);
        assert_eq!(coords_to_index(0, 1, 4), 4);
    }

    #[test]
    fn test_is_inside() {
        assert!(!is_inside((-1, 0), (3, 3)));
        assert!(!is_inside((0, 4), (4, 4)));
    }
}
