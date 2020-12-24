use hashbrown::{HashMap, HashSet};
use std::hash::Hash;
use std::iter::FromIterator;
use std::fmt::{Display, Formatter};
use std::fmt;
use std::collections::BTreeMap;

pub fn hash_set<T>(data: &[T]) -> HashSet<T>
    where
        T: Hash + Eq + Clone,
{
    HashSet::from_iter(data.iter().cloned())
}

pub fn hash_map<K, V>(data: &[(K, V)]) -> HashMap<K, V>
    where
        K: Hash + Eq + Clone,
        V: Clone,
{
    HashMap::from_iter(data.iter().cloned())
}

pub struct Metrics<'a> {
    counters: BTreeMap<&'a str, i32>,
    accumulators: BTreeMap<&'a str, Vec<i32>>,
    averages: BTreeMap<&'a str, (&'a str, &'a str)>
}

impl<'a> Metrics<'a> {
    pub fn new() -> Metrics<'a> {
        Metrics {
            counters: BTreeMap::new(),
            accumulators: BTreeMap::new(),
            averages: BTreeMap::new()
        }
    }

    pub fn inc(&mut self, key: &'a str) {
        self.counters.entry(key).and_modify(|v| *v += 1).or_insert(1);
    }

    pub fn dec(&mut self, key: &'a str) {
        self.counters.entry(key).and_modify(|v| *v -= 1).or_insert(-1);
    }

    pub fn init_counter(&mut self, key: &'a str, value: i32) {
        self.counters.insert(key, value);
    }

    pub fn get_counter(&self, key: &'a str) -> Option<i32> {
        self.counters.get(key).cloned()
    }

    pub fn acc(&mut self, key: &'a str, value: i32) {
        self.accumulators.entry(key).and_modify(|a| a.push(value)).or_insert(vec![value]);
    }

    pub fn init_acc(&mut self, key: &'a str, value: Vec<i32>) {
        self.accumulators.insert(key, value);
    }

    pub fn get_acc(&self, key: &'a str) -> Option<Vec<i32>> {
        self.accumulators.get(key).cloned()
    }

    pub fn avg(&mut self, key: &'a str, value: (&'a str, &'a str)) {
        self.averages.insert(key, value);
    }

    pub fn print(&self, msg: Option<&str>) {
        if let Some(string) = msg {
            println!("{}", string)
        }
        println!("{}", self)
    }
}

impl Display for Metrics<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut output = String::new();
        if !self.counters.is_empty() {
            output.push_str("-- Counters --\n");
            self.counters.iter().for_each(|(&k, v)| {
                output = format!("{}{}: {}\n", output, k, v);
            });
            output.push_str("\n")
        }

        if !self.accumulators.is_empty() {
            output.push_str("-- Accumulators --\n");
            self.accumulators.iter().for_each(|(&k, v)| {
                output = format!("{}{}: {:?}\n", output, k, v);
            });
            output.push_str("\n")
        }

        let avg_results: Vec<(String, f64)> = self.averages.iter().filter_map(|(k, &v)| {
            if let (Some(value1), Some(value2)) = (self.counters.get(v.0), self.counters.get(v.1)) {
                let result = *value1 as f64 / *value2 as f64;
                Some((k.to_string(), result))
            } else { None }
        }).collect();
        if !avg_results.is_empty() {
            output.push_str("-- Calculations --\n");
            avg_results.iter().for_each(|(k, result)| {
                output = format!("{}{}: {}\n", output, k, result)
            });
            output.push_str("\n")
        }

        write!(f, "{}", output)
    }
}
