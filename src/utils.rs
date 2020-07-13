use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::iter::FromIterator;

pub fn hash_set<T: Hash + Eq + Clone>(data: &[T]) -> HashSet<T> {
    HashSet::from_iter(data.iter().cloned())
}

pub fn hash_map<K: Hash + Eq + Clone, V: Clone>(data: &[(K, V)]) -> HashMap<K, V> {
    HashMap::from_iter(data.iter().cloned())
}
