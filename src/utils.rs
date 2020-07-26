use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::iter::FromIterator;

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
