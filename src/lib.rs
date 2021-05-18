pub mod graph;
pub mod io;
pub mod utils;
pub mod wfc;

#[allow(clippy::upper_case_acronyms)]
pub type MSu16xNU = utote::Multiset<u16, 4>;  // cache line size!!!
