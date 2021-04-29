pub mod graph;
pub mod io;
pub mod utils;
pub mod wfc;

#[allow(clippy::upper_case_acronyms)]
// pub type MSu16xNU = utote::MSu16x8<1>;
// pub type MSu16xNU = utote::MSu16<8>;
pub type MSu16xNU = utote::Multiset2<u16, 32>;  // cache line size!!!
