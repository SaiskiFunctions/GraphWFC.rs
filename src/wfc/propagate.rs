use crate::graph::graph::{EdgeDirection, VertexIndex};

#[derive(Debug, Clone, Copy)]
pub struct Propagate {
    pub from: VertexIndex,
    pub to: VertexIndex,
    pub direction: EdgeDirection,
}

impl Propagate {
    pub fn new(from: VertexIndex, to: VertexIndex, direction: EdgeDirection) -> Propagate {
        Propagate {
            from,
            to,
            direction,
        }
    }
}
