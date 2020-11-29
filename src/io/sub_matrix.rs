use nalgebra::{DMatrix};

pub trait SubMatrix {
    fn crop_left(self, offset: usize) -> DMatrix<u32>;
    fn crop_right(self, offset: usize) -> DMatrix<u32>;
    fn crop_top(self, offset: usize) -> DMatrix<u32>;
    fn crop_bottom(self, offset: usize) -> DMatrix<u32>;
    fn sub_matrix(&self, position: (u32, u32), size: (u32, u32)) -> DMatrix<u32>;
}

impl SubMatrix for DMatrix<u32> {
    fn crop_left(self, offset: usize) -> DMatrix<u32> {
        self.remove_columns(0, offset)
    }

    fn crop_right(self, offset: usize) -> DMatrix<u32> {
        let cols = self.ncols();
        if offset >= cols { return self }
        self.remove_columns(offset, cols - offset)
    }

    fn crop_top(self, offset: usize) -> DMatrix<u32> {
        self.remove_rows(0, offset)
    }

    fn crop_bottom(self, offset: usize) -> DMatrix<u32> {
        let rows = self.nrows();
        if offset >= rows { return self }
        self.remove_rows(offset, rows - offset)
    }

    fn sub_matrix(&self, position: (u32, u32), size: (u32, u32)) -> DMatrix<u32> {
        self
            .clone()
            .crop_left(position.0 as usize)
            .crop_right(size.0 as usize)
            .crop_top(position.1 as usize)
            .crop_bottom(size.1 as usize)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_sub_matrix() {
        let v = vec![0, 1, 2,
                     3, 4, 5,
                     6, 7, 8];
        //                                   ┏━━━━━ Height
        //                                   ┃
        //                                   ┃  ┏━━ Width
        //                                   V  V
        let matrix = DMatrix::from_row_slice(3, 3, &v);

        let x = vec![4, 5];
        let target_a = DMatrix::from_row_slice(1, 2, &x);

        let x = vec![1, 2, 4, 5];
        let target_b = DMatrix::from_row_slice(2, 2, &x);

        let x = vec![0, 1, 2];
        let target_c = DMatrix::from_row_slice(1, 3, &x);

        let x = vec![4];
        let target_d = DMatrix::from_row_slice(1, 1, &x);

        //                                    ┏━━━━━ Width
        //                                    ┃
        //                                    ┃  ┏━━ Height
        //                                    V  V
        assert_eq!(matrix.sub_matrix((0, 0), (3, 3)), matrix);

        assert_eq!(matrix.sub_matrix((1, 1), (2, 1)), target_a);

        assert_eq!(matrix.sub_matrix((1, 0), (2, 2)), target_b);

        assert_eq!(matrix.sub_matrix((0, 0), (3, 1)), target_c);

        assert_eq!(matrix.sub_matrix((1, 1), (1, 1)), target_d);
    }
}