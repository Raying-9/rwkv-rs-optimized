use burn::prelude::{Backend, Tensor};

fn adjust_dim<const D: usize>(dim: isize) -> usize {
    let adjusted_dim = if dim < 0 {
        let dim = dim + D as isize;

        assert!(dim >= 0, "Dimension out of range (adjusted_dim={})", dim);

        dim as usize
    } else {
        dim as usize
    };

    assert!(
        adjusted_dim < D,
        "Dimension out of range (input has {} dimensions, but got dim={})",
        D,
        dim
    );

    adjusted_dim
}

pub fn normalize<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    p: f32,
    dim: isize,
    epsilon: f32,
) -> Tensor<B, D> {
    let adjusted_dim = adjust_dim::<D>(dim);

    let norm = input
        .clone()
        .abs()
        .powf_scalar(p)
        .sum_dim(adjusted_dim)
        .powf_scalar(1.0 / p);

    let denom = norm.clamp_min(epsilon);

    input.div(denom)
}

pub fn normalize_l2<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    dim: isize,
    epsilon: f32,
) -> Tensor<B, D> {
    let adjusted_dim = adjust_dim::<D>(dim);

    let norm = input.clone().powf_scalar(2.0).sum_dim(adjusted_dim).sqrt();
    let denom = norm.clamp_min(epsilon);

    input.div(denom)
}
