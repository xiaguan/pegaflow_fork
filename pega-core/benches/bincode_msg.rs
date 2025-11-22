use bincode;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;

fn generate_strings(n: usize, len: usize) -> Vec<String> {
    let mut rng = rand::rng();
    (0..n)
        .map(|_| {
            let s: String = (0..len)
                .map(|_| {
                    let c = rng.random_range(b'a'..=b'z') as char;
                    c
                })
                .collect();
            s
        })
        .collect()
}

fn bench_bincode_vec_string(c: &mut Criterion) {
    let cases = vec![(1000, 64), (2000, 64)];

    for (n, slen) in cases {
        let data = generate_strings(n, slen);

        let id = format!("bincode serialize {} strings of {} bytes", n, slen);

        c.bench_function(&id, |b| {
            b.iter(|| {
                let encoded =
                    bincode::encode_to_vec(black_box(&data), bincode::config::standard()).unwrap();
                black_box(encoded);
            })
        });
    }
}

criterion_group!(benches, bench_bincode_vec_string);
criterion_main!(benches);
