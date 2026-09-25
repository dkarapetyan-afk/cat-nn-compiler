use catnn_core::lens::{first_of_pair, identity_f64, PairState};
use proptest::prelude::*;

proptest! {
    #[test]
    fn first_of_pair_laws(a in -1e6f64..1e6, b in -1e6f64..1e6, v1 in -1e6f64..1e6, v2 in -1e6f64..1e6) {
        let lens = first_of_pair();
        let s = PairState { first: a, second: b };
        lens.check_laws(&s, &v1, &v2).unwrap();
    }

    #[test]
    fn identity_laws(s in -1e6f64..1e6, v1 in -1e6f64..1e6, v2 in -1e6f64..1e6) {
        identity_f64().check_laws(&s, &v1, &v2).unwrap();
    }
}
