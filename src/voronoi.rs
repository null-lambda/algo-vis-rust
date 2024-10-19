mod cmp {
    use std::cmp::Ordering;

    // x <= y iff x = y
    #[derive(Debug, Copy, Clone, Default)]
    pub struct Trivial<T>(pub T);

    impl<T> PartialEq for Trivial<T> {
        #[inline]
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl<T> Eq for Trivial<T> {}

    impl<T> PartialOrd for Trivial<T> {
        #[inline]
        fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
            Some(Ordering::Equal)
        }
    }

    impl<T> Ord for Trivial<T> {
        #[inline]
        fn cmp(&self, _other: &Self) -> Ordering {
            Ordering::Equal
        }
    }
}

#[macro_use]
pub mod geometry {
    use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

    use crate::utils::console_log;

    #[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
    pub struct Ordered<T>(T);

    impl From<f64> for Ordered<f64> {
        fn from(x: f64) -> Self {
            debug_assert!(!x.is_nan());
            Self(x)
        }
    }

    impl Into<f64> for Ordered<f64> {
        fn into(self) -> f64 {
            self.0
        }
    }

    impl From<f32> for Ordered<f64> {
        fn from(x: f32) -> Self {
            debug_assert!(x.is_finite());
            Self(x as f64)
        }
    }

    impl<T: PartialEq> Eq for Ordered<T> {}
    impl<T: PartialOrd> Ord for Ordered<T> {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    pub trait Scalar:
        Copy
        + Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + Div<Output = Self>
        + Neg<Output = Self>
        + PartialOrd
        + PartialEq
        + Default
        + std::fmt::Debug
    {
        fn zero() -> Self {
            Self::default()
        }

        fn one() -> Self;

        fn two() -> Self {
            Self::one() + Self::one()
        }

        fn min(self, other: Self) -> Self {
            if self < other {
                self
            } else {
                other
            }
        }

        fn max(self, other: Self) -> Self {
            if self < other {
                other
            } else {
                self
            }
        }

        fn abs(self) -> Self {
            if self < Self::zero() {
                -self
            } else {
                self
            }
        }

        fn sq(self) -> Self {
            self * self
        }
    }

    impl Scalar for f64 {
        fn one() -> Self {
            1.0
        }
    }
    // impl Scalar for i64 {}
    // impl Scalar for i32 {}

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    pub struct PointNd<const N: usize, T>(pub [T; N]);

    impl<const N: usize, T: Scalar> PointNd<N, T> {
        fn map<F, S>(self, mut f: F) -> PointNd<N, S>
        where
            F: FnMut(T) -> S,
        {
            PointNd(self.0.map(|x| f(x)))
        }
    }

    impl<const N: usize, T: Scalar> From<[T; N]> for PointNd<N, T> {
        fn from(p: [T; N]) -> Self {
            Self(p)
        }
    }

    impl<const N: usize, T: Scalar> Index<usize> for PointNd<N, T> {
        type Output = T;
        fn index(&self, i: usize) -> &Self::Output {
            &self.0[i]
        }
    }

    impl<const N: usize, T: Scalar> IndexMut<usize> for PointNd<N, T> {
        fn index_mut(&mut self, i: usize) -> &mut Self::Output {
            &mut self.0[i]
        }
    }

    macro_rules! impl_binop_dims {
        ($N:expr, $($idx:expr )+, $trait:ident, $fn:ident) => {
            impl<T: Scalar> $trait for PointNd<$N, T> {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    PointNd([$(self[$idx].$fn(other[$idx])),+])
                }
            }
        };
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident) => {
            impl_binop_dims!(2, 0 1, $trait, $fn);
            impl_binop_dims!(3, 0 1 2, $trait, $fn);
        };
    }

    impl_binop!(Add, add);
    impl_binop!(Sub, sub);
    impl_binop!(Mul, mul);
    impl_binop!(Div, div);

    impl<const N: usize, T: Scalar> Default for PointNd<N, T> {
        fn default() -> Self {
            PointNd([T::zero(); N])
        }
    }

    impl<const N: usize, T: Scalar> Neg for PointNd<N, T> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            PointNd(self.0.map(|x| -x))
        }
    }

    impl<const N: usize, T: Scalar> Mul<T> for PointNd<N, T> {
        type Output = Self;
        fn mul(self, k: T) -> Self::Output {
            PointNd(self.0.map(|x| x * k))
        }
    }

    impl<const N: usize, T: Scalar> PointNd<N, T> {
        pub fn zero() -> Self {
            Self::default()
        }

        pub fn dot(self, other: Self) -> T {
            self.0
                .into_iter()
                .zip(other.0)
                .map(|(a, b)| a * b)
                .reduce(|acc, x| acc + x)
                .unwrap()
        }

        pub fn cross(self, other: Point<T>) -> T {
            self[0] * other[1] - self[1] * other[0]
        }

        pub fn norm_sq(self) -> T {
            self.dot(self)
        }

        pub fn max_norm(self) -> T {
            self.0
                .into_iter()
                .map(|a| a.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap()
        }
    }

    pub type Point<T> = PointNd<2, T>;

    impl<T: Scalar> Point<T> {
        pub fn new(x: T, y: T) -> Self {
            Self([x, y])
        }

        pub fn rot(&self) -> Self {
            Point::new(-self[1], self[0])
        }
    }

    // predicate 1 for voronoi diagram
    pub fn signed_area<T: Scalar>(p: Point<T>, q: Point<T>, r: Point<T>) -> T {
        (q - p).cross(r - p)
    }

    pub fn line_intersection(
        p1: Point<f64>,
        dir1: Point<f64>,
        p2: Point<f64>,
        dir2: Point<f64>,
    ) -> Option<Point<f64>> {
        let denom = dir1.cross(dir2);
        let numer = p1 * denom + dir1 * (p2 - p1).cross(dir2);
        // f64ODO: add threshold
        let result = numer * denom.recip();
        (!result[0].is_nan() && !result[1].is_nan()).then(|| result)
    }

    // predicate 2 for voronoi diagram
    pub fn circumcenter(p: Point<f64>, q: Point<f64>, r: Point<f64>) -> Option<Point<f64>> {
        let d = f64::one() / (f64::one() + f64::one());
        line_intersection((p + q) * d, (q - p).rot(), (p + r) * d, (r - p).rot())
    }

    // predicate 3 for voronoi diagram
    pub fn breakpoint_x(left: Point<f64>, right: Point<f64>, sweepline: f64) -> f64 {
        let y1 = left[1] - sweepline;
        let y2 = right[1] - sweepline;
        let dx = right[0] - left[0];

        let a = y2 - y1;
        let b_2 = y1 * dx;
        let c = y1 * y2 * (y1 - y2) - y1 * dx * dx;

        let det_4 = b_2 * b_2 - a * c;
        let sign = a.signum() * dx.signum();
        let mut x = left[0] + (-b_2 - sign * det_4.max(0.0).sqrt()) / a;
        if !x.is_finite() {
            x = left[0] - c / b_2;
            if !x.is_finite() {
                x = left[0] + dx * 0.5;
            }
        }
        x
    }
}

// Assign an unique tag for each node, for debug print
#[cfg(debug_assertions)]
pub mod tag {
    pub type Tag = i32;
    static mut TAG: i32 = -1;
    pub fn next() -> Tag {
        unsafe {
            TAG += 1;
            TAG
        }
    }
}

// Doubly connected edge list
pub mod graph {
    use super::geometry::Point;

    pub const UNSET: usize = 1 << 31;
    pub const INF: usize = 1 << 30;

    #[derive(Debug)]
    pub struct Vertex {
        pub halfedge: usize,
    }

    #[derive(Debug)]
    pub struct HalfEdge {
        pub start: usize,
        pub face_left: usize,
        pub flip: usize,
    }

    #[derive(Debug)]
    pub struct Face {
        edge: usize,
    }

    #[derive(Debug)]
    pub struct Topology {
        pub verts: Vec<Vertex>,
        pub half_edges: Vec<HalfEdge>,
        pub faces: Vec<Face>,
    }

    #[derive(Debug)]
    pub struct Graph {
        pub topo: Topology,
        pub vert_coord: Vec<Point<f64>>,
        pub face_center: Vec<Point<f64>>,
    }
}

#[cfg(not(debug_assertions))]
pub mod tag {
    pub type Tag = ();
    pub fn next() -> Tag {}
}

pub mod builder {
    mod splay {
        // top-down splay tree for beachline
        use super::super::tag;

        use std::{
            borrow::BorrowMut,
            cmp::Ordering,
            fmt, iter, mem,
            ptr::{self, NonNull},
        };

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum Branch {
            Left = 0,
            Right = 1,
        }

        use Branch::{Left, Right};

        impl Branch {
            pub fn inv(&self) -> Self {
                match self {
                    Branch::Left => Branch::Right,
                    Branch::Right => Branch::Left,
                }
            }

            pub fn iter() -> iter::Chain<iter::Once<Self>, iter::Once<Self>> {
                iter::once(Branch::Left).chain(iter::once(Branch::Right))
            }
        }

        impl From<usize> for Branch {
            fn from(x: usize) -> Self {
                match x {
                    0 => Left,
                    1 => Right,
                    _ => panic!(),
                }
            }
        }

        type OwnedLink = Option<Box<Node>>;
        type RefLink = Option<NonNull<Node>>;

        #[derive(Debug)]
        pub struct Breakpoint {
            pub site_right: usize,
            pub halfedge: usize,
        }

        pub struct Node {
            tag: tag::Tag,
            pub children: [OwnedLink; 2], // binary search tree structure
            pub side: [RefLink; 2],       // linked list structure
            pub value: Breakpoint,
        }

        #[derive(Debug)]
        pub struct Tree {
            pub root: OwnedLink,
        }

        impl Node {
            fn new(value: Breakpoint) -> Self {
                Self {
                    tag: tag::next(),
                    children: [None, None],
                    side: [None, None],
                    value,
                }
            }

            fn link_sides(
                dir: Branch,
                mut lhs: impl BorrowMut<Self>,
                mut rhs: impl BorrowMut<Self>,
            ) {
                lhs.borrow_mut().side[dir.inv() as usize] = Some(NonNull::from(rhs.borrow()));
                rhs.borrow_mut().side[dir as usize] = Some(NonNull::from(lhs.borrow()));
            }
        }

        impl Tree {
            pub fn new() -> Self {
                Self { root: None }
            }

            pub fn splay_by<F>(self: &mut Self, mut cmp: F)
            where
                F: FnMut(&Node) -> Ordering,
            {
                let Some(root) = self.root.as_mut() else {
                    return;
                };
                let mut side_nodes = [vec![], vec![]];

                // cmp is called at most once for each nodes
                let mut ord = cmp(&root);
                loop {
                    let branch = match ord {
                        Ordering::Equal => break,
                        Ordering::Less => Left,
                        Ordering::Greater => Right,
                    };

                    let Some(mut child) = root.children[branch as usize].take() else {
                        break;
                    };
                    let child_ord = cmp(&child);

                    if child_ord == ord {
                        root.children[branch as usize] =
                            child.children[branch.inv() as usize].take();
                        mem::swap(root, &mut child);
                        root.children[branch.inv() as usize] = Some(child);

                        let Some(next_child) = root.children[branch as usize].take() else {
                            break;
                        };
                        child = next_child;

                        ord = cmp(&child);
                    } else {
                        ord = child_ord;
                    }
                    side_nodes[branch.inv() as usize].push(mem::replace(root, child));
                }

                for (branch, nodes) in side_nodes.into_iter().enumerate() {
                    root.children[branch] = nodes.into_iter().rev().fold(
                        root.children[branch].take(),
                        |acc, mut node| {
                            node.children[Branch::from(branch).inv() as usize] = acc;
                            Some(node)
                        },
                    );
                }
            }

            pub fn splay_first(&mut self) {
                self.splay_by(|_| Ordering::Less);
            }

            pub fn splay_last(&mut self) {
                self.splay_by(|_| Ordering::Greater);
            }

            pub fn insert(&mut self, branch: Branch, value: Breakpoint) {
                let mut new_node = Box::new(Node::new(value));
                let Some(root) = self.root.as_mut() else {
                    self.root = Some(new_node);
                    return;
                };

                let mut side = root.children[branch as usize].take();
                side.as_mut()
                    .map(|side| Node::link_sides(branch, side.as_mut(), new_node.as_mut()));
                Node::link_sides(branch, new_node.as_mut(), root.as_mut());

                new_node.children[branch as usize] = side;
                root.children[branch as usize] = Some(new_node);
            }

            pub fn pop_root(&mut self) -> Option<Box<Node>> {
                let mut root = self.root.take()?;

                let mut left = root.children[Left as usize].take();
                let mut right = root.children[Right as usize].take();

                unsafe {
                    match (left.as_mut(), right.as_mut()) {
                        (Some(_), Some(_)) => {
                            let mut prev = root.side[Left as usize].unwrap();
                            let mut next = root.side[Right as usize].unwrap();
                            Node::link_sides(Left, prev.as_mut(), next.as_mut());

                            let mut left = Tree { root: left };
                            left.splay_last();
                            let mut left = left.root.unwrap();
                            debug_assert!(left.children[Right as usize].is_none());

                            left.children[Right as usize] = right;
                            self.root = Some(left);
                        }
                        (Some(_), None) => {
                            root.side[Left as usize].unwrap().as_mut().side[Right as usize] = None;
                            self.root = left;
                        }
                        (None, Some(_)) => {
                            root.side[Right as usize].unwrap().as_mut().side[Left as usize] = None;
                            self.root = right;
                        }
                        (None, None) => {}
                    }
                    Some(root)
                }
            }

            // Test whether the linked list structure is valid
            pub fn validate_side_links(&self) {
                if !cfg!(debug_assertions) {
                    return;
                }

                // grab all nodes in inorder
                let mut inorder: Vec<&Box<Node>> = vec![];
                let mut current: Option<&Box<Node>> = self.root.as_ref();
                let mut stack: Vec<&Box<Node>> = vec![];
                while current.is_some() || !stack.is_empty() {
                    while let Some(node) = current {
                        stack.push(node);
                        current = node.children[Left as usize].as_ref();
                    }
                    current = stack.pop();
                    inorder.push(current.unwrap());
                    current = current.unwrap().children[Right as usize].as_ref();
                }

                // check the linked list structure
                for i in 0..inorder.len() {
                    let node: &Box<Node> = inorder[i];
                    let prev = (i >= 1).then(|| inorder[i - 1]);
                    let next = (i + 1 < inorder.len()).then(|| inorder[i + 1]);

                    unsafe {
                        fn option_nonnull_to_ptr<T>(x: Option<NonNull<T>>) -> *const T {
                            x.map_or(ptr::null(), |x| x.as_ptr())
                        }

                        fn option_box_to_ptr<T>(x: Option<&Box<T>>) -> *const T {
                            x.map_or(ptr::null(), |x| x.as_ref())
                        }

                        debug_assert!(
                            ptr::eq(
                                option_nonnull_to_ptr(node.side[Left as usize]),
                                option_box_to_ptr(prev)
                            ),
                            // "prev_next: {:?}, node_prev: {:?}",
                            // prev.map(|x| x.as_ref().side[Right as usize]),
                            // node.side[Left as usize].map(|x| x.as_ref())
                        );
                        debug_assert!(ptr::eq(
                            option_nonnull_to_ptr(node.side[Right as usize]),
                            option_box_to_ptr(next)
                        ));
                        if let Some(prev) = prev {
                            debug_assert!(ptr::eq(
                                option_nonnull_to_ptr(prev.as_ref().side[Right as usize]),
                                node.as_ref()
                            ));
                        }
                        if let Some(next) = next {
                            debug_assert!(ptr::eq(
                                option_nonnull_to_ptr(next.as_ref().side[Left as usize]),
                                node.as_ref()
                            ));
                        }
                    }
                }
            }
        }

        impl fmt::Debug for Node {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(
                    f,
                    "{} {:?} {}",
                    self.children[Left as usize]
                        .as_ref()
                        .map_or("_".to_owned(), |x| format!("({:?})", x)),
                    self.tag,
                    self.children[Right as usize]
                        .as_ref()
                        .map_or("_".to_owned(), |x| format!("({:?})", x)),
                )
            }
        }

        #[test]
        fn test_splay_tree() {
            //
            let mut tree = Tree::new();

            let gen_breakpoint = || Breakpoint {
                site_right: 0,
                halfedge: 0,
            };

            for branch in [Left, Left, Right, Left, Left, Right, Right] {
                tree.insert(branch, gen_breakpoint());
                tree.validate_side_links();
                println!("insert {branch:?}, {:?}", tree);
            }
            for i in 0..7 {
                if i % 3 == 0 {
                    tree.splay_first();
                    tree.validate_side_links();
                    println!("splay first, {:?}", tree);
                } else {
                    tree.splay_last();
                    tree.validate_side_links();
                    println!("splay last, {:?}", tree);
                }

                debug_assert!(tree.pop_root().is_some());
                tree.validate_side_links();
                println!("remove root, {:?}", tree);
            }
            debug_assert!(tree.pop_root().is_none());
        }
    }

    use core::f64;
    use std::{
        cmp::{Ordering, Reverse},
        collections::{BinaryHeap, HashSet},
        iter,
    };

    use splay::{Branch, Node};
    use wasm_bindgen::prelude::wasm_bindgen;

    use super::{
        cmp::Trivial,
        geometry::{self, Ordered, Point, PointNd},
    };

    use super::graph::{self, HalfEdge};

    #[derive(Debug)]
    pub enum Event {
        Site(Site),
        Circle(Circle),
    }

    #[derive(Debug)]
    pub struct Site {
        idx: usize,
    }

    #[derive(Debug)]
    pub struct Circle {
        idx_arc: usize,
        prev: usize,
        next: usize,
    }

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct EventKey(u32);

    #[derive(Debug)]
    struct RemovableHeap<T> {
        current_key: EventKey,
        heap: BinaryHeap<(T, Trivial<EventKey>)>,
        removed: HashSet<EventKey>,
    }

    impl<T: Ord> RemovableHeap<T> {
        fn new() -> Self {
            Self {
                current_key: EventKey(0),
                heap: Default::default(),
                removed: Default::default(),
            }
        }

        fn push(&mut self, value: T) -> EventKey {
            self.heap.push((value, Trivial(self.current_key)));
            let old_key = self.current_key;
            self.current_key.0 += 1;
            old_key
        }

        fn remove(&mut self, key: EventKey) {
            self.removed.insert(key);
        }

        fn pop(&mut self) -> Option<T> {
            while let Some((value, Trivial(key))) = self.heap.pop() {
                if !self.removed.contains(&key) {
                    return Some(value);
                }
            }
            None
        }
    }

    #[derive(Debug)]
    pub struct Builder {
        events: RemovableHeap<(Reverse<(Ordered<f64>, Ordered<f64>)>, Trivial<Event>)>,
        pub beachline: splay::Tree,
        pub directrix: Ordered<f64>,
        pub graph: graph::Graph,
        _init: bool,
    }

    impl Builder {
        pub fn new() -> Self {
            Self {
                events: RemovableHeap::new(),
                beachline: splay::Tree::new(),
                directrix: f64::NEG_INFINITY.into(),
                graph: graph::Graph {
                    topo: graph::Topology {
                        verts: vec![],
                        half_edges: vec![],
                        faces: vec![],
                    },
                    vert_coord: vec![],
                    face_center: vec![],
                },
                _init: false,
            }
        }

        pub fn add_points<I>(&mut self, points: I)
        where
            I: IntoIterator<Item = Point<f64>>,
        {
            // self.graph.face_center.extend(points);
            for p in points {
                self.graph.face_center.push(p);

                let idx = self.graph.face_center.len() - 1;
                self.events.push((
                    Reverse((p[1].into(), p[0].into())),
                    Trivial(Event::Site(Site { idx })),
                ));
            }
        }

        fn new_breakpoint(graph: &mut graph::Graph, sites: [usize; 2]) -> splay::Breakpoint {
            let idx_halfedge = graph.topo.half_edges.len();
            graph.topo.half_edges.push(HalfEdge {
                start: graph::INF,
                face_left: sites[0],
                flip: idx_halfedge + 1,
            });
            graph.topo.half_edges.push(HalfEdge {
                start: graph::UNSET,
                face_left: sites[1],
                flip: idx_halfedge,
            });
            splay::Breakpoint {
                site_right: sites[1],
                halfedge: idx_halfedge,
            }
        }

        fn eval_breakpoint_x(
            graph: &graph::Graph,
            sweep: Ordered<f64>,
            node: &Node,
        ) -> Ordered<f64> {
            let left = node.side[Branch::Left as usize];
            let left_x: Ordered<f64> = left
                .map_or(f64::NEG_INFINITY, |left| unsafe {
                    geometry::breakpoint_x(
                        graph.face_center[left.as_ref().value.site_right],
                        graph.face_center[node.value.site_right],
                        sweep.into(),
                    )
                })
                .into();
            left_x
        }

        pub fn init(&mut self) {
            debug_assert!(!self._init, "Builder::init must be called only once");
            self._init = true;

            let n: usize = self.graph.face_center.len();
            debug_assert!(n >= 1);
            // insert leftmost dummy breakpoint,
            // which has x = -infty (see eval_breakpoint_x)
            let Event::Site(Site { idx: idx_site }) = self.events.pop().unwrap().1 .0 else {
                unreachable!()
            };
            self.beachline.insert(
                Branch::Right,
                splay::Breakpoint {
                    site_right: idx_site,
                    halfedge: graph::UNSET,
                },
            );
        }

        pub fn step(&mut self) -> bool {
            debug_assert!(
                self._init,
                "Builder::init must be called before Builder::step"
            );

            let Some((Reverse(_y), Trivial(event))) = self.events.pop() else {
                return false;
            };
            match event {
                Event::Site(Site { idx: idx_site }) => {
                    let PointNd([px, py]) = self.graph.face_center[idx_site];
                    self.directrix = Ordered::from(py);

                    self.beachline.splay_by(|node| {
                        Ordered::from(px).cmp(&Self::eval_breakpoint_x(
                            &self.graph,
                            self.directrix,
                            node,
                        ))
                    });

                    let root = self.beachline.root.as_ref().unwrap();
                    match Ordered::from(px).cmp(&Self::eval_breakpoint_x(
                        &self.graph,
                        self.directrix,
                        root,
                    )) {
                        Ordering::Greater | Ordering::Equal => {
                            let old_site = root.value.site_right;
                            self.beachline.insert(
                                Branch::Right,
                                Self::new_breakpoint(&mut self.graph, [idx_site, old_site]),
                            );
                            self.beachline.insert(
                                Branch::Right,
                                Self::new_breakpoint(&mut self.graph, [old_site, idx_site]),
                            );
                        }
                        Ordering::Less => {
                            let old_site = root.children[Branch::Left as usize]
                                .as_ref()
                                .unwrap()
                                .value
                                .site_right;
                            self.beachline.insert(
                                Branch::Left,
                                Self::new_breakpoint(&mut self.graph, [old_site, idx_site]),
                            );
                            self.beachline.insert(
                                Branch::Left,
                                Self::new_breakpoint(&mut self.graph, [idx_site, old_site]),
                            );
                        }
                    }
                }
                Event::Circle(circle) => {}
            }
            println!("{:?}", self.beachline);
            true
        }

        pub fn run(&mut self) {
            self.init();
            while self.step() {}
        }
    }

    #[test]
    fn test_builder() {
        let mut seed = 42u64;
        let mut rng = std::iter::from_fn(move || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            Some(seed)
        });

        let mut builder = Builder::new();
        let n = 20;
        let mut points = vec![];
        for _ in 0..n {
            let x = rng.next().unwrap() % 100;
            let y = rng.next().unwrap() % 100;
            points.push(Point::new(x as f64, y as f64));
        }

        builder.add_points(points);
        builder.run();
    }
}
