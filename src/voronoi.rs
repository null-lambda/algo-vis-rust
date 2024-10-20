pub mod cmp {
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
        let result = p1 + dir1 * (p2 - p1).cross(dir2) * denom.recip();
        (result[0].is_finite() && result[1].is_finite()).then(|| result)
    }

    pub fn circumcenter(p: Point<f64>, q: Point<f64>, r: Point<f64>) -> Option<Point<f64>> {
        line_intersection((p + q) * 0.5, (q - p).rot(), (p + r) * 0.5, (r - p).rot())
    }

    // predicate 2 for voronoi diagram
    pub fn breakpoint_x(left: Point<f64>, right: Point<f64>, sweepline: f64) -> f64 {
        let y1 = left[1] - sweepline;
        let y2 = right[1] - sweepline;
        let xm = (left[0] + right[0]) * 0.5;
        let dx_half = (right[0] - left[0]) * 0.5;

        let a = y2 - y1;
        let b_2 = (y1 + y2) * dx_half;
        let c = y1 * y2 * (y1 - y2) - (y1 - y2) * dx_half * dx_half;

        let det_4 = b_2 * b_2 - a * c;
        let sign = a.signum() * (y2 - y1).signum();
        let mut x = xm + (-b_2 - sign * det_4.max(0.0).sqrt()) / a;
        if !x.is_finite() {
            x = xm - c * 0.5 / b_2;
            if !x.is_finite() {
                x = xm + dx_half;
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

    #[derive(Debug, Copy, Clone)]
    pub struct Vertex {
        pub half_edge: usize,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct HalfEdge {
        pub vert: usize,
        pub face_left: usize,
        pub flip: usize,
    }

    #[derive(Debug, Copy, Clone)]
    pub struct Face {
        half_edge: usize,
    }

    #[derive(Debug)]
    pub struct Topology {
        pub verts: Vec<Vertex>,
        pub half_edges: Vec<HalfEdge>,
        pub faces: Vec<Face>,
    }

    impl Topology {
        pub fn flipped(&mut self, idx_half_edge: usize) -> usize {
            let half_edge = &mut self.half_edges[idx_half_edge];
            half_edge.flip
        }
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
    pub mod splay {
        use crate::utils::console_log;

        // bottom-up splay tree for beachline
        use super::super::tag;

        use std::{
            cmp::Ordering,
            fmt, iter,
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

        type Link = Option<NonNull<Node>>;

        #[derive(Debug)]
        pub struct Breakpoint {
            pub site: usize,
            pub left_half_edge: usize,
        }

        pub struct Node {
            tag: tag::Tag,
            pub children: [Link; 2], // binary search tree structure
            pub side: [Link; 2],     // linked list structure
            pub parent: Link,
            pub value: Breakpoint,
        }

        #[derive(Debug)]
        pub struct Tree {
            pub root: Link,
        }

        impl Node {
            pub fn new(value: Breakpoint) -> Self {
                Self {
                    tag: tag::next(),
                    children: [None, None],
                    side: [None, None],
                    parent: None,
                    value,
                }
            }

            pub fn new_nonnull(value: Breakpoint) -> NonNull<Self> {
                unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(Self::new(value)))) }
            }

            fn link_sides(mut lhs: NonNull<Self>, mut rhs: NonNull<Self>) {
                unsafe {
                    lhs.as_mut().side[Right as usize] = Some(rhs);
                    rhs.as_mut().side[Left as usize] = Some(lhs);
                }
            }

            fn attach(&mut self, branch: Branch, child: Option<NonNull<Self>>) {
                unsafe {
                    debug_assert_ne!(Some(self as *mut _), child.map(|x| x.as_ptr()));
                    self.children[branch as usize] = child;
                    if let Some(mut child) = child {
                        child.as_mut().parent = Some(self.into());
                    }
                }
            }

            fn detach(&mut self, branch: Branch) -> Option<NonNull<Self>> {
                unsafe {
                    self.children[branch as usize].take().map(|mut child| {
                        child.as_mut().parent = None;
                        child
                    })
                }
            }

            fn is_root(&self) -> bool {
                self.parent.is_none()
            }

            fn branch(node: NonNull<Self>) -> Option<(Branch, NonNull<Node>)> {
                unsafe {
                    node.as_ref().parent.map(|parent| {
                        let branch = match parent.as_ref().children[Branch::Left as usize] {
                            Some(child) if ptr::eq(node.as_ptr(), child.as_ptr()) => Branch::Left,
                            _ => Branch::Right,
                        };
                        (branch, parent)
                    })
                }
            }

            fn rotate(mut node: NonNull<Self>) -> Option<()> {
                unsafe {
                    let (branch, mut parent) = Node::branch(node)?;

                    let child = node.as_mut().detach(branch.inv());
                    parent.as_mut().attach(branch, child);

                    if let Some((grandbranch, mut grandparent)) = Node::branch(parent) {
                        grandparent.as_mut().attach(grandbranch, Some(node));
                    } else {
                        node.as_mut().parent = None;
                    }
                    node.as_mut().attach(branch.inv(), Some(parent));

                    Some(())
                }
                // println!("after rotate:\n{:?}", self);
            }

            pub fn validate_parents(node: NonNull<Self>) {
                unsafe {
                    if let Some((branch, parent)) = Node::branch(node) {
                        debug_assert_eq!(
                            node.as_ptr(),
                            parent.as_ref().children[branch as usize].unwrap().as_ptr(),
                            "Parent's child pointer does not point to self"
                        );
                    }
                    for branch in Branch::iter() {
                        if let Some(child) = node.as_ref().children[branch as usize] {
                            debug_assert_eq!(
                                node.as_ptr(),
                                child.as_ref().parent.unwrap().as_ptr(),
                                "Child's parent pointer does not point to self: {:?} {:?}",
                                node,
                                child
                            );
                            debug_assert_ne!(child.as_ptr(), node.as_ptr(), "Self loop detected");
                        }
                    }
                }
            }

            pub fn splay(root: &mut NonNull<Node>, node: NonNull<Node>) {
                while let Some((branch, parent)) = Node::branch(node) {
                    if let Some((grandbranch, _)) = Node::branch(parent) {
                        if branch == grandbranch {
                            Node::rotate(parent);
                        } else {
                            Node::rotate(node);
                        }
                    }
                    Node::rotate(node);
                }
                *root = node;
            }

            // splay the last truthy element in [true, ..., true, false, ..., false]
            // if there is no such element, return false
            pub fn splay_last_by<F>(root: &mut NonNull<Node>, pred: F) -> bool
            where
                F: Fn(&Node) -> bool,
            {
                unsafe {
                    let mut current = *root;
                    let mut last_pred;
                    loop {
                        last_pred = pred(current.as_ref());
                        let branch = if last_pred { Right } else { Left };
                        let Some(child) = current.as_ref().children[branch as usize] else {
                            break;
                        };
                        current = child;
                    }
                    if !last_pred {
                        let Some(left) = current.as_ref().side[Left as usize] else {
                            return false;
                        };
                        current = left;
                    }
                    debug_assert!(
                        pred(current.as_ref())
                            && current.as_ref().children[Right as usize]
                                .map_or(true, |x| !pred(x.as_ref()))
                    );
                    Node::splay(root, current);
                    true
                }
            }

            pub fn splay_by<F>(root: &mut NonNull<Node>, cmp: F)
            where
                F: Fn(&Node) -> Ordering,
            {
                unsafe {
                    let mut current = *root;
                    loop {
                        let branch = match cmp(current.as_ref()) {
                            Ordering::Less => Left,
                            Ordering::Greater => Right,
                            Ordering::Equal => break,
                        };
                        let Some(child) = current.as_ref().children[branch as usize] else {
                            break;
                        };
                        current = child;
                    }
                    Node::splay(root, current);
                }
            }

            pub fn splay_first(root: &mut NonNull<Node>) {
                Node::splay_by(root, |_| Ordering::Less);
            }

            pub fn splay_last(root: &mut NonNull<Node>) {
                Node::splay_by(root, |_| Ordering::Greater);
            }

            pub fn insert_right(mut root: NonNull<Node>, mut new_node: NonNull<Node>) {
                unsafe {
                    // console_log!("before insert_right: {:?}", root.as_ref());

                    let right = root.as_mut().children[Right as usize];
                    if right.is_some() {
                        let next = root.as_ref().side[Right as usize].unwrap();
                        Node::link_sides(new_node, next);
                    }
                    Node::link_sides(root, new_node);

                    root.as_mut().attach(Right, Some(new_node));
                    new_node.as_mut().attach(Right, right);

                    // console_log!("after insert_right: {:?}", root.as_ref());
                }
            }

            pub fn pop_root(root: &mut Option<NonNull<Node>>) -> Option<NonNull<Node>> {
                unsafe {
                    let mut old_root = (*root)?;

                    let left = old_root.as_mut().detach(Left);
                    let right = old_root.as_mut().detach(Right);

                    match (left, right) {
                        (Some(_), Some(_)) => {
                            let prev = old_root.as_ref().side[Left as usize].unwrap();
                            let next = old_root.as_ref().side[Right as usize].unwrap();
                            Node::link_sides(prev, next);

                            let mut left = left.unwrap();
                            // Node::splay_last(&mut left);
                            Node::splay(&mut left, prev);
                            debug_assert!(left.as_ref().children[Right as usize].is_none());

                            left.as_mut().attach(Right, right);
                            *root = Some(left);
                        }
                        (Some(_), None) => {
                            let mut left = left.unwrap();
                            Node::splay_last(&mut left);
                            left.as_mut().side[Right as usize] = None;
                            *root = Some(left);
                        }
                        (None, Some(_)) => {
                            let mut right = right.unwrap();
                            Node::splay_first(&mut right);
                            right.as_mut().side[Left as usize] = None;
                            *root = Some(right);
                        }
                        (None, None) => {
                            *root = None;
                        }
                    }
                    old_root.as_mut().side = [None, None];
                    Some(old_root)
                }
            }

            // Test whether the linked list structure is valid
            pub fn validate_side_links(root: NonNull<Node>) {
                if !cfg!(debug_assertions) {
                    return;
                }

                unsafe {
                    // grab all nodes in inorder
                    let mut inorder = vec![];
                    Node::inorder(root, |node| inorder.push(node));

                    // check the linked list structure
                    for i in 0..inorder.len() {
                        let node: NonNull<Node> = inorder[i];
                        let prev = (i >= 1).then(|| inorder[i - 1]);
                        let next = (i + 1 < inorder.len()).then(|| inorder[i + 1]);

                        fn option_nonnull_to_ptr<T>(x: Option<NonNull<T>>) -> *const T {
                            x.map_or(ptr::null(), |x| x.as_ptr())
                        }

                        debug_assert!(ptr::eq(
                            option_nonnull_to_ptr(node.as_ref().side[Left as usize]),
                            option_nonnull_to_ptr(prev)
                        ));
                        debug_assert!(
                            ptr::eq(
                                option_nonnull_to_ptr(node.as_ref().side[Right as usize]),
                                option_nonnull_to_ptr(next),
                            ),
                            "side_right: {:?}, inorder_right: {:?}",
                            node.as_ref().side[Right as usize],
                            next
                        );
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

            pub fn inorder<F>(root: NonNull<Node>, mut visitor: F)
            where
                F: FnMut(NonNull<Node>),
            {
                pub fn inner<F>(node: NonNull<Node>, visitor: &mut F)
                where
                    F: FnMut(NonNull<Node>),
                {
                    unsafe {
                        if let Some(left) = node.as_ref().children[Left as usize] {
                            inner(left, visitor);
                        }
                        visitor(node);
                        if let Some(right) = node.as_ref().children[Right as usize] {
                            inner(right, visitor);
                        }
                    }
                }

                inner(root, &mut visitor);
            }
        }

        impl fmt::Debug for Node {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                unsafe {
                    write!(
                        f,
                        "{} {:?} {}",
                        self.children[Left as usize]
                            .as_ref()
                            .map_or("_".to_owned(), |x| format!("({:?})", x.as_ref())),
                        self.tag,
                        self.children[Right as usize]
                            .as_ref()
                            .map_or("_".to_owned(), |x| format!("({:?})", x.as_ref())),
                    )
                }
            }
        }

        //#[test]
        //fn test_splay_tree() {
        //    //
        //    let gen_breakpoint = || {
        //        Node::new_nonnull(Breakpoint {
        //            site: 0,
        //            left_half_edge: 0,
        //        })
        //    };
        //    let mut tree = gen_breakpoint();

        //    unsafe {
        //        for branch in [Left, Left, Right, Left, Left, Right, Right] {
        //            // tree.insert(branch, gen_breakpoint());
        //            Node::insert_right(tree, gen_breakpoint());
        //            Node::validate_side_links(tree);
        //            println!("insert {branch:?}, {:?}", tree.as_ref());
        //        }
        //        for i in 0..7 {
        //            if i % 3 == 0 {
        //                Node::splay_first(&mut tree);
        //                Node::validate_side_links(tree);
        //                Node::validate_parents(tree);
        //                println!("splay first, {:?}", tree.as_ref());
        //            } else {
        //                Node::splay_last(&mut tree);
        //                Node::validate_side_links(tree);
        //                Node::validate_parents(tree);
        //                println!("splay last, {:?}", tree.as_ref());
        //            }

        //            let mut tree_wrapped = Some(tree);
        //            Node::pop_root(&mut tree_wrapped);
        //            tree = tree_wrapped.unwrap();
        //            Node::validate_side_links(tree);
        //            println!("remove root, {:?}", tree.as_ref());
        //        }
        //    }
        //}
    }

    use core::f64;
    use std::{
        cmp::Reverse,
        collections::{BinaryHeap, HashSet},
        fmt,
        hash::Hash,
        ptr::{self, NonNull},
    };

    use splay::{Branch, Node};

    use crate::utils::console_log;

    use super::{
        cmp::Trivial,
        geometry::{self, Ordered, Point, PointNd},
    };

    use super::graph::{self, HalfEdge};

    #[derive(Debug, Clone)]
    pub enum Event {
        Site(Site),
        Circle(Circle),
    }

    #[derive(Debug, Clone)]
    pub struct Site {
        pub idx: usize,
    }

    #[derive(Clone)]
    pub struct Circle {
        pub node: NonNull<Node>,
        pub prev_idx: usize,
        pub next_idx: usize,
    }

    impl fmt::Debug for Circle {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            unsafe {
                write!(
                    f,
                    "Circle {{ node: {:?}, left: {:?}, right: {:?} }}",
                    self.node.as_ref(),
                    self.prev_idx,
                    self.next_idx,
                )
            }
        }
    }

    #[derive(Debug)]
    pub struct Builder {
        pub events: BinaryHeap<(Reverse<(Ordered<f64>, Ordered<f64>)>, Trivial<Event>)>,
        pub added_circles: HashSet<[usize; 3]>, // double-checker for robustness
        pub beachline: NonNull<splay::Node>,
        pub directrix: Ordered<f64>,
        pub graph: graph::Graph,
        pub beachline_node_pool: Vec<Box<Node>>,
        _init: bool,
    }

    fn allocate_node(pool: &mut Vec<Box<Node>>, value: splay::Breakpoint) -> NonNull<Node> {
        pool.push(Box::new(Node::new(value)));
        unsafe { NonNull::new_unchecked(pool.last_mut().unwrap().as_mut()) }
    }

    fn new_breakpoint_node(
        pool: &mut Vec<Box<Node>>,
        graph: &mut graph::Graph,
        sites: [usize; 2],
    ) -> NonNull<Node> {
        let idx_halfedge = graph.topo.half_edges.len();
        graph.topo.half_edges.push(HalfEdge {
            vert: graph::INF,
            face_left: sites[0],
            flip: idx_halfedge + 1,
        });
        graph.topo.half_edges.push(HalfEdge {
            vert: graph::UNSET,
            face_left: sites[1],
            flip: idx_halfedge,
        });
        let value = splay::Breakpoint {
            site: sites[1],
            left_half_edge: idx_halfedge,
        };
        allocate_node(pool, value)
    }

    fn eval_left_breakpoint_x(
        graph: &graph::Graph,
        sweep: Ordered<f64>,
        node: &Node,
    ) -> Ordered<f64> {
        let left = node.side[Branch::Left as usize];
        let left_x: Ordered<f64> = left
            .map_or(f64::NEG_INFINITY, |left| unsafe {
                geometry::breakpoint_x(
                    graph.face_center[left.as_ref().value.site],
                    graph.face_center[node.value.site],
                    sweep.into(),
                )
            })
            .into();
        left_x
    }

    fn new_circle_event(
        node: NonNull<Node>,
        graph: &graph::Graph,
        added_circles: &mut HashSet<[usize; 3]>,
    ) -> Option<(Reverse<(Ordered<f64>, Ordered<f64>)>, Trivial<Event>)> {
        unsafe {
            let prev = node.as_ref().side[Branch::Left as usize]?;
            let next = node.as_ref().side[Branch::Right as usize]?;

            let mut indices = [
                node.as_ref().value.site,
                prev.as_ref().value.site,
                next.as_ref().value.site,
            ];

            let p0 = graph.face_center[indices[0]];
            let p1 = graph.face_center[indices[1]];
            let p2 = graph.face_center[indices[2]];

            if geometry::signed_area(p0, p1, p2) >= 1e-9 {
                return None;
            }
            console_log!("{:?}", added_circles);
            console_log!("{:?}", indices);

            // indices.sort();
            // if !added_circles.insert(indices) {
            //     return None;
            // }

            console_log!("circumcenter: {:?}", geometry::circumcenter(p0, p1, p2));

            let center = geometry::circumcenter(p0, p1, p2)?;
            let radius = (center - p0).norm_sq().sqrt();
            let y = center[1] + radius;
            let x = center[0];

            let event = Circle {
                node,
                prev_idx: prev.as_ref().value.site,
                next_idx: next.as_ref().value.site,
            };
            Some((Reverse((y.into(), x.into())), Trivial(Event::Circle(event))))
        }
    }

    pub fn check_circle_event(circle: &Circle) -> bool {
        unsafe {
            let Some(prev) = circle.node.as_ref().side[Branch::Left as usize] else {
                return false;
            };
            let Some(next) = circle.node.as_ref().side[Branch::Right as usize] else {
                return false;
            };
            prev.as_ref().value.site == circle.prev_idx
                && next.as_ref().value.site == circle.next_idx
        }
    }

    impl Builder {
        pub fn new() -> Self {
            Self {
                events: BinaryHeap::new(),
                added_circles: HashSet::new(),
                beachline: NonNull::dangling(),
                beachline_node_pool: vec![],
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
            debug_assert!(!self._init, "No modification after init");
            for p in points {
                self.graph.face_center.push(p);

                let idx = self.graph.face_center.len() - 1;
                self.events.push((
                    Reverse((p[1].into(), p[0].into())),
                    Trivial(Event::Site(Site { idx })),
                ));
            }
        }

        pub fn init(&mut self) {
            if self._init {
                return;
            }
            self._init = true;

            if self.events.is_empty() {
                return;
            }

            // create root arc (no breakpoint),
            // which has x = -infty (see eval_breakpoint_x)
            let Event::Site(Site { idx }) = self.events.pop().unwrap().1 .0 else {
                unreachable!()
            };

            self.beachline = splay::Node::new_nonnull(splay::Breakpoint {
                site: idx,
                left_half_edge: graph::UNSET,
            });
        }

        pub fn step(&mut self) -> bool {
            debug_assert!(
                self._init,
                "Builder::init must be called before Builder::step"
            );

            loop {
                let Some((Reverse((py, px)), Trivial(event))) = self.events.pop() else {
                    return false;
                };
                self.directrix = py;
                match event {
                    Event::Site(Site { idx: idx_site }) => {
                        console_log!("processing site event {:?}", idx_site);

                        let PointNd([px, _py]) = self.graph.face_center[idx_site];

                        Node::splay_last_by(&mut self.beachline, |node| {
                            eval_left_breakpoint_x(&self.graph, self.directrix, node)
                                <= Ordered::from(px)
                        });

                        let old_site = unsafe { self.beachline.as_ref().value.site };

                        let left = self.beachline;
                        let mid = new_breakpoint_node(
                            &mut self.beachline_node_pool,
                            &mut self.graph,
                            [old_site, idx_site],
                        );
                        let right = new_breakpoint_node(
                            &mut self.beachline_node_pool,
                            &mut self.graph,
                            [idx_site, old_site],
                        );

                        Node::insert_right(self.beachline, right);
                        Node::insert_right(self.beachline, mid);

                        self.events.extend(new_circle_event(
                            left,
                            &self.graph,
                            &mut self.added_circles,
                        ));
                        self.events.extend(new_circle_event(
                            right,
                            &self.graph,
                            &mut self.added_circles,
                        ));
                    }
                    Event::Circle(circle) => unsafe {
                        console_log!("processing circle event: {:?}", circle);

                        if !check_circle_event(&circle) {
                            continue;
                        };

                        let Circle { node, .. } = circle;
                        Node::splay(&mut self.beachline, node);

                        let next = node.as_ref().side[Branch::Right as usize].unwrap();
                        let prev = node.as_ref().side[Branch::Left as usize].unwrap();

                        console_log!("before pop {:?}", self.beachline.as_ref());

                        let mut beachline: Option<NonNull<Node>> = Some(self.beachline);
                        Node::pop_root(&mut beachline);
                        let Some(beachline) = beachline else {
                            return false;
                        };
                        self.beachline = beachline;

                        console_log!("after pop {:?}", self.beachline.as_ref());
                        Node::validate_side_links(self.beachline);

                        let vert_idx = self.graph.topo.verts.len();
                        self.graph
                            .vert_coord
                            .push(Point::new(px.into(), py.into()).into());

                        let he1 = self.graph.topo.half_edges[node.as_ref().value.left_half_edge];
                        let he2 = self.graph.topo.half_edges[next.as_ref().value.left_half_edge];
                        self.graph.topo.half_edges[he1.flip].vert = vert_idx;
                        self.graph.topo.half_edges[he2.flip].vert = vert_idx;

                        let he3_idx = self.graph.topo.half_edges.len();
                        self.graph.topo.half_edges.push(HalfEdge {
                            vert: vert_idx,
                            face_left: prev.as_ref().value.site,
                            flip: he3_idx + 1,
                        });

                        self.graph.topo.half_edges.push(HalfEdge {
                            vert: graph::UNSET,
                            face_left: next.as_ref().value.site,
                            flip: he3_idx,
                        });

                        self.events.extend(
                            new_circle_event(prev, &self.graph, &mut self.added_circles)
                                .into_iter(),
                        );
                        self.events.extend(
                            new_circle_event(next, &self.graph, &mut self.added_circles)
                                .into_iter(),
                        );
                    },
                }
                Node::validate_parents(self.beachline);
                Node::validate_side_links(self.beachline);

                println!("{:?}", self.beachline);
                return true;
            }
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
