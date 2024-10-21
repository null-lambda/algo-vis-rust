export const throttle = (f, limit) => {
    let inThrottle;
    return (...args) => {
        if (!inThrottle) {
            f(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
};

let seed = 42;
export const rand_next = () => {
    seed = ((seed * 166455) + 10139043) % 0xffffffff;
    return seed;
};
export const rand_range = (a, b) => {
    return a + rand_next() % (b - a + 1);
};
