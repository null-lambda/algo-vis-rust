export const throttle = (f, limit) => {
    let in_throttle;
    return (...args) => {
        if (!in_throttle) {
            f(...args);
            in_throttle = true;
            setTimeout(() => in_throttle = false, limit);
        }
    };
};

export const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

let seed = 42;
export const rand_next = () => {
    seed = ((seed * 166455) + 10139043) % 0xffffffff;
    return seed;
};
export const rand_range = (a, b) =>  a + rand_next() % (b - a + 1);
