function rng(x)
    randn('seed',x) % generates randomly dist from an std normal dist
    rand('seed',x) % generates randomly from a unif dist 

end