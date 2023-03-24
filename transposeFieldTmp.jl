using DelimitedFiles, DataFrames, CSV, Random
using Plots

PATH = pwd() * "/Results/Data/"
filenames = readdir(PATH * "train/");
fnCh = filenames[contains.(filenames, "Chemical")];

for file in fnCh
    ch = Array(transpose(reshape(readdlm(PATH * "train/" * file ), 256,256)))
    writedlm(PATH * "train/" * file, reshape(ch, :, 1))
end

PATH = pwd() * "/Results/Data/"
filenames = readdir(PATH * "val/");
fnCh = filenames[contains.(filenames, "Chemical")];

for file in fnCh
    ch = Array(transpose(reshape(readdlm(PATH * "val/" * file ), 256,256)))
    writedlm(PATH * "val/" * file, reshape(ch, :, 1))
end