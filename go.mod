module github.com/barqi/ml

go 1.22.3

replace github.com/ml/matrix => ./matrix

require github.com/ml/neuralnetwork v0.0.0-00010101000000-000000000000

require github.com/ml/matrix v0.0.0-00010101000000-000000000000 // indirect

replace github.com/ml/neuralnetwork => ./nn
