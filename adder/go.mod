module github.com/ml/adder

go 1.22.3

replace github.com/ml/neuralnetwork => ../nn

replace github.com/ml/matrix => ../matrix

require (
	github.com/ml/neuralnetwork v0.0.0-00010101000000-000000000000
	github.com/ml/visualization v0.0.0-00010101000000-000000000000
)

require (
	github.com/ebitengine/purego v0.7.1 // indirect
	github.com/gen2brain/raylib-go/raylib v0.0.0-20240628125141-62016ee92fc0
	github.com/ml/matrix v0.0.0-00010101000000-000000000000
	golang.org/x/exp v0.0.0-20240613232115-7f521ea00fb8 // indirect
	golang.org/x/sys v0.21.0 // indirect
)

replace github.com/ml/visualization => ../visualization
